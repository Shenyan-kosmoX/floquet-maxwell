"""Bounded native Meep finite-Q eigfrequency attempt with log-based acceptance.

Run under the existing WSL Meep environment. The supervisor launches a separate
worker with a hard 120-second timeout per attempt and preserves raw native logs.
Only a native log with no CW failures and a converged outer frequency change
permits one tighter independent-seed repeat. A returned frequency alone is not
accepted. This changes no existing runner or installed solver.
"""
from __future__ import annotations
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='2'
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def worker(args):
    import meep as mp
    from scipy.interpolate import RegularGridInterpolator
    from meep_cutoff_probe import Config,make_simulation,sampling_grid
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    height=1/(3.2*.703)
    kka=2*np.pi-2*np.pi/(height*np.sqrt(15))-.04
    cfg=Config(kx_ka=kka,resolution=24,padding=1.,pml=.5,frequency=args.guess_real)
    start=time.perf_counter()
    metadata={'status':'RUNNING_NATIVE_EIGFREQ','config':asdict(cfg),
              'guessfreq':[args.guess_real,args.guess_imag], 'tolerance':args.tolerance,
              'maxiters':15,'cwtol':args.cwtol,'cwmaxiters':300,'L':10,
              'native_meep_version':mp.__version__,
              'units':'cyclic f*a/c; k input radians k*a converted to native k/(2pi)',
              'symmetries':'MirrorY phase -1, MirrorZ phase +1',
              'scope':'Finite-Q independent static native frequency-domain attempt, not Floquet'}
    write(output/'worker_status.json',metadata)
    mp.verbosity(1)
    try:
        sim=make_simulation(cfg)
        sim.sources=[mp.Source(mp.ContinuousSource(frequency=args.guess_real),component=mp.Hz,
                                center=mp.Vector3(x,0.,0.),amplitude=amplitude)
                     for x,amplitude in [(0.,1.),(.17,.43)]]
        sim.init_sim()
        pole=sim.solve_eigfreq(tol=args.tolerance,maxiters=15,
             guessfreq=complex(args.guess_real,args.guess_imag),cwtol=args.cwtol,cwmaxiters=300,L=10)
        metadata.update(status='RETURNED_UNVALIDATED_PENDING_LOG_REVIEW',pole=[float(pole.real),float(pole.imag)],
                        solve_seconds=time.perf_counter()-start)
        write(output/'worker_status.json',metadata)
        size=mp.Vector3(1.,1.,2*cfg.field_extent)
        native_axes=sim.get_array_metadata(center=mp.Vector3(),size=size)
        arrays={c:sim.get_array(center=mp.Vector3(),size=size,component=getattr(mp,c),cmplx=True)
                for c in ('Ex','Ey','Ez','Hx','Hy','Hz')}
        np.savez_compressed(output/'native_eigenfields_returned.npz',**arrays,
                            x=native_axes[0],y=native_axes[1],z=native_axes[2],weights=native_axes[3])
        x,y,z,grid,metric=sampling_grid(cfg)
        sampled=np.stack([RegularGridInterpolator(native_axes[:3],arrays[c],bounds_error=True)(grid.reshape(-1,3)).reshape(grid.shape[:-1])
                          for c in ('Hx','Hy','Hz')],axis=-1)
        periodic=sampled*np.exp(-1j*kka*grid[...,0])[...,None]
        np.savez_compressed(output/'common_H_returned.npz',H=sampled,H_periodic=periodic,
                            H_weighted_periodic=(periodic*np.sqrt(metric)[...,None]).reshape(-1),
                            x=x,y=y,z=z,weights=metric,pole=complex(pole),kpoint_ka=[kka,0.])
        metadata['elapsed_seconds']=time.perf_counter()-start
        metadata['field_status']='Native returned field, not accepted until log and repeat checks pass'
        write(output/'worker_status.json',metadata)
        sim.reset_meep()
        print(json.dumps(metadata,indent=2),flush=True)
    except Exception as exc:
        metadata.update(status='NATIVE_EIGFREQ_EXCEPTION',exception_type=type(exc).__name__,
                        exception_message=str(exc),elapsed_seconds=time.perf_counter()-start)
        write(output/'worker_status.json',metadata)
        raise


def inspect_attempt(folder,returncode,timed_out,seconds):
    logfile=folder/'native.log'
    log=logfile.read_text(encoding='utf-8',errors='replace')
    worker_path=folder/'worker_status.json'
    worker_record=json.loads(worker_path.read_text()) if worker_path.exists() else {}
    failures=re.findall(r'CONVERGENCE FAILURE[^\n]*',log)
    pattern=r'Eigensolver step\s+(\d+):\s+(\d+) CG iters, freq = ([+\-\d.eE]+)([+\-][\d.eE]+)i \(change = ([+\-\d.eE]+)([+\-][\d.eE]+)i\)'
    outer=[]
    for match in re.finditer(pattern,log):
        n,it,re_f,im_f,re_d,im_d=match.groups()
        outer.append({'step':int(n),'CG_iters':int(it),'frequency':[float(re_f),float(im_f)],
                      'change':[float(re_d),float(im_d)]})
    tolerance=worker_record.get('tolerance',1e-7)
    if outer:
        last=outer[-1];freq=complex(*last['frequency']);delta=complex(*last['change'])
        relative=abs(delta)/max(abs(freq),np.finfo(float).tiny)
    else:relative=None
    returned=worker_record.get('status')=='RETURNED_UNVALIDATED_PENDING_LOG_REVIEW'
    native_ok=not timed_out and returncode==0 and returned and not failures and relative is not None and relative<=tolerance
    status=('NATIVE_LOG_CONVERGENCE_PASSED_PENDING_REPEAT' if native_ok else
            'FAILED_INNER_CW_CONVERGENCE' if failures else 'NOT_ACCEPTED_NATIVE_EIGFREQ')
    record={'status':status,'returncode':returncode,'process_timeout':timed_out,'process_timeout_seconds':120,
            'wall_seconds':seconds,'worker':worker_record,'CW_failure_messages':failures,
            'outer_steps':outer,'last_outer_relative_frequency_change':relative,
            'all_inner_solves_and_outer_frequency_check_passed':native_ok,
            'raw_log':str(logfile),'fields_are_accepted':False,
            'reason':('Log criteria passed; still requires independent repeat/field validation' if native_ok else
                      'Native inner CW reported convergence failure; stopped without accepting a pole or field' if failures else
                      'Timeout, missing return or unverified outer convergence; no returned field/frequency is promoted')}
    write(folder/'attempt_review.json',record)
    return record


def one_attempt(output,real,imag,tol,cwtol):
    output.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--output',str(output),
         '--guess-real',str(real),'--guess-imag',str(imag),'--tolerance',str(tol),'--cwtol',str(cwtol)]
    start=time.perf_counter();timed_out=False;returncode=None
    logfile=output/'native.log'
    with logfile.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
        while process.poll() is None:
            failed='CONVERGENCE FAILURE' in logfile.read_text(encoding='utf-8',errors='replace')
            timed_out=time.perf_counter()-start>=120
            if failed or timed_out:
                process.terminate()
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                stream.write('\nSUPERVISOR: '+('FAILED_INNER_CW_CONVERGENCE' if failed else 'PROCESS_TIMEOUT_AFTER_120_SECONDS')+'; native result not accepted.\n')
                break
            time.sleep(.25)
        returncode=process.wait()
    return inspect_attempt(output,returncode,timed_out,time.perf_counter()-start)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results/meep_cutoff_eigfreq')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--guess-real',type=float,default=.667)
    parser.add_argument('--guess-imag',type=float,default=-.009)
    parser.add_argument('--tolerance',type=float,default=1e-7)
    parser.add_argument('--cwtol',type=float,default=1e-8)
    args=parser.parse_args()
    if args.worker:return worker(args)
    args.output.mkdir(parents=True,exist_ok=True)
    first=one_attempt(args.output/'attempt1',args.guess_real,args.guess_imag,args.tolerance,args.cwtol)
    attempts=[first];comparison=None
    print(json.dumps({'attempt':1,'status':first['status'],'timed_out':first['process_timeout'],
                      'CW_failure_count':len(first['CW_failure_messages'])}),flush=True)
    if first['all_inner_solves_and_outer_frequency_check_passed']:
        pole=complex(*first['worker']['pole'])
        second=one_attempt(args.output/'attempt2',pole.real+1e-4,pole.imag*.9,3e-8,3e-9)
        attempts.append(second)
        if second['all_inner_solves_and_outer_frequency_check_passed']:
            pole2=complex(*second['worker']['pole'])
            a=np.load(args.output/'attempt1/common_H_returned.npz')['H_weighted_periodic']
            b=np.load(args.output/'attempt2/common_H_returned.npz')['H_weighted_periodic']
            overlap=float(abs(np.vdot(a,b))**2/(np.vdot(a,a).real*np.vdot(b,b).real))
            comparison={'complex_frequency_absolute_difference':abs(pole2-pole),'field_overlap_squared':overlap,
                        'repeat_stable':abs(pole2-pole)<2e-6 and overlap>.99}
    passed=bool(comparison and comparison['repeat_stable'])
    summary={'status':'NATIVE_FINITE_Q_RETURN_AND_REPEAT_VERIFIED_DISCRETIZATION_ONLY' if passed else 'BOUNDED_NATIVE_FINITE_Q_ATTEMPT_NOT_ACCEPTED',
             'attempts':attempts,'repeat_comparison':comparison,'max_attempts':2,'max_native_seconds_per_attempt':120,
             'accepted_as_independent_QNM_reference':False,
             'limits':['No frequency or field is accepted on the return value alone','Grid/PML convergence is separate',
                       'A stable native eigenfield is not residue-normalized by this script','No forced identification with the Gamma parent or the two strongly observed Harminv branches']}
    write(args.output/'summary.json',summary)
    lines=['# 有限Q截止邻域 native solve_eigfreq 有界试验','',f"状态：**{summary['status']}**。",'',
           '固定原圆柱ε16、r/a=.3125、h/a=1/(3.2*.703)，kx a=2π−2π/(h√15)−.04，r24，padding1/PML.5，MirrorY−1/MirrorZ+1。初guessfreq=.667−.009i，不是Γ BIC。','',
           '每次原生worker进程硬限120秒，tol1e−7/maxiters15、cwtol1e−8/cwmaxiters300/L10。只有所有nativeCW日志无失败且外层频率变化通过才允许第二次更严格复核。','']
    for i,a in enumerate(attempts,1):
        lines.append(f"- attempt {i}: {a['status']}; wall={a['wall_seconds']:.2f}s, timeout={a['process_timeout']}, native CW failures={len(a['CW_failure_messages'])}, outer relative change={a['last_outer_relative_frequency_change']}.")
    lines+=['','原始日志、worker异常/状态及自动日志审查JSON均保存。若失败，没有把返回值或残留场转化为合格极点。成功重复也只验证固定离散模型，空间/PML收敛及全局模式身份仍独立验收。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'attempt_count':len(attempts),'repeat':comparison},indent=2),flush=True)


if __name__=='__main__':main()
