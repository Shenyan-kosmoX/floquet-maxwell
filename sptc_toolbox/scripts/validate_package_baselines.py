"""Recompute both literature finite-model baselines through ModePackage.

Read existing trusted native spatial caches; do not change historical results,
material choices, mode identities, rejected points or reproduction claims.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,pickle,json,hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.mode_package import ModePackage
from sptc_toolbox.open_system import solve_open_floquet
from lpor_71318.field_projection import weak_modulation_coefficient
from lpor_71318.modal_floquet import select_target_mode
OUT=ROOT/'sptc_toolbox/results/integrated_backend/baselines'


def load_cache(path):
    with path.open('rb') as f:return pickle.load(f)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lpr=[]
    for config in ['custom_g3.1_v4','custom_g5.1_v6','custom_g7.1_v8','average_g7.1_v8']:
        olddir=ROOT/'sptc_toolbox/results/baselines/lpr'/config
        old=json.loads((olddir/'summary.json').read_text());mode=old['mode']
        cache=ROOT/'lpor_71318/results/static_probe'/config/'gme_native.pkl'
        package=ModePackage.from_legume(load_cache(cache),[mode],source=str(cache.relative_to(ROOT)))
        drive=1e14*400e-9/299792458.
        static=package.open_model(drive);s0=solve_open_floquet(static,n_harmonics=3,n_samples=128)
        material=package.open_model(drive,amplitude=0.)
        # Target selection uses the positive carrier of the retained static mode.
        w=package.arrays['omega']/drive
        from lpor_71318.modal_floquet import InverseDielectricModel
        p=dict(eigvals_chi=package.arrays['eigvals_chi'],displacement_weights=package.arrays['displacement_weights'],
               electric_overlap=package.arrays['electric_overlap'],epsilon=package.metadata['epsilon'])
        mm=InverseDielectricModel(w,p['eigvals_chi'],p['displacement_weights'],epsilon=p['epsilon'])
        baseline=select_target_mode(mm,s0,0).frequency_over_Omega
        rows=np.loadtxt(olddir/'frequency_shift.csv',delimiter=',',skiprows=1)
        errors=[]
        for amp,shift,imag in rows:
            spectrum=solve_open_floquet(package.open_model(drive,amplitude=amp),n_harmonics=3,n_samples=128)
            value=select_target_mode(mm,spectrum,0).frequency_over_Omega
            errors.append(abs((value-baseline).real-shift))
        coefficient=weak_modulation_coefficient(w,p)
        assert max(errors)<1e-11 and abs(coefficient-old['C_second_order'])<1e-12
        lpr.append(dict(config=config,maximum_shift_difference=max(errors),C=coefficient,
            old_C=old['C_second_order'],status='UNCHANGED_FINITE_MODEL_BASELINE',
            cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest()))
    root=ROOT/'sptc_toolbox/results/baselines/science_adv/g5.1_v24_s7'
    gme=load_cache(root/'gme.pkl');old=json.loads((root/'R1_summary_N5_extra0.json').read_text())
    science=[]
    for row in old:
        ik=row['k_index'];package=ModePackage.from_legume(gme,row['modes'],kind=ik,source=str((root/'gme.pkl').relative_to(ROOT)))
        if not row['cases']:
            rejected=False
            try:package.open_model(2*np.pi*136e12*3296e-9/299792458.,allow_onshell_approximation=True,max_relative_spread=.08)
            except ValueError as exc:rejected=True;reason=str(exc)
            assert rejected,'Previously rejected physical scope was silently accepted'
            science.append(dict(k_index=ik,status='REJECTION_PRESERVED',reason=reason));continue
        cases={}
        for name,case in row['cases'].items():
            drive=2*np.pi*case['drive_thz']*1e12*3296e-9/299792458.
            model=package.open_model(drive,amplitude=case['absolute_epsilon_amplitude'],static_offset=case['DC_epsilon_shift'],
                                    phase=np.pi/2,allow_onshell_approximation=True,max_relative_spread=.08)
            spectrum=solve_open_floquet(model,n_harmonics=5,n_samples=256)
            actual=spectrum.eigenvalues[spectrum.physical_modes()]
            a=np.asarray(case['quasifrequencies']);expected=a[:,0]+1j*a[:,1]
            actual=np.exp(-2j*np.pi*actual);expected=np.exp(-2j*np.pi*expected)
            ri,ci=linear_sum_assignment(abs(actual[:,None]-expected[None,:]))
            error=float(np.max(abs(actual[ri]-expected[ci])))
            assert error<1e-9,(ik,name,error)
            cases[name]=dict(multiplier_difference=error)
        science.append(dict(k_index=ik,status='UNCHANGED_FINITE_MODEL_BASELINE',tracking_ambiguous=row['tracking_ambiguous'],cases=cases))
    result=dict(status='BOTH_EXISTING_PAPER_BASELINES_RECOMPUTED_THROUGH_MODEPACKAGE',lpr=lpr,science=science,
        scope='Interface regression on the same native spatial caches and finite radiation model; original LPR coefficient and Science full-path gaps remain unresolved',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],lpr_configs=len(lpr),science_points=len(science),
                         preserved_rejections=sum(s['status']=='REJECTION_PRESERVED' for s in science)),indent=2))


if __name__=='__main__':main()
