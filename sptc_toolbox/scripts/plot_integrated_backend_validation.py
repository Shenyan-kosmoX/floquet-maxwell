"""Plot saved physical slab scattering and local complex-pole controls."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'sptc_toolbox/results/integrated_backend'


def main():
    sfile=ROOT/'sptc_toolbox/results/floquet_slab/scattering_sweep.npz'
    qfile=ROOT/'sptc_toolbox/results/floquet_slab_qnm/summary.json'
    data=np.load(sfile);q=json.loads(qfile.read_text());rows=q['results'][1:]
    f=data['incident_frequency'];h=data['harmonic_indexes']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                         'svg.fonttype':'none','pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(9,6.5))
    fig.subplots_adjust(left=.085,right=.975,bottom=.17,top=.86,wspace=.32,hspace=.48)
    axs[0,0].plot(f,data['static_total_output_input_power_ratio'],color='#657080',label='Static, absorbing slab')
    axs[0,0].plot(f,data['total_output_input_power_ratio'],color='#0072B2',label='Time-modulated slab')
    axs[0,0].axhline(1.,color='k',ls=':',lw=.7)
    axs[0,0].set(xlabel=r'Incident frequency / $\Omega$',ylabel='Total output / input power',title='a  Physical two-port response')
    axs[0,0].legend(frameon=False,fontsize=8)
    idx=int(np.argmin(abs(f-.37)));amplitude=data['dynamic_output_E'][idx]
    for y,label,color in zip(abs(amplitude)**2,['Reflected','Transmitted'],['#D55E00','#0072B2']):
        axs[0,1].semilogy(h,y,'o-',label=label,color=color,ms=3)
    axs[0,1].set(xlabel='Output harmonic n',ylabel='Channel power / input power',
                 title=f'b  Sidebands at incident {f[idx]:.3f}'+r'$\Omega$')
    axs[0,1].legend(frameon=False,fontsize=8)
    ns=np.array([r['N'] for r in rows]);freq=np.array([complex(*r['frequency']) for r in rows])
    axs[1,0].semilogy(ns[:-1],abs(freq[:-1].real-freq[-1].real),'o-',color='#009E73',ms=4,label='Real part')
    axs[1,0].semilogy(ns[:-1],abs(freq[:-1].imag-freq[-1].imag),'s--',color='#CC79A7',ms=4,label='Imaginary part')
    axs[1,0].set(xlabel='Harmonic truncation N',ylabel=r'Difference from N=6 pole / $\Omega$',title='c  Complex outgoing-pole convergence',xticks=ns[:-1])
    axs[1,0].legend(frameon=False,fontsize=8)
    for n,color in [(4,'#D55E00'),(6,'#0072B2')]:
        p=np.load(ROOT/f'sptc_toolbox/results/floquet_slab_qnm/pole_N{n}.npz')
        axs[1,1].loglog(p['residue_steps'],p['residue_errors'],'o-',label=f'N={n}',color=color)
    axs[1,1].set(xlabel=r'Distance from pole $|\delta\omega|/\Omega$',ylabel='Relative Laurent-residue error',
                 title='d  Scattering residue from left/right modes')
    axs[1,1].legend(frameon=False,fontsize=8)
    for ax in axs.flat:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Open dispersive Floquet slab: scattering, sidebands and QNM residue',fontsize=12,y=.975)
    fig.text(.5,.92,'Independent 1D Maxwell boundary reference; spatially homogeneous slab, normal incidence',ha='center',fontsize=9)
    fig.text(.5,.035,'Finite harmonic model; local pole tracking does not establish a complete spectrum.\n'
             'Half-pump coherent-power controls are saved separately; this sweep retains its original power mask.',ha='center',fontsize=8,color='#505c67')
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'physical_validation.{ext}',dpi=230)
    (OUT/'figure_manifest.json').write_text(json.dumps(dict(source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),sfile,qfile]},
        scope='Saved independent physical slab controls, no invented model parameters or interpolated evidence'),indent=2)+'\n')


if __name__=='__main__':main()
