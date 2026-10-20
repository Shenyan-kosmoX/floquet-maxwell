"""Actual outgoing complex Floquet poles/residues of a dispersive slab fixture."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.floquet_slab import FloquetSlab,SlabLorentzPole,solve_slab_scattering
from sptc_toolbox.qnm import simple_pole_residue
from ptc_fig4.static_modes import solve_nonlinear_mode
OUT=ROOT/'sptc_toolbox/results/floquet_slab_qnm'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    static=FloquetSlab({0:2.7},.72,(SlabLorentzPole(.6,1.7,.08),))
    slab=FloquetSlab({0:2.7,1:.11,-1:.11},.72,
        (SlabLorentzPole(.6,1.7,.08,{0:1.,1:.035,-1:.035}),))
    seed=solve_nonlinear_mode(lambda z:static.operator(z,0),2.6-1.1j,search_radius=1.5)
    folded_seed=seed.frequency-2
    unmodulated=solve_nonlinear_mode(lambda z:static.operator(z,3),folded_seed,search_radius=.2)
    assert abs(unmodulated.frequency-folded_seed)<1e-10
    rows=[];previous=None;previous_n=None
    for n in [1,2,3,4,5,6]:
        op=lambda z:slab.operator(z,n)
        candidate=solve_nonlinear_mode(op,folded_seed if n<=2 else previous.frequency,
                                      search_radius=.4,tolerance=1e-11)
        pole=simple_pole_residue(candidate,op,residual_tolerance=1e-9)
        rhs=slab.boundary(candidate.frequency,n)[1];scattering_residue=pole.residue@rhs
        h=np.array([1e-3,1e-4,1e-5])
        errs=[float(np.linalg.norm(step*solve_slab_scattering(slab,candidate.frequency+step,n).S-scattering_residue)/np.linalg.norm(scattering_residue)) for step in h]
        assert errs[-1]<1e-3 and errs[-1]<.02*errs[0]
        overlap=None;change=None
        if n>=3:
            before=np.zeros((2,2*n+1),complex)
            before[:,n-previous_n:n+previous_n+1]=previous.right.reshape(2,2*previous_n+1)
            after=candidate.right.reshape(2,2*n+1)
            overlap=float(abs(np.vdot(before,after))**2/(np.vdot(before,before).real*np.vdot(after,after).real))
            change=float(abs(candidate.frequency-previous.frequency))
        status='DIFFERENT_TRUNCATION_ROOT_TARGET_STATIC_HARMONIC_NOT_RETAINED' if n==1 else 'LOCAL_OUTGOING_FLOQUET_POLE'
        rows.append(dict(N=n,status=status,frequency=[candidate.frequency.real,candidate.frequency.imag],
            residual=candidate.relative_residual,relative_derivative_pairing=candidate.derivative_pairing_relative,
            residue_errors=errs,pole_change_from_previous=change,channel_amplitude_overlap_squared=overlap,
            identity_metric='Common outgoing air-channel coefficient vector, for branch comparison only; QNM normalization uses A derivative'))
        np.savez_compressed(OUT/f'pole_N{n}.npz',frequency=candidate.frequency,right=candidate.right,left=candidate.left,
                            inverse_operator_residue=pole.residue,scattering_residue=scattering_residue,
                            residue_steps=h,residue_errors=errs)
        if n>=2:previous=candidate;previous_n=n
    assert rows[-1]['pole_change_from_previous']<1e-6
    assert rows[-1]['channel_amplitude_overlap_squared']>.999
    def common_residue(n):
        with np.load(OUT/f'pole_N{n}.npz') as saved:
            h=np.arange(-n,n+1)
            indices=np.concatenate([side*len(h)+np.flatnonzero(abs(h)<=2) for side in range(2)])
            return saved['scattering_residue'][np.ix_(indices,indices)]
    reference=common_residue(6)
    for row in rows[1:]:
        row['common_channel_residue_relative_difference_from_N6']=float(np.linalg.norm(common_residue(row['N'])-reference)/np.linalg.norm(reference))
    assert rows[-2]['common_channel_residue_relative_difference_from_N6']<1e-7
    sources=[Path(__file__),ROOT/'sptc_toolbox/floquet_slab.py',ROOT/'sptc_toolbox/qnm.py',ROOT/'ptc_fig4/src/ptc_fig4/static_modes.py']
    report=dict(status='LOCAL_COMPLEX_FLOQUET_POLE_AND_SCATTERING_RESIDUE_VALIDATED_FOR_PHYSICAL_SLAB',
        slab_parameters=dict(epsilon_inf=2.7,epsilon_cosine_amplitude=.22,thickness=.72,wp=.6,wr=1.7,gamma=.08,strength_cosine_amplitude=.07,drive=1.),
        static_seed=[seed.frequency.real,seed.frequency.imag],zero_modulation_folded_pole_error=float(abs(unmodulated.frequency-folded_seed)),
        results=rows,scope='Local tracked pole and Laurent residue in an analytic 1D Maxwell Floquet slab; not a global spectrum or arbitrary photonic-crystal QNM adapter',
        limitations=['N1 does not retain the static parent at harmonic +2 and locates a different pole','No claim that every physical pole was found','Residue validity is local; a finite pole sum needs an explicit analytic background','Bulk material and temporal harmonic assumptions are the specified scalar slab fixture'],
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],last_two=rows[-2:]),indent=2))


if __name__=='__main__':main()
