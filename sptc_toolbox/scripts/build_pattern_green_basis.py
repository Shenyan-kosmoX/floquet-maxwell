"""Build an actual Legume mode plus a static-response residual source basis.

The second real template is selected from static carrier fields alone; no
dynamic sideband target or fitted resonance enters this enrichment.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,pickle,hashlib
import numpy as np
from sptc_toolbox.field_chart import native_coefficients_to_physical

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'sptc_toolbox/results/patterned_green_20261016/basis'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    spatial=ROOT/'sptc_toolbox/results/connected_physics/spatial/g6.1_v01234567/native.pkl'
    carrier=ROOT/'sptc_toolbox/results/connected_physics/ports/weak_pattern_r24_time/carrier.npz'
    with spatial.open('rb') as stream:g=pickle.load(stream)
    d=np.load(carrier);x,y,z=[d[a] for a in ('x','y','z')]
    xx,yy,zz=np.meshgrid(x,y,z,indexing='ij')
    mask=(abs(zz)<.21)&~(((xx+.22)**2+yy**2<.12**2)|(((xx-.22)/.12)**2+(yy/.09)**2<1))
    unique=(xx>=-.5)&(xx<.5)&(yy>=-.5)&(yy<.5)
    selected=mask&unique;dv=np.prod([np.mean(np.diff(a)) for a in (x,y,z)])
    xy=np.array(np.meshgrid(x,y,indexing='ij')).reshape(2,-1)
    phase=np.exp(-1j*g.gvec.T@xy)
    native=np.stack([(native_coefficients_to_physical(g,'E',np.asarray(g.ft_field_xy('E',0,5,float(t+.21))))@phase)
        .T.reshape(len(x),len(y),3) for t in z],axis=2)
    def inner(a,b):return np.sum(a[selected].conj()*b[selected])*dv
    angle=.5*np.angle(np.sum(native[selected]**2)*dv)
    aligned=native*np.exp(-1j*angle)
    imaginary_fraction=float(np.sqrt(inner(aligned.imag,aligned.imag).real/inner(aligned,aligned).real))
    if imaginary_fraction>1e-8:raise ValueError('Native Gamma mode is not one real source template')
    q0=aligned.real/np.sqrt(inner(aligned.real,aligned.real).real)
    lower=np.stack([d[k]/d['incident'] for k in ('Ex','Ey','Ez')],axis=-1)
    # Each physical inward port has field sqrt(2)*lower per unit power.
    # Unit total symmetric source [1,1]/sqrt(2) therefore gives lower+upper.
    upper=lower[:,:,::-1,:]*np.array([1,1,-1])
    incident=lower+upper
    alpha=inner(q0,incident);residual=incident-q0*alpha
    candidates=np.stack([residual.real,residual.imag],axis=-1)
    weighted=candidates[selected].reshape(-1,2)*np.sqrt(dv)
    _,singular,vh=np.linalg.svd(weighted,full_matrices=False)
    combination=vh[0].real
    q1=np.einsum('...cr,r->...c',candidates,combination)/singular[0]
    full=np.stack([q0,q1],axis=-1)
    basis=full[selected]
    gram=np.einsum('scr,scq->rq',basis,basis)*dv
    if np.linalg.norm(gram-np.eye(2))>1e-10:raise ValueError('Source basis not orthonormal')
    coords=np.stack([xx[selected],yy[selected],zz[selected]],axis=-1)
    weights=np.full(len(coords),dv)
    grid_id=hashlib.sha256(coords.tobytes()+weights.tobytes()).hexdigest()
    basis_id=hashlib.sha256(basis.tobytes()+grid_id.encode()).hexdigest()
    def coverage(rank):
        projected=np.einsum('scr,sci->ri',basis[:,:,:rank],incident[selected][:,:,None])*dv
        rest=incident[selected]-np.einsum('scr,ri->sci',basis[:,:,:rank],projected)[:,:,0]
        return float(np.linalg.norm(rest)/np.linalg.norm(incident[selected]))
    np.savez_compressed(OUT/'source_basis.npz',x=x,y=y,z=z,Q_full=full,Q=basis,volume_weights=weights,
        selected=selected,coordinates=coords,grid_id=grid_id,basis_id=basis_id,
        geometry_id='square-circle-ellipse-eps11.5-thickness.42',static_incident_symmetric=incident[selected])
    report=dict(status='NATIVE_MODE_PLUS_STATIC_BACKGROUND_RESIDUAL_REAL_SOURCE_BASIS',
        native_modes=[5],native_frequency_fa_over_c=float(g.freqs[0,5]),rank=2,
        grid_resolution=24,spatial_samples=len(coords),grid_id=grid_id,basis_id=basis_id,
        geometry_id='square-circle-ellipse-eps11.5-thickness.42',
        native_phase_real_residual=imaginary_fraction,gram_error=float(np.linalg.norm(gram-np.eye(2))),
        source_template_conjugation_closed=True,
        rank1_static_symmetric_field_residual=coverage(1),rank2_static_symmetric_field_residual=coverage(2),
        residual_real_imag_singular_values=singular.tolist(),
        basis_sources=['Actual Legume mode5, real Gamma gauge','Leading real source shape from static carrier background residual'],
        source_interpolation='Unmasked physical fields on regular grid; interpolate components then apply physical dielectric mask at Yee source sites',
        primary_scope='Hybrid independent Green experiment, not pure Legume full scattering',
        no_dynamic_target_fit=True,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (spatial,carrier)})
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
