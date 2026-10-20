"""Actual patterned slab: static reciprocal fields vs polarization-current solve."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,hashlib
import numpy as np
from sptc_toolbox.reciprocal_scattering import ScatteringFields,first_order_conversion

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'sptc_toolbox/results/targeted_resolution/ports'
def read_fields(path):
    d=np.load(path);x0,y0,z0=[d[k] for k in ('x','y','z')]
    ix=np.flatnonzero((x0>=-.5)&(x0<.5));iy=np.flatnonzero((y0>=-.5)&(y0<.5));iz=np.flatnonzero(abs(z0)<.21)
    axes=(x0[ix],y0[iy],z0[iz])
    field=np.stack([d[k][np.ix_(ix,iy,iz)]/d['incident'] for k in ('Ex','Ey','Ez')],axis=-1)
    return axes,field

def main():
    source=ROOT/'sptc_toolbox/results/connected_physics/ports/weak_pattern_r24_time'
    axes,carrier=read_fields(source/'carrier.npz');x,y,z=axes
    xx,yy=np.meshgrid(x,y,indexing='ij')
    mask=~(((xx+.22)**2+yy*yy<.12**2)|(((xx-.22)/.12)**2+(yy/.09)**2<1))
    mask=np.broadcast_to(mask[:,:,None],carrier.shape[:-1]);dv=np.prod([np.mean(np.diff(a)) for a in axes])
    weights=np.full(mask.size,dv)
    grid=hashlib.sha256(b''.join(a.tobytes() for a in axes)).hexdigest();geometry='square-slab-circle-ellipse-eps11.5-d.42'
    meta=dict(normalization='unit_incident_time_averaged_power',reference_planes=[-.7,.7],source='Actual Meep static plane-wave driven fields',
        normalization_from_cache='E/incident multiplied by sqrt(2) for air unit area, time-averaged power convention')
    incident=ScatteringFields(2*np.pi*.398,np.sqrt(2)*carrier.reshape(1,-1,3),('bottom',),grid,geometry,meta)
    reference=json.loads((source/'summary.json').read_text());rows=[];sources={}
    if reference['resolution']!=24 or not np.isclose(reference['carrier_fa_over_c'],.398):
        raise ValueError('Carrier field cache frequency/resolution changed')
    for h,name in [(-1,'reciprocal_f0388_r24'),(1,'reciprocal_f0408_r24')]:
        path=OUT/name/'carrier.npz';test_axes,lower=read_fields(path)
        test_meta=json.loads((path.parent/'summary.json').read_text())
        if test_meta['resolution']!=24 or not np.isclose(test_meta['carrier_fa_over_c'],.398+h*.01) or test_meta['geometry']!='same circle/ellipse slab':
            raise ValueError('Reciprocal field cache geometry/frequency/resolution differs')
        for a,b in zip(axes,test_axes):np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(z,-z[::-1],atol=1e-14)
        # Actual z mirror maps the opposite inward port, not a Hermitian
        # conjugation of the forward mode. E is a polar vector under reflection.
        upper=lower[:,:,::-1,:]*np.array([1.,1.,-1.])
        reciprocal=ScatteringFields(2*np.pi*(.398+h*.01),np.sqrt(2)*np.stack([lower.reshape(-1,3),upper.reshape(-1,3)]),
            ('bottom','top'),grid,geometry,meta)
        result=first_order_conversion(incident,reciprocal,.005*mask.ravel(),weights)
        expected=next(r for r in reference['sidebands'] if r['harmonic']==h)
        direct=np.array([complex(*expected['lower_amplitude']),complex(*expected['upper_amplitude'])])
        actual=result['amplitudes'][:,0]
        rows.append(dict(harmonic=h,reciprocity_amplitudes=np.stack([actual.real,actual.imag],axis=-1).tolist(),
            polarization_current_amplitudes=np.stack([direct.real,direct.imag],axis=-1).tolist(),
            reciprocal_total_power=float(np.sum(abs(actual)**2)),direct_total_power=float(np.sum(abs(direct)**2)),
            relative_complex_amplitude_difference=float(np.linalg.norm(actual-direct)/np.linalg.norm(direct)),
            reciprocal_static_power_balance=json.loads((path.parent/'summary.json').read_text())['carrier_power_balance']))
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    sources[str((source/'carrier.npz').relative_to(ROOT))]=hashlib.sha256((source/'carrier.npz').read_bytes()).hexdigest()
    report=dict(status='ACTUAL_STATIC_SCATTERING_FIELD_RECIPROCITY_CHECK',rows=rows,
        formula='deltaS=i omega_out/4 integral E_test . deltaepsilon . E_in, no complex conjugate, unit incoming power',
        source_sha256=sources,quadrature='Full unit cell x/y endpoint-excluded voxel samples; interior slab z samples; physical high-epsilon mask',
        scope='Independent first-order response extraction from two static fields; Gamma, orthogonal zeroth-order x-polarized air ports, actual z mirror; no native-mode residue fit',
        limits=['Fields and masks are finite-grid interpolated; interface smoothing derivative is not identical to a sharp mask',
            'Both references share a static full-wave geometry; numerical agreement does not prove arbitrary Floquet completeness',
            'Weak modulation only; no carrier depletion or repeated conversion'])
    (OUT/'reciprocity_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()
