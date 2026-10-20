"""Count and inspect every actual raw source/frequency/window record."""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/'source_column_manifest.json').read_text())
assert manifest['completed']
expected_f=np.array([.378,.388,.398,.408,.418])
reference=np.load(HERE/'incident_even/response_t2000.npz')
points=reference['points'];weights=reference['weights'];operator=str(reference['operator_id'])
rows=[]
for source in manifest['rows']:
    path=HERE/source['case']
    assert source['completed']
    snapshots=[]
    for t in (1000,1500,2000):
        data=np.load(path/f'response_t{t}.npz')
        assert np.array_equal(data['frequencies'],expected_f)
        assert np.array_equal(data['points'],points)
        assert np.array_equal(data['weights'],weights)
        assert str(data['operator_id'])==operator
        assert float(data['time'])==t
        assert float(data['dt'])==.5/32
        assert np.allclose(data['kappa'],2*np.sin(np.pi*expected_f*float(data['dt']))/float(data['dt']),rtol=0,atol=1e-14)
        assert data['electric'].shape==(5,32,32,26,3)
        assert data['displacement'].shape==(5,32,32,26,3)
        assert data['output'].shape==data['incoming'].shape==(5,4)
        for field in ('electric','displacement','output','incoming','source_fourier','normalization'):
            assert np.all(np.isfinite(data[field])),(source['case'],t,field)
        assert np.all(abs(data['source_fourier'])>0)
        actual_stride=int(data['dft_decimation']) if 'dft_decimation' in data else 1
        assert actual_stride==source['dft_decimation']
        if source['kind']=='template':
            assert str(data['basis_sha256'])==source['basis_sha256']
        snapshots.append(dict(time=t,frequencies=expected_f.tolist(),dft_decimation=actual_stride))
    rows.append(dict(source=source['case'],kind=source['kind'],snapshot_count=len(snapshots),snapshots=snapshots))
result=dict(status='RAW_DATASET_INVENTORY_AND_COORDINATE_UNIT_VALIDATION',completed=True,
    spatial_sources=len(rows),template_sources=sum(x['kind']=='template' for x in rows),incident_sources=sum(x['kind']=='incident' for x in rows),
    frequencies=expected_f.tolist(),actual_source_frequency_pairs=len(rows)*5,frequency_window_records=len(rows)*5*3,
    raw_npz_files=len(rows)*3,native_field_shape=[5,32,32,26,3],port_order=['bottom x','top x','bottom y','top y'],
    incoming_scope='Two x-polarized parity inputs; this does not include independently driven y-polarized incident channels.',
    units='fa/c, t*c/a, native Yee J=-i*kappa*Q, unit power incoming parity channel',
    operator_id=operator,rows=rows,
    limitation='These checks certify inventory, shape, coordinates, units and provenance. Raw finite-window responses with long-lived poles are not certified steady-state Green responses; tail extraction and physical validation are separate.')
(HERE/'raw_dataset_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('completed','spatial_sources','template_sources','incident_sources','actual_source_frequency_pairs','frequency_window_records','raw_npz_files')}))
