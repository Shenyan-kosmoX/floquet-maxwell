"""Validate study-specific native cache contracts before convergence analysis."""
from pathlib import Path
import hashlib,json,pickle
import numpy as np
from sptc_toolbox.baselines.minkov2018_legume_dynamic import ROOT,OUT,Q,save_json

GEOMETRY=dict(epsilon=12.1,d=.55,r=.25,Lx=1,Ly=7*np.sqrt(3))


def require_guided(fields,tolerance=1e-10):
    gamma=np.asarray(fields['gamma'])
    if np.iscomplexobj(gamma) or not np.isfinite(gamma).all() or np.any(gamma<0):
        raise ValueError('Native radiation must be finite, real and nonnegative')
    if np.max(gamma)>tolerance:
        raise ValueError('Nonzero native radiation exceeds the guided-only model scope')


def load_checked(directory,*,nk,gmax,folds=3,z_order=10,momentum_chart='wrapped'):
    directory=Path(directory);path=directory/'native_fields.npz'
    contract=json.loads((directory/'basis_contract.json').read_text())
    for key,value in [('nk',nk),('gmax',gmax),('fold_halfwidth',folds),('z_order',z_order),('gmode_inds',[0,3])]:
        if contract.get(key)!=value:raise ValueError(f'Native cache numerical contract differs for{key}')
    if contract.get('geometry')!=GEOMETRY:raise ValueError('Native cache physical geometry differs')
    if momentum_chart not in ('wrapped','extended') or contract.get('momentum_chart','wrapped')!=momentum_chart:
        raise ValueError('Native Bloch coordinate chart differs from requested numerical basis')
    digest=hashlib.sha256(path.read_bytes()).hexdigest();source_digest=hashlib.sha256((directory/'basis_contract.json').read_bytes()).hexdigest()
    provenance_dir=ROOT/'sptc_toolbox/results/convergence/w1/cache_checks';provenance_dir.mkdir(parents=True,exist_ok=True)
    name=hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:16];proof=provenance_dir/(name+'.json')
    authority=None
    if contract.get('field_arrays_sha256'):
        if digest!=contract['field_arrays_sha256']:raise ValueError('Native field content differs from export contract')
        authority='Export-time field digest'
    elif proof.exists():
        prior=json.loads(proof.read_text())
        if digest!=prior['fields_sha256'] or source_digest!=prior['contract_sha256']:
            raise ValueError('Native cache changed after independent content validation')
        authority='Previously independently validated legacy content'
    elif directory.resolve()==(OUT/'main_g3.1_k32_f3').resolve():
        historical=json.loads((OUT/'delivery_manifest.json').read_text())
        if digest!=historical['data_sha256']['native_fields.npz']:raise ValueError('Historical delivered native field digest changed')
        authority='Historical delivery manifest, independently hash-checked'
    fields=dict(np.load(path,allow_pickle=False))
    q=np.pi+fields['k_super'][:,None]+fields['sectors'][None,:]*Q
    if momentum_chart=='wrapped':q=(q+np.pi)%(2*np.pi)-np.pi
    if len(fields['k_super'])!=nk or not np.array_equal(fields['sectors'],np.arange(-folds,folds+1)) or not np.allclose(fields['q_primitive'],q,rtol=0,atol=1e-12):
        raise ValueError('Native cache momentum coordinates disagree with controls')
    if any(not np.isfinite(a).all() for a in fields.values()):raise ValueError('Nonfinite native field cache')
    if fields['omega'].shape!=(nk,2*folds+1) or fields['gamma'].shape!=fields['omega'].shape or np.any(fields['omega']<=0):
        raise ValueError('Native spectral array shapes or frequency domain differ')
    if fields['E'].shape!=(nk,2*folds+1,z_order,3,fields['gvec'].shape[1]) or fields['E_mid'].shape!=(nk,2*folds+1,3,fields['gvec'].shape[1]):
        raise ValueError('Native field array layout differs')
    if authority is None:
        # Old caches without an export digest are bound only after an actual
        # reconstruction from their hash-verified native solver object.
        native=directory/'native.pkl'
        if hashlib.sha256(native.read_bytes()).hexdigest()!=contract['native_sha256']:
            raise ValueError('Legacy native object digest differs')
        with native.open('rb') as stream:gme=pickle.load(stream)
        if not np.array_equal(gme.gvec,fields['gvec']) or not np.allclose(gme.kpoints[0],q.ravel(),atol=1e-12,rtol=0):
            raise ValueError('Legacy native coordinates do not match field cache')
        maximum=0.
        for index,(mode,omega) in enumerate(zip(fields['mode_indices'].ravel(),fields['omega'].ravel())):
            ik,j=divmod(index,2*folds+1)
            if abs(2*np.pi*gme.freqs[index,int(mode)]-omega)>1e-12:raise ValueError('Legacy native frequency differs')
            mid=np.asarray(gme.ft_field_xy('E',index,int(mode),.275))
            maximum=max(maximum,float(np.max(abs(mid-fields['E_mid'][ik,j]))))
            # Validate every saved quadrature field, not a sample of entries.
            for iz,z in enumerate(fields['z']):
                value=np.asarray(gme.ft_field_xy('E',index,int(mode),float(z)))
                maximum=max(maximum,float(np.max(abs(value-fields['E'][ik,j,iz]))))
        if maximum>1e-12:raise ValueError('Legacy fields differ from native re-export')
        authority='All cached E/z and E/midplane fields independently re-exported from verified native object'
    require_guided(fields)
    save_json(proof,dict(directory=str(directory.resolve()),fields_sha256=digest,contract_sha256=source_digest,
        authority=authority,historical_source_hashes=contract.get('source_hashes',{}),
        source_version_policy='Historical hashes retained; current source equality is not falsely asserted',
        geometry=GEOMETRY,numerical_controls=dict(nk=nk,gmax=gmax,folds=folds,z_order=z_order,gmode_inds=[0,3],momentum_chart=momentum_chart)))
    return fields,dict(fields_sha256=digest,cache_validation=str(proof.relative_to(ROOT)),authority=authority)
