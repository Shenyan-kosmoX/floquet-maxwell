"""Convergence studies may not silently reuse wrong or radiative field data."""
import hashlib,json
import numpy as np
import pytest
from sptc_toolbox.scripts.w1_convergence_data import load_checked,require_guided,GEOMETRY
from sptc_toolbox.baselines.minkov2018_legume_dynamic import Q


def cache(tmp_path):
    p=tmp_path/'native_fields.npz';k=np.array([-.05,.05]);sectors=np.arange(-1,2)
    q=(np.pi+k[:,None]+Q*sectors[None,:]+np.pi)%(2*np.pi)-np.pi
    np.savez(p,k_super=k,sectors=sectors,q_primitive=q,omega=np.ones((2,3)),gamma=np.zeros((2,3)),
        E=np.ones((2,3,2,3,1),complex),E_mid=np.ones((2,3,3,1),complex),gvec=np.zeros((2,1)))
    c=dict(nk=2,gmax=3.1,fold_halfwidth=1,z_order=2,gmode_inds=[0,3],geometry=GEOMETRY,
           field_arrays_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (tmp_path/'basis_contract.json').write_text(json.dumps(c));return c


def test_cached_physical_geometry_and_guided_families_are_checked(tmp_path):
    c=cache(tmp_path);c['geometry']={**c['geometry'],'epsilon':12.2}
    (tmp_path/'basis_contract.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='physical geometry'):load_checked(tmp_path,nk=2,gmax=3.1,folds=1,z_order=2)
    c['geometry']=GEOMETRY;c['gmode_inds']=[0]
    (tmp_path/'basis_contract.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='gmode_inds'):load_checked(tmp_path,nk=2,gmax=3.1,folds=1,z_order=2)


def test_field_content_change_cannot_hide_behind_unchanged_controls(tmp_path):
    cache(tmp_path)
    p=tmp_path/'native_fields.npz';d=dict(np.load(p));d['E'][0,0,0,0,0]+=1.;np.savez(p,**d)
    with pytest.raises(ValueError,match='content differs'):load_checked(tmp_path,nk=2,gmax=3.1,folds=1,z_order=2)


@pytest.mark.parametrize('value',[np.nan,np.inf,-.1,.0003])
def test_nonfinite_or_open_radiation_cannot_enter_guided_convergence(value):
    with pytest.raises(ValueError):require_guided({'gamma':np.array([0.,value])})
