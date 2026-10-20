"""Independent finite-wave reality, positive API and signed static checks."""
import numpy as np
import pytest
from unittest.mock import patch

import sphere_lattice_smin as lattice


@pytest.mark.parametrize('lmax',[1,2,3])
def test_signed_frequency_basis_relation_from_native_field_functions(lmax):
    sc=lattice.treams.special
    for ell in range(1,lmax+1):
        for m in range(-ell,ell+1):
            for pol,functions in [(0,[sc.vsw_M,sc.vsw_rM]),(1,[sc.vsw_N,sc.vsw_rN])]:
                phase=(-1.)**(ell+m+1+pol)
                for fn in functions:
                    positive=fn(np.int32(ell),np.int32(-m),complex(2.37),.71,.39)
                    negative=fn(np.int32(ell),np.int32(m),complex(-2.37),.71,.39)
                    np.testing.assert_allclose(negative,phase*positive.conj(),atol=1e-13,rtol=2e-13)


def test_signed_lattice_relation_from_finite_real_space_sum():
    modes=lattice.basis_modes(1);ell,m,pol=modes.T
    c=lattice.reality_transform(1)
    def direct(omega,kpar,cutoff):
        total=np.zeros((6,6),complex)
        for x in range(-cutoff,cutoff+1):
            for y in range(-cutoff,cutoff+1):
                if x==y==0:continue
                r=np.hypot(x,y);phi=np.arctan2(-y,-x)
                term=lattice.treams.sw.translate(ell[:,None],m[:,None],pol[:,None],
                    ell,m,pol,complex(omega*r),np.pi/2,phi,poltype='parity',singular=True)
                total+=term*np.exp(1j*np.dot(kpar,[x,y]))
        return total
    kpar=np.array([.37,.82])
    for cutoff in [1,2,4]:
        negative=direct(-2.37,kpar,cutoff)
        expected=c@direct(2.37,-kpar,cutoff).conj()@c
        np.testing.assert_allclose(negative,expected,atol=2e-13,rtol=2e-13)


def test_positive_matches_native_high_level_interaction_with_dtype_only_adapter():
    # The legacy wheel assumes C long, but NumPy2 Windows defaults to int64.
    # Only inside this assertion adapt the integer dispatch of the native
    # high-level call; the runtime helper never monkeypatches package globals.
    original=lattice.treams.sw.translate_periodic
    def compatible(ks,kpar,a,rs,out,in_=None,rsin=None,poltype=None,eta=0,func=None):
        out=tuple(np.asarray(v,dtype='l') for v in out)
        in_=None if in_ is None else tuple(np.asarray(v,dtype='l') for v in in_)
        return original(ks,np.asarray(kpar,float),np.asarray(a,float),rs,out,in_,rsin,
                        poltype,eta,lattice._long_lattice_sum)
    for omega,kpar in [(1.7,[.3,.4]),(2.4,[1.,1.]),(3.4,[2.1,.7])]:
        tmatrix=lattice.treams.TMatrix.sphere(1,omega,1/3,[11.68,1],poltype='parity')
        ours=lattice.compose_operator(tmatrix,lattice.lattice_coupling(omega,kpar))
        with patch.object(lattice.treams.sw,'translate_periodic',compatible):
            reference=np.asarray(tmatrix.latticeinteraction(lattice.treams.Lattice(np.eye(2)),np.asarray(kpar,float)))
        np.testing.assert_allclose(ours,reference,atol=2e-14,rtol=2e-14)
    assert lattice.treams.sw.translate_periodic is original


@pytest.mark.parametrize('omega',[1.7,2.4,-1.7,-2.4])
def test_ewald_split_invariance_in_real_frequency_domain(omega):
    reference=lattice.lattice_coupling(omega,[.37,.82],eta=0.)
    for eta in [1.,2.]:
        np.testing.assert_allclose(lattice.lattice_coupling(omega,[.37,.82],eta=eta),reference,atol=2e-12,rtol=2e-12)


def test_static_signed_smin_uses_independent_signed_sphere_boundary_solver():
    from floquet_sphere_tmatrix import sphere_tmatrix
    modes=lattice.basis_modes(1)
    for omega,kpar in [(1.7,np.array([.37,.82])),(2.4,np.array([2.1,.7]))]:
        positive=sphere_tmatrix(omega,3.,delta_epsilon=0.,n_harmonics=0)
        negative=sphere_tmatrix(-omega,3.,delta_epsilon=0.,n_harmonics=0)
        tp=np.diag([positive.blocks[int(pol)][0,0] for ell,m,pol in modes])
        tn=np.diag([negative.blocks[int(pol)][0,0] for ell,m,pol in modes])
        splus=lattice.raw_smin(tp,lattice.lattice_coupling(omega,kpar))
        sminus=lattice.raw_smin(tn,lattice.lattice_coupling(-omega,-kpar))
        np.testing.assert_allclose(sminus,splus,rtol=1e-12,atol=1e-12)


def test_harmonic_block_order_and_zero_limit_rejection():
    frequencies=np.array([-1.2,1.8,4.8]);matrix=lattice.harmonic_lattice_coupling(frequencies,[.4,.6])
    assert matrix.shape==(18,18)
    for index,omega in enumerate(frequencies):
        np.testing.assert_array_equal(matrix[6*index:6*index+6,6*index:6*index+6],lattice.lattice_coupling(omega,[.4,.6]))
    with pytest.raises(ValueError,match='static-limit'):
        lattice.lattice_coupling(0.,[.4,.6])
    with pytest.raises(ValueError,match='signed real'):
        lattice.lattice_coupling(1.-1e-6j,[.4,.6])
