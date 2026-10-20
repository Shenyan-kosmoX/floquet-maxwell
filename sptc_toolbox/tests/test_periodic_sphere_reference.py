import numpy as np
import pytest
from contextlib import ExitStack
from unittest.mock import patch
from numpy.testing import assert_allclose
from sptc_toolbox.sphere_material import SphereMaterial,LorentzSpherePole
from sptc_toolbox.periodic_sphere_reference import scatter_sphere_array,sphere_matrix,incident_spherical,outgoing_plane_map
from toolbox_experiments.nphoton_2024_01563 import sphere_lattice_smin as lattice


@pytest.mark.parametrize('sign',[1,-1])
@pytest.mark.parametrize('direction',[1,-1])
@pytest.mark.parametrize('polarization',[0,1,'mixed'])
def test_signed_outgoing_flux_from_native_cartesian_fields(sign,direction,polarization):
    # E comes from native Cartesian vector plane waves. H is derived directly
    # from curl(E)=i*omega*H, without calling the adapter's power calculation.
    w=1.1;kp=np.array([.23,-.17]);q=np.sqrt(w*w-kp@kp)
    chart_k=np.r_[sign*kp,direction*q]
    electric_basis=np.array([
        fn(*chart_k,.37,-.29,.61)
        for fn in (lattice.treams.special.vpw_M,lattice.treams.special.vpw_N)])
    if sign<0:electric_basis=electric_basis.conj()
    physical_k=sign*chart_k
    if polarization=='mixed':
        # Exercise the actual outgoing map with coherent TE/TM amplitudes.
        rng=np.random.default_rng(721)
        coefficients=rng.normal(size=16)+1j*rng.normal(size=16)
        amplitude=outgoing_plane_map(sign*w,kp,2,49.,direction)@coefficients
        assert np.all(abs(amplitude)>1e-3)
    else:
        amplitude=np.eye(2,dtype=complex)[polarization]
    electric=amplitude@electric_basis
    magnetic=np.cross(physical_k,electric)/(sign*w)
    flux=.5*np.cross(electric,magnetic.conj()).real
    expected=.5*direction*q/w*np.vdot(amplitude,amplitude).real
    assert_allclose(flux[2],expected,atol=2e-15,rtol=2e-14)
    assert_allclose(physical_k@electric,0.,atol=2e-15)


@pytest.mark.parametrize('polarization',[0,1])
def test_lossy_oblique_static_array_matches_native_smatrix_and_poynting(polarization):
    # Independent high-level route: native Mie sphere, native lattice solve,
    # native S-matrix conversion, and native Poynting normalization. The
    # low-level calls below adapt only C-long dispatch for the Windows wheel.
    tr=lattice.treams
    old_translate=tr.sw.translate_periodic
    old_outgoing=tr.sw.periodic_to_pw
    old_incident=tr.pw.to_sw
    def integer(value):return np.asarray(value,dtype='l')
    def translate(ks,kpar,a,rs,out,in_=None,rsin=None,poltype=None,eta=0,func=None):
        out=tuple(integer(v) for v in out)
        in_=None if in_ is None else tuple(integer(v) for v in in_)
        return old_translate(ks,np.asarray(kpar,float),np.asarray(a,float),rs,
                             out,in_,rsin,poltype,eta,lattice._long_lattice_sum)
    def outgoing(kx,ky,kz,pol,l,m,qol,area,poltype=None,*args,**kwargs):
        return old_outgoing(kx,ky,kz,integer(pol),integer(l),integer(m),
                            integer(qol),area,poltype,*args,**kwargs)
    def incident(l,m,polsw,kx,ky,kz,polpw,poltype=None,*args,**kwargs):
        return old_incident(integer(l),integer(m),integer(polsw),kx,ky,kz,
                            integer(polpw),poltype,*args,**kwargs)
    omega=.613;period=11.93;kp=np.array([.04,-.03]);lmax=3
    geometry=tr.Lattice(np.eye(2)*period)
    basis=tr.PlaneWaveBasisByComp.diffr_orders(kp,geometry,omega+np.linalg.norm(kp))
    assert len(basis)==10  # Five propagating orders, each with two polarizations.
    epsilon=1+11/(1-omega**2-1j*.125*omega)
    tm=tr.TMatrix.sphere(lmax,omega,2.,[epsilon,1.],poltype='parity')
    with ExitStack() as stack:
        for module,name,adapter in [(tr.sw,'translate_periodic',translate),
                                    (tr.sw,'periodic_to_pw',outgoing),
                                    (tr.pw,'to_sw',incident)]:
            stack.enter_context(patch.object(module,name,adapter))
        coupled=tm.latticeinteraction.solve(geometry,kp)
        sm=tr.SMatrices.from_array(coupled,basis)
        illumination=np.zeros(len(basis),complex)
        selected=(abs(basis.kx-kp[0])<1e-12)&(abs(basis.ky-kp[1])<1e-12)&(basis.pol==polarization)
        assert np.count_nonzero(selected)==1
        illumination[selected]=1
        native_t,native_r=sm.tr(illumination)
        native_up,native_down=sm.illuminate(illumination)
    assert tr.sw.translate_periodic is old_translate
    assert tr.sw.periodic_to_pw is old_outgoing
    assert tr.pw.to_sw is old_incident
    material=SphereMaterial(1.,(LorentzSpherePole(np.sqrt(11),1.,.125),))
    result=scatter_sphere_array(material,omega,.1,radius=2.,period=period,
                                n_harmonics=0,lmax=lmax,kparallel=kp,
                                polarization=polarization)
    assert_allclose([result['transmissivity'],result['reflectivity']],
                    [native_t,native_r],atol=2e-13,rtol=2e-13)
    assert 0<result['net_exchange']<1
    assert result['componentwise_linear_residual']<1e-12
    for channel in result['channels']:
        transverse=kp+np.asarray(channel['G'])*2*np.pi/period
        native=np.asarray(native_up if channel['direction']==1 else native_down)
        expected=[]
        for pol in (0,1):
            selected=(abs(basis.kx-transverse[0])<1e-12)&(abs(basis.ky-transverse[1])<1e-12)&(basis.pol==pol)
            assert np.count_nonzero(selected)==1
            expected.append(native[selected][0])
        actual=np.asarray(channel['amplitude_real'])+1j*np.asarray(channel['amplitude_imag'])
        assert_allclose(actual,expected,atol=2e-13,rtol=2e-13)


@pytest.mark.parametrize('omega',[.173,1.013])
@pytest.mark.parametrize('polarization',[0,1])
def test_lossless_static_periodic_array_conserves_all_open_order_flux(omega,polarization):
    result=scatter_sphere_array(SphereMaterial(4.),omega,.1,radius=2.,period=7.,n_harmonics=0,lmax=3,polarization=polarization)
    assert_allclose(result['reflectivity']+result['transmissivity'],1.,rtol=2e-11,atol=2e-11)


def test_vacuum_is_direct_unit_transmission_for_both_polarizations():
    for polarization in (0,1):
        result=scatter_sphere_array(SphereMaterial(1.),.1813,.1,radius=4.27,period=11.93,n_harmonics=3,lmax=2,polarization=polarization)
        assert result['reflectivity']<1e-25
        assert_allclose(result['transmissivity'],1.,atol=1e-13)


def test_zero_modulation_floquet_replicas_do_not_add_spurious_scattering():
    material=SphereMaterial(1.,(LorentzSpherePole(np.sqrt(11),1.,.125),))
    a=scatter_sphere_array(material,.1813,.1,radius=4.27,period=11.93,n_harmonics=0,lmax=3)
    b=scatter_sphere_array(material,.1813,.1,radius=4.27,period=11.93,n_harmonics=5,lmax=3)
    for key in ['reflectivity','transmissivity']:assert_allclose(a[key],b[key],atol=2e-13)
    assert a['net_exchange']>0


def test_common_pump_phase_preserves_power_but_transforms_sideband_amplitudes():
    def material(phase):return SphereMaterial(1.,(LorentzSpherePole(np.sqrt(11),1.,.125,{0:1,1:.25*np.exp(-1j*phase),-1:.25*np.exp(1j*phase)}),))
    args=dict(radius=4.27,period=11.93,n_harmonics=5,lmax=2)
    a=scatter_sphere_array(material(0),.1813,.1,**args);b=scatter_sphere_array(material(.43),.1813,.1,**args)
    assert_allclose([a['reflectivity'],a['transmissivity']],[b['reflectivity'],b['transmissivity']],atol=1e-12)
    for ca,cb in zip(a['channels'],b['channels']):
        aa=np.array(ca['amplitude_real'])+1j*np.array(ca['amplitude_imag'])
        ab=np.array(cb['amplitude_real'])+1j*np.array(cb['amplitude_imag'])
        assert_allclose(ab,aa*np.exp(-1j*ca['harmonic']*.43),atol=1e-12)


def test_real_field_frequency_collisions_and_zero_sidebands_are_rejected():
    for omega in [.15,.2]:
        with pytest.raises(ValueError,match='frequency|Frequency'):
            scatter_sphere_array(SphereMaterial(4.),omega,.1,radius=2.,period=7.,n_harmonics=3,lmax=1)
