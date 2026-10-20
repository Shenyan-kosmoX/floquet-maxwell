"""Independently anchor dipole phases in the paper's actual field convention."""
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import lpmv, spherical_jn, spherical_yn

from sptc_toolbox.baselines.garg_dipole_observables import carrier_dipoles
from sptc_toolbox.periodic_sphere_reference import (
    basis_modes, outgoing_plane_map, scatter_sphere_array, treams)
from sptc_toolbox.sphere_material import LorentzSpherePole, SphereMaterial


def _si_s7(ell, m, x, theta, phi):
    # Direct SI S7 expressions, independent of native angular/VSW functions.
    p = lpmv(m, ell, np.cos(theta))
    previous = lpmv(m, ell-1, np.cos(theta)) if abs(m) <= ell-1 else 0.
    pi_function = m*p/np.sin(theta)
    tau_function = (ell*np.cos(theta)*p-(ell+m)*previous)/np.sin(theta)
    gamma = 1j*np.sqrt((2*ell+1)/(4*np.pi*ell*(ell+1))
                      *math.factorial(ell-m)/math.factorial(ell+m))
    azimuth = np.exp(1j*m*phi)
    angular_x = gamma*azimuth*np.array([0., 1j*pi_function, -tau_function])
    radial_cross_x = np.array([0., -angular_x[2], angular_x[1]])
    h = spherical_jn(ell, x)+1j*spherical_yn(ell, x)
    derivative = spherical_jn(ell, x, derivative=True)+1j*spherical_yn(ell, x, derivative=True)
    magnetic = h*angular_x
    electric = (derivative+h/x)*radial_cross_x
    electric[0] += gamma*ell*(ell+1)*p*azimuth*h/x
    return magnetic, electric


@pytest.mark.parametrize('ell,m', [(1,-1), (1,0), (1,1), (2,-2), (2,1), (3,-2)])
def test_native_vsw_matches_independently_evaluated_paper_si_s7(ell, m):
    for x, theta, phi in [(1.73, .61, -.47), (3.4, 2.1, 1.1), (.91, 1.3, 2.2)]:
        magnetic, electric = _si_s7(ell, m, x, theta, phi)
        assert_allclose(treams.special.vsw_M(np.int32(ell), np.int32(m), x, theta, phi), magnetic,
                        rtol=2e-13, atol=2e-13)
        assert_allclose(treams.special.vsw_N(np.int32(ell), np.int32(m), x, theta, phi), electric,
                        rtol=2e-13, atol=2e-13)


def test_negative_pi_kerker_condition_is_derived_from_outgoing_fields():
    modes = basis_modes(1)
    coefficients = np.zeros(len(modes), complex)
    # For normal upward TE input, m=+/-1 are linked by mirror symmetry.
    # Construct only dipoles satisfying the paper's AMD/AED=-1 at m=-1.
    for mode, value in [((1,-1,0), -1), ((1,-1,1), 1),
                        ((1,1,0), 1), ((1,1,1), 1)]:
        coefficients[np.all(modes == mode, axis=1)] = value
    backward = outgoing_plane_map(.1813, (0,0), 1, 11.93**2, -1)@coefficients
    forward = outgoing_plane_map(.1813, (0,0), 1, 11.93**2, 1)@coefficients
    assert_allclose(backward, 0., atol=2e-14)
    assert np.linalg.norm(forward) > .1
    dipoles = carrier_dipoles(coefficients, modes, [.1813], .1813)
    assert_allclose(dipoles['amplitude_ratio'], 1.)
    assert_allclose(dipoles['phase_radians'], -np.pi)
    assert dipoles['relative_phase_correction_radians'] == 0


def test_static_actual_array_mirror_and_incident_polarization_do_not_require_phase_fit():
    material = SphereMaterial(1., (LorentzSpherePole(np.sqrt(11), 1., .125),))
    extracted = []
    for polarization in (0, 1):
        result = scatter_sphere_array(material, .1813, .1, radius=4.27, period=11.93,
                                      n_harmonics=0, lmax=3, polarization=polarization)
        args = (result['scattered_coefficients'], result['basis_modes'],
                result['signed_frequencies'], .1813)
        negative_m = carrier_dipoles(*args)
        positive_m = carrier_dipoles(*args, magnetic_order=1)
        assert_allclose(negative_m['complex_ratio'], -positive_m['complex_ratio'], atol=2e-13)
        # A phase chart change cannot alter the actual complex ratio.
        assert_allclose(np.exp(1j*negative_m['phase_radians']),
                        negative_m['complex_ratio']/negative_m['amplitude_ratio'], atol=2e-14)
        assert result['reflectivity'] < .001
        assert abs(negative_m['amplitude_ratio']-1) < .1
        assert abs(negative_m['phase_radians']+np.pi) < .1
        extracted.append(negative_m['complex_ratio'])
    assert_allclose(extracted[0], extracted[1], atol=2e-13)


def test_carrier_index_and_vsw_row_order_are_explicit_not_inferred():
    modes = basis_modes(1)[::-1]
    coefficients = np.ones((3,len(modes)), complex)
    coefficients[2, np.all(modes == (1,-1,0), axis=1)] = -2j
    result = carrier_dipoles(coefficients, modes, [-.1, .2, .3], .3)
    assert result['carrier_index'] == 2
    assert result['amplitude_ratio'] == 2
    assert_allclose(result['phase_radians'], -np.pi/2)
    with pytest.raises(ValueError, match='Exactly one retained'):
        carrier_dipoles(coefficients, modes, [-.1, .2, .3], .4)
