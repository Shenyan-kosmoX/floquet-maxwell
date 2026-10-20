"""Static Mie, primitive material ADE and covariance checks for sphere materials."""
from pathlib import Path
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp, simpson

from sptc_toolbox.sphere_material import (LorentzSpherePole, SphereMaterial,
                                         lorentz_sphere_tmatrix)
from toolbox_experiments.nphoton_2024_01563.floquet_sphere_tmatrix import sphere_tmatrix


def _garg(depth=0.):
    return SphereMaterial(1., (LorentzSpherePole(np.sqrt(11.), 1., .125,
        {0: 1., 1: depth/2, -1: depth/2}),))


@pytest.mark.parametrize("ell", [1, 2])
@pytest.mark.parametrize("carrier", [.17, .43, .87, 1.31])
def test_static_all_sidebands_match_treams_dispersive_mie(ell, carrier):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "ptc_fig4/literature/software/_probe"))
    import treams
    material = _garg()
    result = material.tmatrix(carrier, .1, radius=4.27, ell=ell, n_harmonics=2)
    for n, frequency in enumerate(result.signed_frequencies.real):
        positive = abs(frequency)
        epsilon = 1+11/(1-positive**2-.125j*positive)
        expected = treams.TMatrix.sphere(ell, positive, 4.27,
            [treams.Material(epsilon), treams.Material(1)], poltype="parity")
        for parity in [0, 1]:
            mask = (expected.basis.l == ell) & (expected.basis.pol == parity)
            coefficient = np.diag(np.asarray(expected))[mask][0]
            if frequency < 0:
                coefficient = coefficient.conjugate()
            assert_allclose(result.blocks[parity][n, n], coefficient, rtol=5e-11, atol=3e-12)
            assert np.max(abs(np.delete(result.blocks[parity][n], n))) < 3e-13
            # Each unmodulated passive partial-wave scattering eigenvalue.
            assert abs(1+2*result.blocks[parity][n, n]) <= 1+2e-11


def test_nondispersive_harmonics_reproduce_original_module_without_behavior_change():
    phase = .43
    old = sphere_tmatrix(2.45+.012j, 4.8, epsilon=11.68, delta_epsilon=.1068,
                         radius=1/3, phase=phase, n_harmonics=2)
    material = SphereMaterial({0: 11.68, 1: .0534*np.exp(-1j*phase), -1: .0534*np.exp(1j*phase)})
    new = material.tmatrix(2.45+.012j, 4.8, radius=1/3, n_harmonics=2)
    for parity in [0, 1]:
        assert_allclose(new.blocks[parity], old.blocks[parity], atol=0, rtol=0)


@pytest.mark.parametrize("depth", [.5, .9])
def test_density_modulated_sphere_phase_covariance_and_negative_frequency_reality(depth):
    material = _garg(depth)
    omega, drive, phase = .437+.013j, .1, .63
    baseline = material.tmatrix(omega, drive, radius=4.27, n_harmonics=3)
    shifted = material.phase_shift(phase).tmatrix(omega, drive, radius=4.27, n_harmonics=3)
    reflected = material.tmatrix(-omega.conjugate(), drive, radius=4.27, n_harmonics=3)
    gauge = np.exp(-1j*baseline.harmonics*phase)
    for parity in [0, 1]:
        assert_allclose(shifted.blocks[parity], gauge[:,None]*baseline.blocks[parity]/gauge[None,:], rtol=5e-10, atol=2e-11)
        assert_allclose(reflected.blocks[parity], baseline.blocks[parity][::-1,::-1].conj(), rtol=5e-10, atol=2e-11)
        assert baseline.diagnostics["boundary"][parity]["relative_residual"] < 1e-12


def test_bulk_output_denominator_matches_primitive_lorentz_time_integration():
    # Use the paper material values, a declared drive, and zero initial P/J.
    # The independent time ODE does not call any constitutive-matrix helper.
    wp2, wr, gamma, drive, carrier, depth = 11., 1., .125, .1, .437, .9
    phase = .43
    period = 2*np.pi/drive
    times = np.linspace(8*period, 10*period, 4001)
    def rhs(t, x):
        forcing = wp2*(1+depth*np.cos(drive*t+phase))*np.exp(-1j*carrier*t)
        return np.array([x[1], forcing-wr**2*x[0]-gamma*x[1]], complex)
    solved = solve_ivp(rhs, (0., times[-1]), np.zeros(2, complex), t_eval=times,
                       method="DOP853", rtol=3e-11, atol=1e-11, max_step=.5)
    assert solved.success
    material = _garg(depth).phase_shift(phase)
    permittivity = material.permittivity(carrier, drive, 2)
    measured = np.array([simpson(solved.y[0]*np.exp(1j*(carrier+n*drive)*times), x=times)
                         /(times[-1]-times[0]) for n in range(-2, 3)])
    predicted = permittivity[:, 2]-np.eye(5)[:, 2]
    assert_allclose(measured, predicted, rtol=2e-8, atol=2e-8)
    # Explicitly distinguish output-kernel modulation, which is a different law.
    wrong_output_kernel = .5*depth*wp2*np.exp(-1j*phase)/(wr**2-carrier**2-1j*gamma*carrier)
    assert abs(measured[3]-wrong_output_kernel) > .1


def test_multi_lorentz_and_epsilon_infinity_harmonics_are_additive_and_analytic():
    p1 = LorentzSpherePole(.8, 1.3, .12, {0:1., 1:.1j, -1:-.1j})
    p2 = LorentzSpherePole(.3, 0., .07, {0:.7})  # Drude contribution, nonzero frequencies.
    epsinf = {0:2.1, 2:.02, -2:.02}
    full = SphereMaterial(epsinf, (p1, p2))
    z, drive, nh = .67+.09j, .2, 2
    expected = (SphereMaterial(epsinf, (p1,)).permittivity(z, drive, nh)
                + SphereMaterial(epsinf, (p2,)).permittivity(z, drive, nh)
                - SphereMaterial(epsinf).permittivity(z, drive, nh))
    assert_allclose(full.permittivity(z, drive, nh), expected, atol=1e-14)
    radius, count = .012, 128
    angle = 2*np.pi*np.arange(count)/count
    contour = np.array([full.permittivity(z+radius*np.exp(1j*t), drive, nh) for t in angle])
    cauchy = np.mean(contour*np.exp(-1j*angle)[:,None,None], axis=0)/radius
    assert_allclose(full.permittivity_derivative(z, drive, nh), cauchy, rtol=2e-12, atol=2e-12)


def test_one_pole_convenience_function_and_canonical_amplitudes():
    result = lorentz_sphere_tmatrix(.437, .1, radius=4.27, plasma_frequency=np.sqrt(11),
        resonance_frequency=1, damping=.125, modulation=.5, phase=.2, n_harmonics=3)
    reference = _garg(.5).phase_shift(.2).tmatrix(.437, .1, radius=4.27, n_harmonics=3)
    for parity in [0, 1]:
        assert_allclose(result.blocks[parity], reference.blocks[parity], atol=0)
    assert "Electric regular/outgoing" in result.diagnostics["amplitude_basis"]


def test_invalid_material_pole_zero_wave_and_complex_instantaneous_material_rejected():
    with pytest.raises(ValueError, match="real time"):
        SphereMaterial({0:1+.2j})
    with pytest.raises(ValueError, match="positivity"):
        _garg(1.1)
    with pytest.raises(ValueError, match="material pole"):
        SphereMaterial(1, (LorentzSpherePole(1, 1, 0),)).permittivity(1, .1, 0)
    with pytest.raises(ValueError, match="zero frequency"):
        _garg(.5).tmatrix(.2, .1, radius=4.27, n_harmonics=2)
    with pytest.raises(ValueError, match="nonnegative"):
        LorentzSpherePole(1, 1, -.01)
