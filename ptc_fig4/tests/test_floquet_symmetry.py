import numpy as np
import pytest
from ptc_fig4.materials import SinusoidalDielectric, LorentzDielectric
from ptc_fig4.floquet_maxwell import solve_floquet, fourier_blocks
from ptc_fig4.references import matched_spectral_error
from ptc_fig4.mode_tracking import track_modes


@pytest.mark.parametrize("model", [SinusoidalDielectric(4,.2,True,.91),
                                  LorentzDielectric(modulation=.1,phase=.91)])
def test_positive_negative_frequency_pairing(model):
    s = solve_floquet(model,1.07,8)
    w = s.eigenvalues[s.physical_modes()]
    assert matched_spectral_error(w,-w.conj())[0] < 1e-11


def test_fourier_sign_against_exact_phase_sensitive_coefficients():
    model, k = SinusoidalDielectric(4,.31,True,.713),1.17
    blocks = fourier_blocks(model,k,10,512)
    for p in range(-10,11):
        assert abs(blocks[p][1,0]-k*model.inverse_epsilon_coefficient(p)) < 1e-15
        assert abs(blocks[p][0,1]-(k if p==0 else 0)) < 1e-15
    assert abs(blocks[1][1,0].imag) > .01


def test_phase_changes_eigenvectors_but_not_quasifrequencies():
    a = solve_floquet(SinusoidalDielectric(4,.15,True,0),1.08,7)
    b = solve_floquet(SinusoidalDielectric(4,.15,True,.713),1.08,7)
    assert matched_spectral_error(a.eigenvalues[a.physical_modes()],
                                 b.eigenvalues[b.physical_modes()])[0] < 1e-12


def test_biorthogonal_tracking_is_invariant_to_column_permutation_and_phase():
    a = solve_floquet(SinusoidalDielectric(),.87,6)
    b = solve_floquet(SinusoidalDielectric(),.871,6)
    ai, bi = a.physical_modes(), b.physical_modes()[::-1]
    baseline = track_modes(a,ai,b,bi)
    b.right[:,bi] *= np.exp(1j*np.array([.2,1.3]))
    b.left[:,bi] *= np.exp(1j*np.array([.2,1.3]))
    again = track_modes(a,ai,b,bi)
    assert np.array_equal(baseline.permutation,again.permutation)
    assert np.allclose(baseline.overlaps,again.overlaps)
    assert np.min(again.overlaps) > .99
    assert not again.ambiguous
