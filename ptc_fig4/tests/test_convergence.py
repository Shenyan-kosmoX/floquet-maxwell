import numpy as np
import pytest
from ptc_fig4.materials import SinusoidalDielectric
from ptc_fig4.floquet_maxwell import solve_floquet, TruncationError
from ptc_fig4.references import monodromy_dielectric, quasifrequencies_from_monodromy, matched_spectral_error


def test_harmonic_truncation_against_independent_reference():
    model, k = SinusoidalDielectric(4,.35,True,.4),1.17
    reference = quasifrequencies_from_monodromy(monodromy_dielectric(model,k,rtol=4e-13,atol=4e-15))
    errors = []
    for n in [2,3,5,7]:
        s = solve_floquet(model,k,n)
        idx = s.physical_modes(edge_tolerance=.2,replica_tolerance=1e-5)
        errors.append(matched_spectral_error(s.eigenvalues[idx],reference)[0])
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 2e-11


def test_sampling_convergence_separate_from_harmonic_convergence():
    model = SinusoidalDielectric(4,.6,True,.4)
    spectra = [solve_floquet(model,1.3,7,n_samples=n) for n in [64,128,256]]
    a = spectra[0].eigenvalues[spectra[0].physical_modes()]
    for s in spectra[1:]:
        assert matched_spectral_error(a,s.eigenvalues[s.physical_modes()])[0] < 1e-12


def test_unresolved_truncation_fails_explicitly():
    s = solve_floquet(SinusoidalDielectric(4,.5),1.2,1)
    with pytest.raises(TruncationError):
        s.physical_modes(edge_tolerance=1e-12)
