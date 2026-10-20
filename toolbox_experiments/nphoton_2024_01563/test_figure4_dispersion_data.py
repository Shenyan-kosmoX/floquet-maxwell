"""Quantity-specific gates preserve independent measured evidence."""
import numpy as np
from figure4_dispersion_data import align_and_audit,complex_pair_trial


def trial(real,imag,ratio=.01):
    return dict(quasifrequency_over_drive_real=real,quasifrequency_over_drive_imag=imag,
                singular_ratio_3_2=ratio,relative_rank2_residual=1e-5)


def test_unstable_decaying_phase_does_not_hide_other_branch_or_growth_rate():
    rows=[trial([.5+delta,.5],[-.001,.001]) for delta in [-2e-5,2e-5,-1e-5,1e-5]]
    result=align_and_audit(rows,1.)
    assert not result['pair_accepted_for_plot']
    negative,positive=result['branches']
    assert not negative['branch_real_plot_allowed']
    assert negative['branch_imag_plot_allowed']
    assert positive['branch_real_plot_allowed'] and positive['branch_imag_plot_allowed']


def test_real_outside_split_survives_unresolved_tiny_imaginary_drift():
    rows=[trial([.499,.501],[delta,-delta]) for delta in [-2e-7,2e-7,-1e-7,1e-7]]
    result=align_and_audit(rows,1.)
    assert not result['pair_accepted_for_plot']
    assert all(b['branch_real_plot_allowed'] for b in result['branches'])
    assert not any(b['branch_imag_plot_allowed'] for b in result['branches'])


def test_global_rank_failure_rejects_every_component():
    result=align_and_audit([trial([.499,.501],[0.,0.],ratio=.2) for _ in range(4)],1.)
    assert all(not b['branch_real_plot_allowed'] and not b['branch_imag_plot_allowed'] for b in result['branches'])


def test_complex_multiplier_preserves_both_unequal_branches():
    drive=.4;t=np.arange(0,400/drive,.1)
    expected=np.array([.49-.0003j,.505+.0005j])
    h=(np.exp(-2j*np.pi*drive*expected[0]*t)+(.6+.3j)*np.exp(-2j*np.pi*drive*expected[1]*t))[:,None]*np.array([[1.+.2j,.3-.7j]])
    row=complex_pair_trial(t,h,drive,(30,380),32,.25)
    actual=np.array(row['quasifrequency_over_drive_real'])+1j*np.array(row['quasifrequency_over_drive_imag'])
    assert max(min(abs(z-actual)) for z in expected)<1e-7
    assert not np.isclose(sum(actual.imag),0,atol=1e-6)
