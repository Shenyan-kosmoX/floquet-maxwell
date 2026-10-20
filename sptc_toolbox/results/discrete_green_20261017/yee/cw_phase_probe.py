"""Independent empty 1D control for Meep solve_cw returned-field phase."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import json
import numpy as np
import meep as mp

mp.verbosity(0)
out = Path(__file__).resolve().parent/'cw_phase_probe'
out.mkdir(exist_ok=True)
f = .388
source_time = mp.ContinuousSource(f, width=10)
sim = mp.Simulation(cell_size=mp.Vector3(z=6), dimensions=1, resolution=40,
    boundary_layers=[mp.PML(1)], force_complex_fields=True,
    sources=[mp.Source(source_time, component=mp.Ex, center=mp.Vector3())])
z = np.array([0., .1, .5, 1.])
def read():
    return np.array([[sim.get_field_point(c, mp.Vector3(z=zz)) for c in (mp.Ex, mp.Hy)] for zz in z])
sim.run(until=200)
before = read(); t_before = sim.meep_time(); dt = sim.fields.dt
success = sim.solve_cw(tol=1e-9, maxiters=3000, L=10)
after = read(); t_after = sim.meep_time()
sim.fields.step()
after_step = read(); t_step = sim.meep_time()
omega = 2*np.pi*f
ratios = after/before
row = dict(frequency=f, dt=dt, time_before=t_before, time_after=t_after, time_after_step=t_step,
    solve_cw_return=success,
    E_after_before_real_imag=np.stack((ratios[:, 0].real, ratios[:, 0].imag), axis=-1).tolist(),
    H_after_before_real_imag=[[float(r.real), float(r.imag)] if abs(b) > 1e-10 else None
        for r, b in zip(ratios[:, 1], before[:, 1])],
    H_ratio_omission='Ratios are omitted where the reference H is a symmetry-node value below 1e-10',
    field_relative_change_before_after=float(np.linalg.norm(after-before)/np.linalg.norm(before)),
    after_one_step_demodulated_relative_change=float(np.linalg.norm(after_step*np.exp(1j*omega*dt)-after)/np.linalg.norm(after)),
    expected_phase_two_steps=[float(np.cos(-2*omega*dt)), float(np.sin(-2*omega*dt))],
    interpretation='Diagnostic compares a settled empty-domain FDTD response to solve_cw at the same saved time; no fitted phase is applied')
sim.reset_meep()
constant = mp.CustomSource(src_func=lambda t: 1, is_integrated=False, center_frequency=f)
sim = mp.Simulation(cell_size=mp.Vector3(z=6), dimensions=1, resolution=40,
    boundary_layers=[mp.PML(1)], force_complex_fields=True,
    sources=[mp.Source(constant, component=mp.Ex, center=mp.Vector3())])
sim.init_sim(); sim.fields.step(); first_step = read()
constant_success = sim.solve_cw(tol=1e-9, maxiters=3000, L=10)
constant_raw = read()
# Derive from solve_cw's operator (U-zI) and its final source-free update Ux.
# E_ret=z E_x-E_first, H_ret=z H_x.  Frozen midpoint current 1 corresponds
# to harmonic current exp(+i*x); source normalization and the H half-step
# therefore yield E_ph=e^(i*x)(E_ret+E_first), H_ph=H_ret.
x = omega*dt/2
constant_phasor = constant_raw.copy()
constant_phasor[:, 0] = np.exp(1j*x)*(constant_raw[:, 0]+first_step[:, 0])
source_factor = source_time.swigobj.current(t_before-.5*dt, dt)*np.exp(1j*omega*(t_before-.5*dt))
time_phasor = before*np.exp(1j*omega*t_before)/source_factor
time_phasor[:, 1] *= np.exp(-1j*x)
row['constant_current_cw'] = dict(success=constant_success,
    corrected_relative_difference=float(np.linalg.norm(constant_phasor-time_phasor)/np.linalg.norm(time_phasor)),
    corrected_E_relative_difference=float(np.linalg.norm(constant_phasor[:, 0]-time_phasor[:, 0])/np.linalg.norm(time_phasor[:, 0])),
    corrected_H_relative_difference=float(np.linalg.norm(constant_phasor[:, 1]-time_phasor[:, 1])/np.linalg.norm(time_phasor[:, 1])),
    normalization='E_ph=exp(i omega dt/2)*(E_ret+E_first_step); H_ph=H_ret for frozen nonintegrated electric current; independently derived, no fit')
np.savez_compressed(out/'fields.npz', z=z, before=before, after=after, after_step=after_step,
    first_step=first_step, constant_raw=constant_raw, constant_phasor=constant_phasor,
    time_phasor=time_phasor, omega=omega, dt=dt)
(out/'summary.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
print(json.dumps(row, indent=2))
