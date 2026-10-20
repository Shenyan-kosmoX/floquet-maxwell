"""Targeted native Meep frequency-domain solve using audited frozen-current convention."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import argparse
import importlib.util
import json
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import meep as mp

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('native_green', HERE.parents[2]/'scripts/meep_yee_green_response.py')
native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
parser = argparse.ArgumentParser()
parser.add_argument('--basis', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--frequency', type=float, default=.388)
parser.add_argument('--template', type=int, default=1)
parser.add_argument('--tolerance', type=float, default=1e-8)
parser.add_argument('--maxiters', type=int, default=3000)
parser.add_argument('--L', type=int, default=10)
args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True); mp.verbosity(0)
basis = np.load(args.basis); axes = tuple(basis[a] for a in ('x', 'y', 'z')); full = basis['Q_full']
f, j, omega, resolution = args.frequency, args.template, 2*np.pi*args.frequency, 24
points, indices, offsets = native.native_points(axes, resolution)
source_time = mp.CustomSource(src_func=lambda t: 1, is_integrated=False, center_frequency=f)
sources = []
for c, component in enumerate((mp.Ex, mp.Ey, mp.Ez)):
    interpolation = RegularGridInterpolator(axes, full[..., c, j], bounds_error=False, fill_value=0)
    def amplitude(r, interp=interpolation):
        return complex(-1j*omega*interp([r.x, r.y, r.z]).item()) if native.mask(r) else 0j
    sources.append(mp.Source(source_time, component=component, center=mp.Vector3(), size=mp.Vector3(1, 1, .42), amp_func=amplitude))
sim = native.fixture(sources, resolution)
current, dt, temporal = native.extract_first_step_source(sim, source_time, points)
first_electric = native.sample_native(sim, points, (mp.Ex, mp.Ey, mp.Ez))
started = time.perf_counter()
success = sim.solve_cw(tol=args.tolerance, maxiters=args.maxiters, L=args.L)
elapsed = time.perf_counter()-started
x = omega*dt/2
raw_electric = native.sample_native(sim, points, (mp.Ex, mp.Ey, mp.Ez))
electric = np.exp(1j*x)*(raw_electric+first_electric)
work = -.5*resolution**-3*np.vdot(electric, current).real
port_fields, fluxes = [], []
for z in (-.7, .7):
    center, size = mp.Vector3(0, 0, z), mp.Vector3(1, 1, 0)
    weight = sim.get_array_metadata(center=center, size=size)[3]
    fields = [sim.get_array(center=center, size=size, component=c, cmplx=True)*phase
        for c, phase in zip((mp.Ex, mp.Ey, mp.Hx, mp.Hy), (np.exp(1j*x), np.exp(1j*x), 1, 1))]
    ex, ey, hx, hy = fields
    fluxes.append(.5*np.sum((ex.conj()*hy-ey.conj()*hx).real*weight))
    port_fields.append(np.array([np.sum(a*weight)/weight.sum() for a in fields]))
low, high = port_fields
output = np.array([(low[0]-low[3])/2, (high[0]+high[3])/2,
    (low[1]+low[2])/2, (high[1]-high[2])/2])/np.sqrt(2)
incoming = np.array([(low[0]+low[3])/2, (high[0]-high[3])/2,
    (low[1]-low[2])/2, (high[1]+high[2])/2])/np.sqrt(2)
net_flux = fluxes[1]-fluxes[0]
output_power = float(np.vdot(output, output).real)
reference_lifts = np.load(HERE/'f0388/source_lift.npz')['Q_yee']
if f != .388:
    raise ValueError('This targeted control currently requires the audited .388 source package')
lift_error = float(np.linalg.norm(current/(-1j*omega)-reference_lifts[..., j])/np.linalg.norm(reference_lifts[..., j]))
R = resolution**-3*np.einsum('im,i->m', reference_lifts.conj().reshape(-1, 2), electric.reshape(-1))
np.savez_compressed(args.output/'response.npz', electric_yee=electric, raw_electric=raw_electric,
    first_electric=first_electric, current=current, output=output, incoming=incoming,
    R_column=R, work=work, net_flux=net_flux, port_fields=np.stack(port_fields), omega=omega, dt=dt)
row = dict(status='MEEP_NATIVE_CW_CONSTANT_CURRENT_CONTROL', completed=True, success=bool(success),
    frequency=f, template=j, tolerance=args.tolerance, maxiters=args.maxiters, L=args.L,
    solver_seconds=elapsed, initial_current_real_imag=[temporal.real, temporal.imag],
    native_source_lift_difference_from_CW_first_step=lift_error,
    native_work=float(work), net_port_flux=float(net_flux), outgoing_power=output_power,
    incoming_power=float(np.vdot(incoming, incoming).real),
    flux_work_relative_residual=float((net_flux-work)/max(abs(net_flux), abs(work))),
    output_work_relative_residual=float((output_power-work)/max(output_power, abs(work))),
    convention='Frozen nonintegrated current: E_ph=exp(i omega dt/2)*(E_solve_cw+E_first_step); H_ph=H_solve_cw; independently audited in 1D',
    scope='Single frequency Q1 diagnostic, same original masked current lift and subpixel material; no new material-derivative claim')
(args.output/'summary.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
print(json.dumps(row, indent=2), flush=True)
