"""Native Yee source/work audit for the patterned static Green fixture.

The source lift is measured from the FIRST Maxwell update, before fields can
propagate: H=0, D_old=0, hence D_new=-dt*J.  This avoids get_source's
last-src_vol overwrite at overlapping periodic source-volume boundaries.
This is a source-sampling diagnostic, not a material-modulation derivative.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import meep as mp


def mask(r):
    return (abs(r.z) < .21 and (r.x+.22)**2+r.y**2 >= .12**2
        and ((r.x-.22)/.12)**2+(r.y/.09)**2 >= 1)


def fixture(sources, resolution):
    return mp.Simulation(cell_size=mp.Vector3(1, 1, 4.8), resolution=resolution,
        k_point=mp.Vector3(), boundary_layers=[mp.PML(.65, direction=mp.Z)],
        geometry=[mp.Block(size=mp.Vector3(mp.inf, mp.inf, .42), material=mp.Medium(epsilon=11.5)),
            mp.Cylinder(radius=.12, height=.42, center=mp.Vector3(-.22, 0), material=mp.air),
            mp.Ellipsoid(size=mp.Vector3(.24, .18, mp.inf), center=mp.Vector3(.22, 0), material=mp.air)],
        sources=sources, force_complex_fields=True, eps_averaging=True)


def native_points(axes, resolution):
    # Independently established by constant/linear source-coordinate probes;
    # monitor.cpp:get_field(component,vec) interpolates on the component grid,
    # and is exact at these Yee sites.  Do not use decoded coordinates at a
    # periodic edge: get_source overwrites distinct source-volume contributions.
    offsets = np.array([[0, -.5, -.5], [-.5, 0, -.5], [-.5, -.5, 0]])/resolution
    point_sets, index_sets = [], []
    for offset in offsets:
        indices, native_axes = [], []
        for d, (axis, delta) in enumerate(zip(axes, offset)):
            native_axis = axis+delta
            keep = np.ones(len(axis), bool) if d == 2 else ((native_axis >= -.5-1e-10)&(native_axis < .5-1e-10))
            indices.append(np.flatnonzero(keep)); native_axes.append(native_axis[keep])
        points = np.stack(np.meshgrid(*native_axes, indexing='ij'), axis=-1)
        point_sets.append(points); index_sets.append(indices)
    return np.stack(point_sets, axis=-2), index_sets, offsets


def sample_native(sim, points, components):
    shape = points.shape[:-2]
    return np.stack([np.asarray([sim.get_field_point(c, mp.Vector3(*r))
        for r in points[..., i, :].reshape(-1, 3)]).reshape(shape)
        for i, c in enumerate(components)], axis=-1)


def source_slice_native(sim, volume, index_sets):
    return np.stack([sim.get_source(component=c, vol=volume)[np.ix_(*indices)]
        for c, indices in zip((mp.Ex, mp.Ey, mp.Ez), index_sets)], axis=-1)


def extract_first_step_source(sim, source_time, points):
    sim.init_sim()
    dt = sim.fields.dt
    if sim.meep_time() != 0:
        raise ValueError('Source lift extraction requires the initial zero field')
    temporal = source_time.swigobj.current(.5*dt, dt)
    if abs(temporal) < 1e-10:
        raise ValueError('Initial source current is too small for a source-lift probe')
    sim.fields.step()
    displacement = sample_native(sim, points, (mp.Dx, mp.Dy, mp.Dz))
    return -displacement/(dt*temporal), dt, temporal


def ports(sim, f):
    """Return physical phasors with integer-time E and half-time H demodulated.

    Raw H at an integer E time t belongs to t-dt/2.  This avoids the cos(omega
    dt/2) amplitude attenuation in synchronize_magnetic_fields().
    """
    phase_e = np.exp(2j*np.pi*f*sim.meep_time())
    phase_h = np.exp(2j*np.pi*f*(sim.meep_time()-.5*sim.fields.dt))
    rows = []
    for z in (-.7, .7):
        center, size = mp.Vector3(0, 0, z), mp.Vector3(1, 1, 0)
        weight = sim.get_array_metadata(center=center, size=size)[3]
        fields = [sim.get_array(center=center, size=size, component=c, cmplx=True)*phase
            for c, phase in zip((mp.Ex, mp.Ey, mp.Hx, mp.Hy), (phase_e, phase_e, phase_h, phase_h))]
        ex, ey, hx, hy = fields
        mean = np.array([np.sum(a*weight)/weight.sum() for a in fields])
        spatial_flux = .5*np.sum((ex.conj()*hy-ey.conj()*hx).real*weight)
        rows.append((mean, spatial_flux))
    low, high = rows[0][0], rows[1][0]
    output = np.array([(low[0]-low[3])/2, (high[0]+high[3])/2,
        (low[1]+low[2])/2, (high[1]-high[2])/2])/np.sqrt(2)
    incoming = np.array([(low[0]+low[3])/2, (high[0]-high[3])/2,
        (low[1]-low[2])/2, (high[1]+high[2])/2])/np.sqrt(2)
    return output, incoming, rows[1][1]-rows[0][1], np.stack([r[0] for r in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frequency', type=float, default=.388)
    parser.add_argument('--templates', type=int, nargs='+', default=[1, 0])
    parser.add_argument('--checkpoints', type=float, nargs='+', default=[1200, 1800, 2400])
    parser.add_argument('--resolution', type=int, default=24)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True); mp.verbosity(0)
    if args.checkpoints != sorted(set(args.checkpoints)):
        parser.error('Checkpoints must be unique and increasing')
    basis = np.load(args.basis); axes = tuple(basis[a] for a in ('x', 'y', 'z'))
    full = basis['Q_full']; omega = 2*np.pi*args.frequency; dv = args.resolution**-3
    volume = mp.Volume(center=mp.Vector3(), size=mp.Vector3(1, 1, .5))
    points, index_sets, offsets = native_points(axes, args.resolution)
    basis_hash = hashlib.sha256(args.basis.read_bytes()).hexdigest()
    # Uniform constant current is independent of the optical response and
    # tests that all periodic source-volume fragments are summed exactly once.
    constant = mp.CustomSource(src_func=lambda t: 1, is_integrated=False)
    sim = fixture([mp.Source(constant, component=c, center=mp.Vector3(), size=mp.Vector3(1, 1, .42))
        for c in (mp.Ex, mp.Ey, mp.Ez)], args.resolution)
    started = time.perf_counter()
    uniform_actual, dt, _ = extract_first_step_source(sim, constant, points)
    uniform_last_slice = source_slice_native(sim, volume, index_sets)
    interior = np.abs(points[..., 2]) < .21-1/args.resolution
    uniform_error = float(np.max(abs(uniform_actual[interior]-1)))
    if uniform_error > 1e-10:
        raise ValueError(f'Native source extraction failed uniform-periodic control: {uniform_error}')
    uniform_row = dict(maximum_interior_error=uniform_error,
        get_source_last_slice_relative_difference=float(np.linalg.norm(uniform_last_slice-uniform_actual)/np.linalg.norm(uniform_actual)),
        extraction_seconds=time.perf_counter()-started,
        formula='D(first step)/(-dt*src.current(dt/2,dt)); initial H=0 and D=0, no PML in sampled support')
    sim.reset_meep()
    lifts, source_slices, extraction_rows = [], [], []
    for j in range(full.shape[-1]):
        src_time = mp.ContinuousSource(args.frequency, width=20)
        sources = []
        for c, component in enumerate((mp.Ex, mp.Ey, mp.Ez)):
            interpolator = RegularGridInterpolator(axes, full[..., c, j], bounds_error=False, fill_value=0)
            def amplitude(r, interp=interpolator):
                return complex(-1j*omega*interp([r.x, r.y, r.z]).item()) if mask(r) else 0j
            sources.append(mp.Source(src_time, component=component, center=mp.Vector3(),
                size=mp.Vector3(1, 1, .42), amp_func=amplitude))
        sim = fixture(sources, args.resolution)
        began = time.perf_counter()
        actual, dt, temporal = extract_first_step_source(sim, src_time, points)
        last_slice = source_slice_native(sim, volume, index_sets)
        lifts.append(actual/(-1j*omega)); source_slices.append(last_slice/(-1j*omega))
        extraction_rows.append(dict(template=j, initial_current_real_imag=[temporal.real, temporal.imag],
            get_source_last_slice_relative_difference=float(np.linalg.norm(last_slice-actual)/np.linalg.norm(actual)),
            extraction_seconds=time.perf_counter()-began))
        sim.reset_meep()
    lifts = np.stack(lifts, axis=-1)
    np.savez_compressed(args.output/'source_lift.npz', Q_yee=lifts, get_source_Q_yee=np.stack(source_slices, axis=-1),
        points=points, unique_volume_weight=dv, component_offsets=offsets,
        uniform_first_step=uniform_actual, uniform_get_source=uniform_last_slice,
        basis_sha256=basis_hash, omega=omega, dt=dt)
    metadata = dict(status='NATIVE_YEE_SOURCE_AND_WORK_DIAGNOSTIC', frequency=args.frequency,
        resolution=args.resolution, dt=dt, omega=omega, omega_discrete=2*np.sin(omega*dt/2)/dt,
        first_step_uniform_control=uniform_row, source_extractions=extraction_rows,
        basis_sha256=basis_hash,
        source_coordinate_method='Component offset independently established with initialized linear source probes; get_field_point at native grid positions',
        source_lift_method='First zero-field Maxwell update recovers sum of all source-volume contributions, unlike get_source slice overwrite',
        projection='Native E paired with measured total current lift using dv once per unique periodic Yee site',
        source_frequency_factor='src.swigobj.current(t,dt)*exp(i omega t), directly evaluated; no response amplitude fit',
        material_derivative_scope='Same old masked interpolated current fixture; not derivative of Meep subpixel epsilon inverse tensor',
        rows=[], completed=False)
    (args.output/'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(dict(source_probe_complete=True, uniform=uniform_row, sources=extraction_rows)), flush=True)
    if args.probe_only:
        metadata['completed'] = True
        (args.output/'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        return
    for j in args.templates:
        src_time = mp.ContinuousSource(args.frequency, width=20)
        sources = []
        for c, component in enumerate((mp.Ex, mp.Ey, mp.Ez)):
            interpolator = RegularGridInterpolator(axes, full[..., c, j], bounds_error=False, fill_value=0)
            def amplitude(r, interp=interpolator):
                return complex(-1j*omega*interp([r.x, r.y, r.z]).item()) if mask(r) else 0j
            sources.append(mp.Source(src_time, component=component, center=mp.Vector3(),
                size=mp.Vector3(1, 1, .42), amp_func=amplitude))
        sim = fixture(sources, args.resolution); sim.init_sim(); previous = None
        started = time.perf_counter()
        for checkpoint in args.checkpoints:
            sim.run(until=checkpoint-sim.meep_time())
            sampling_start = time.perf_counter()
            current_time = sim.meep_time()
            electric = sample_native(sim, points, (mp.Ex, mp.Ey, mp.Ez))*np.exp(1j*omega*current_time)
            output, incoming, net_flux, port_fields = ports(sim, args.frequency)
            # E(t_n) and H(t_n-dt/2) have individually correct phasors.
            # The drive current belongs to midpoint t_n-dt/2, and its forward
            # dipole difference gives the exact factor below relative to it.
            midpoint = current_time-.5*dt
            source_factor = src_time.swigobj.current(midpoint, dt)*np.exp(1j*omega*midpoint)
            current = -1j*omega*lifts[..., j]*source_factor
            work = -.5*dv*np.vdot(electric, current).real
            raw_R = dv*np.einsum('im,i->m', lifts.conj().reshape(-1, lifts.shape[-1]), electric.reshape(-1))
            centered = np.stack([sim.get_array(vol=volume, component=c, cmplx=True)
                for c in (mp.Ex, mp.Ey, mp.Ez)], axis=-1)*np.exp(1j*omega*current_time)
            old_R = dv*np.einsum('im,i->m', basis['Q'].conj().reshape(-1, lifts.shape[-1]), centered[basis['selected']].reshape(-1))
            name = f'q{j}_t{checkpoint:g}.npz'
            np.savez_compressed(args.output/name, electric_yee=electric, electric_full=centered,
                R_column=raw_R, old_center_R_column=old_R, output=output, incoming=incoming,
                port_fields=port_fields, source_factor=source_factor, work=work, net_flux=net_flux,
                omega=omega, template=j, actual_time=current_time, basis_sha256=basis_hash)
            outgoing_power = float(np.vdot(output, output).real)
            row = dict(template=j, time=current_time, data=name,
                native_work=float(work), net_port_flux=float(net_flux), outgoing_power=outgoing_power,
                incoming_power=float(np.vdot(incoming, incoming).real),
                flux_work_relative_residual=float((net_flux-work)/max(abs(work), abs(net_flux), 1e-300)),
                output_work_relative_residual=float((outgoing_power-work)/max(abs(work), outgoing_power, 1e-300)),
                source_factor_real_imag=[source_factor.real, source_factor.imag],
                field_change_from_previous=None if previous is None else float(np.linalg.norm(electric-previous)/np.linalg.norm(electric)),
                native_sampling_seconds=time.perf_counter()-sampling_start,
                elapsed_seconds=time.perf_counter()-started)
            metadata['rows'].append(row)
            (args.output/'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True); previous=electric.copy()
        sim.reset_meep()
    metadata['completed'] = True
    (args.output/'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
