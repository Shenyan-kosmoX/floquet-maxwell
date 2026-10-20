"""Initialize the same Meep sources and inspect get_source, without FDTD.

get_source reports source amplitudes on an array with get_array dimensions.
This checks the source-to-observation-grid mapping; it is not an exact native
Yee adjoint/work integral and must not be used to rescale measured responses.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import meep as mp


def mask_array(x, y, z, margin=0.):
    return ((abs(z) < .21-margin)
        & ((x+.22)**2+y*y >= (.12+margin)**2)
        & (((x-.22)/(.12+margin))**2+(y/(.09+margin))**2 >= 1))


def source_coordinate_probes(args, volume, axes):
    """Determine what coordinates source slices actually encode.

    Meep array_slice.cpp puts each source amplitude at the corresponding
    Dielectric storage index, without interpolating that amplitude to the
    centered field grid. Linear source functions expose the staggered offset.
    These are initialized coordinate fixtures, not fits to field responses.
    """
    values = {}
    for name, function in [('one', lambda r: 1.), ('xy', lambda r: r.x+1j*r.y), ('z', lambda r: r.z)]:
        sources = [mp.Source(mp.ContinuousSource(args.frequency), component=c,
            center=mp.Vector3(), size=mp.Vector3(1, 1, .42), amp_func=function)
            for c in (mp.Ex, mp.Ey, mp.Ez)]
        sim = mp.Simulation(cell_size=mp.Vector3(1, 1, 4.8), resolution=args.resolution,
            boundary_layers=[mp.PML(.65, direction=mp.Z)], k_point=mp.Vector3(),
            sources=sources, force_complex_fields=True)
        sim.init_sim()
        values[name] = np.stack([sim.get_source(component=c, vol=volume) for c in (mp.Ex, mp.Ey, mp.Ez)], axis=-1)
        sim.reset_meep()
    coordinates = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    interior = np.all(abs(coordinates) < np.array([.35, .35, .1]), axis=-1)
    weights = values['one'].real
    source_coordinates = np.stack([values['xy'].real, values['xy'].imag, values['z'].real], axis=-1)
    source_coordinates = np.divide(source_coordinates, weights[..., None],
        out=np.zeros_like(source_coordinates), where=weights[..., None] != 0)
    rows, offsets = [], []
    for c in range(3):
        np.testing.assert_allclose(values['one'][..., c][interior], 1., atol=1e-13)
        delta = (source_coordinates[..., c, :]-coordinates)[interior]
        offset = delta.mean(axis=0)
        residual = float(np.max(abs(delta-offset)))
        if residual > 1e-12:
            raise ValueError('Source-array coordinate probe is not a fixed staggered offset')
        offsets.append(offset)
        rows.append(dict(component=('Ex', 'Ey', 'Ez')[c], source_minus_metadata_coordinate=offset.tolist(),
            offset_in_grid_cells=(offset*args.resolution).tolist(), maximum_coordinate_probe_residual=residual))
    return np.asarray(offsets), rows, weights, source_coordinates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frequency', type=float, default=.398)
    parser.add_argument('--resolution', type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mp.verbosity(0)
    basis = np.load(args.basis)
    axes = tuple(basis[a] for a in ('x', 'y', 'z'))
    full = basis['Q_full']
    selected = basis['selected']
    xx, yy, zz = np.meshgrid(*axes, indexing='ij')
    unique = (xx >= -.5)&(xx < .5)&(yy >= -.5)&(yy < .5)
    physical = mask_array(xx, yy, zz)
    interior = mask_array(xx, yy, zz, 1/args.resolution)&unique
    interface = selected&~interior
    omega = 2*np.pi*args.frequency
    geometry = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, .42), material=mp.Medium(epsilon=11.5)),
        mp.Cylinder(radius=.12, height=.42, center=mp.Vector3(-.22, 0), material=mp.air),
        mp.Ellipsoid(size=mp.Vector3(.24, .18, mp.inf), center=mp.Vector3(.22, 0), material=mp.air)]
    volume = mp.Volume(center=mp.Vector3(), size=mp.Vector3(1, 1, .5))
    offsets, coordinate_probes, source_volume_weights, decoded_coordinates = source_coordinate_probes(args, volume, axes)
    rows, saved = [], {}
    for j in range(full.shape[-1]):
        sources = []
        for c, component in enumerate((mp.Ex, mp.Ey, mp.Ez)):
            values = full[..., c, j]
            if np.linalg.norm(values) == 0:
                continue
            interpolation = RegularGridInterpolator(axes, values, bounds_error=False, fill_value=0)
            def amplitude(point, interp=interpolation):
                if not mask_array(point.x, point.y, point.z):
                    return 0j
                return complex(-1j*omega*interp([point.x, point.y, point.z]).item())
            sources.append(mp.Source(mp.ContinuousSource(args.frequency, width=20), component=component,
                center=mp.Vector3(), size=mp.Vector3(1, 1, .42), amp_func=amplitude))
        sim = mp.Simulation(cell_size=mp.Vector3(1, 1, 4.8), resolution=args.resolution,
            k_point=mp.Vector3(), boundary_layers=[mp.PML(.65, direction=mp.Z)],
            geometry=geometry, sources=sources, force_complex_fields=True, eps_averaging=True)
        sim.init_sim()
        measured_axes = sim.get_array_metadata(vol=volume)[:3]
        for a, b in zip(axes, measured_axes):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-13)
        actual = np.stack([sim.get_source(component=c, vol=volume) for c in (mp.Ex, mp.Ey, mp.Ez)], axis=-1)
        expected = -1j*omega*full[..., j]*physical[..., None]
        if actual.shape != expected.shape:
            raise ValueError(f'get_source dimensions {actual.shape} differ from expected {expected.shape}')
        difference = actual-expected
        # Correctly interpret source-slice indices before checking amp_func.
        # A source-volume boundary can carry partial volume weights; separate
        # it from the interior instead of treating a cell-center mismatch as
        # an incorrectly injected current.
        expected_staggered = np.zeros_like(actual)
        expected_with_volume_weights = np.zeros_like(actual)
        staggered_inside = np.zeros_like(actual, dtype=bool)
        coords = np.stack([xx, yy, zz], axis=-1)
        for c in range(3):
            native_coords = coords+offsets[c]
            interp = RegularGridInterpolator(axes, full[..., c, j], bounds_error=False, fill_value=0)
            inside = mask_array(*np.moveaxis(native_coords, -1, 0))
            expected_staggered[..., c] = -1j*omega*interp(native_coords)*inside
            decoded = decoded_coordinates[..., c, :]
            expected_with_volume_weights[..., c] = (-1j*omega*interp(decoded)
                *mask_array(*np.moveaxis(decoded, -1, 0))*source_volume_weights[..., c])
            staggered_inside[..., c] = ((abs(native_coords[..., 2]) < .21-1/args.resolution)
                & (abs(native_coords[..., 0]) < .5-1/args.resolution)
                & (abs(native_coords[..., 1]) < .5-1/args.resolution))
        staggered_difference = actual-expected_staggered
        regions = {}
        for name, region in [('all_unique', unique), ('physical_source', selected), ('interior', interior), ('interface', interface)]:
            den = np.linalg.norm(expected[region])
            regions[name] = dict(samples=int(region.sum()), reference_norm=float(den),
                actual_norm=float(np.linalg.norm(actual[region])),
                relative_difference=float(np.linalg.norm(difference[region])/den) if den else None,
                error_fraction_of_total_source=float(np.linalg.norm(difference[region])/np.linalg.norm(expected[unique])))
        # The interpolated current slice can have values just outside a sharp
        # source boundary even when amp_func injects no current outside it.
        # Such values are not automatically a source leakage implementation bug.
        outside = unique&~physical
        rows.append(dict(template=j, elapsed_simulation_time=float(sim.meep_time()), regions=regions,
            staggered_coordinate_comparison=dict(
                full_array_relative_difference=float(np.linalg.norm(staggered_difference[unique])/np.linalg.norm(expected_staggered[unique])),
                source_volume_interior_relative_difference=float(np.linalg.norm(staggered_difference[staggered_inside])/np.linalg.norm(expected_staggered[staggered_inside])),
                with_coordinate_and_source_volume_weight_probes_relative_difference=float(np.linalg.norm(actual-expected_with_volume_weights)/np.linalg.norm(expected_with_volume_weights)),
                interpretation='Independent constant/linear source fixtures identify staggered coordinates and source-volume weights; no dynamic response fit'),
            center_slice_outside_physical_mask_fraction=float(np.linalg.norm(actual[outside])/np.linalg.norm(actual[unique])),
            center_slice_actual_expected_complex_overlap=[float(np.vdot(expected[unique], actual[unique]).real), float(np.vdot(expected[unique], actual[unique]).imag)],
            per_component_source_region_relative_difference=[float(np.linalg.norm(difference[..., c][selected])/max(np.linalg.norm(expected[..., c][selected]), 1e-300)) for c in range(3)]))
        saved[f'actual_source_q{j}'] = actual
        saved[f'expected_center_source_q{j}'] = expected
        saved[f'expected_staggered_source_q{j}'] = expected_staggered
        saved[f'expected_with_volume_weights_q{j}'] = expected_with_volume_weights
        sim.reset_meep()
    report = dict(status='INITIALIZED_CURRENT_SOURCE_SLICE_AUDIT_NO_FDTD_PROPAGATION',
        frequency_fa_over_c=args.frequency, resolution=args.resolution,
        basis_sha256=hashlib.sha256(args.basis.read_bytes()).hexdigest(), rows=rows,
        coordinate_probes=coordinate_probes,
        formula='expected voxel-center source = -i omega Q_full times physical source mask',
        scope='Actual get_source amplitude arrays versus pointwise center expectation, initialized with identical source construction',
        limitations=['get_source has get_array dimensions but raw source amplitudes are stored at dielectric indices with component staggering; naive center comparison is not injection error',
            'Coordinate probes identify source-array staggering; this does not establish an exact adjoint projection of observed fields',
            'Source-volume edge weights and component staggering can place returned values outside the center-point mask; not by itself physical source leakage',
            'This comparison does not correct R, T or mode amplitudes and is not a dynamic response fit',
            'No time evolution; cannot measure source settling or attribute all optical-theorem residual to sampling'],
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(args.output/'source_slices.npz', **saved, x=axes[0], y=axes[1], z=axes[2], selected=selected, interior=interior,
        source_volume_weights=source_volume_weights, source_coordinates=decoded_coordinates)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
