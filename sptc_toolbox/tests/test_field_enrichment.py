"""Physical source-space enrichment, including reuse of actual Green responses."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sptc_toolbox.projected_green import enrich_field_basis, weighted_svd_field_basis
from sptc_toolbox.uniform_green import UniformSlabGreen


META = dict(source='independent physical static fields', coordinates='Cartesian E',
            field_units='c=mu=1', length_units='a')


def make_basis(fields, weights, name='old'):
    return weighted_svd_field_basis(fields, weights, grid_id='shared-grid',
        basis_id=name, geometry_id='same-slab', metadata=META)


def test_enrichment_preserves_existing_sources_and_complex_static_field_span():
    rng = np.random.default_rng(301)
    fields = rng.normal(size=(14, 3, 3))+1j*rng.normal(size=(14, 3, 3))
    weights = np.linspace(.01, .3, 14)
    weights[[0, -1]] = 0
    old = make_basis(fields[..., :1], weights)
    new, transform = enrich_field_basis(old, fields, basis_id='expanded', source='static source snapshots')
    assert_array_equal(new.electric[..., :old.rank], old.electric)
    assert_array_equal(transform[:, 0], np.array([1, 0, 0, 0]))
    assert new.rank == 3  # First supplied column already lies in the old span.
    assert new.metadata['retained_rank'] == 3
    assert new.metadata['parent_basis_generation']['retained_rank'] == 1
    assert_allclose(new.project(new.electric), np.eye(3), atol=2e-14)
    assert max(new.coverage_residual(fields)) < 2e-14
    combined = np.concatenate((old.electric, fields), axis=-1)
    reconstructed = np.einsum('sci,ir->scr', combined, transform)
    assert_allclose(reconstructed[weights > 0], new.electric[weights > 0], atol=2e-14)
    assert_array_equal(new.electric[weights == 0], 0)
    assert not transform.flags.writeable


def test_source_transform_reuses_full_internal_and_port_green_responses():
    slab = UniformSlabGreen(3.7, .64, 32)
    w = 2.1
    raw = slab.static(w)
    q = np.zeros((32, 3, 1), complex)
    q[:, 0, 0] = 1
    old = make_basis(q, np.full(32, slab.spacing))
    # Two actual counterpropagating static incident fields, not a dynamic fit.
    candidates = np.zeros((32, 3, 2), complex)
    candidates[:, 0] = raw.incident_fields
    new, transform = enrich_field_basis(old, candidates,
        basis_id='incident-enriched', source='uniform Maxwell static incident fields')
    combined = np.concatenate((old.electric, candidates), axis=-1)[:, 0]
    actual_internal = w*w*raw.volume_green@combined
    actual_outgoing = w*w*raw.port_green@combined
    # Preserving the columns allows both outputs to use the SAME transform.
    assert_allclose(actual_internal@transform, w*w*raw.volume_green@new.electric[:, 0], atol=4e-14)
    assert_allclose(actual_outgoing@transform, w*w*raw.port_green@new.electric[:, 0], atol=4e-14)
    assert_array_equal((actual_internal@transform)[:, 0], actual_internal[:, 0])
    assert max(old.coverage_residual(candidates)) > .1
    assert max(new.coverage_residual(candidates)) < 1e-13


def test_real_odd_background_completes_single_side_source_without_rotating_even_mode():
    z = np.linspace(-1, 1, 25)
    weights = np.full(len(z), 2/24)
    weights[[0, -1]] *= .5
    even = np.zeros((len(z), 3, 1))
    even[:, 0, 0] = 1+z*z
    old = make_basis(even, weights)
    odd = np.zeros((len(z), 3, 2))
    odd[:, 0, 0] = z
    odd[:, 0, 1] = z**3
    new, _ = enrich_field_basis(old, odd, basis_id='both-parities', source='odd static fields')
    single_side = old.electric+odd[..., :1]+.17j*odd[..., 1:]
    assert old.coverage_residual(single_side)[0] > .3
    assert new.coverage_residual(single_side)[0] < 1e-14
    assert_allclose(new.electric[..., 1:].imag, 0, atol=0)
    assert_allclose(new.electric[::-1, :, 1:], -new.electric[:, :, 1:], atol=1e-14)
    assert_array_equal(new.electric[..., :1], old.electric)


def test_duplicate_static_fields_do_not_create_roundoff_source_directions():
    rng = np.random.default_rng(98)
    old = make_basis(rng.normal(size=(15, 3, 3)), np.ones(15)/15)
    duplicate = old.reconstruct(rng.normal(size=(3, 6))+1j*rng.normal(size=(3, 6)))
    new, transform = enrich_field_basis(old, duplicate,
        basis_id='same-span', source='duplicate static snapshots')
    assert new.rank == old.rank
    assert_array_equal(new.electric, old.electric)
    assert_array_equal(transform[:3], np.eye(3))
    assert_array_equal(transform[3:], 0)
    with pytest.raises(ValueError, match='independent residual'):
        enrich_field_basis(old, duplicate, basis_id='invalid', source='duplicates', added_rank=1)
