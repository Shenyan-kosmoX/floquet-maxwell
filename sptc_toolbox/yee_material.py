"""Consistent material perturbations of a sampled native Yee constitutive map.

The sparse matrix K maps native displacement degrees of freedom to electric
degrees of freedom.  It includes the solver's cross-component stencil; a
coincident 3 by 3 tensor is insufficient at a smoothed material interface.
This adapter does not solve the spatial Maxwell problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, splu


def sparse_operator_id(matrix) -> str:
    """Stable identity for a canonical sparse operator, independent of storage."""
    a = csc_matrix(matrix, dtype=np.complex128)
    a.sum_duplicates(); a.sort_indices(); a.eliminate_zeros()
    h = hashlib.sha256()
    for value in (np.asarray(a.shape, dtype=np.int64), a.indptr.astype(np.int64),
                  a.indices.astype(np.int64), a.data):
        h.update(value.tobytes())
    return h.hexdigest()


@dataclass
class NativeYeeSymmetryChart:
    """Explicit native-owner restriction and ghost reconstruction.

    ``owners`` selects independent measured rows. ``lift`` reconstructs the
    full exported chart including ghost rows in one declared symmetry sector.
    This is a coordinate transformation, not symmetrization of a response.
    The caller must independently validate the ownership map against fields.
    """
    owners: object
    lift: object
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.owners=np.asarray(self.owners,dtype=int)
        given=csc_matrix(self.lift,dtype=complex)
        if not np.all(np.isfinite(given.data)) or np.max(np.abs(given.data.imag),initial=0)>1e-13:
            raise ValueError("This Gamma-point ghost chart requires finite real reconstruction coefficients")
        self.lift=csc_matrix(given.real,dtype=float)
        if self.owners.ndim!=1 or self.lift.shape[1]!=len(self.owners):
            raise ValueError("Owner indices and reconstruction size disagree")
        if len(np.unique(self.owners))!=len(self.owners) or np.any(self.owners<0) or np.any(self.owners>=self.lift.shape[0]):
            raise ValueError("Owner indices must be distinct valid measured degrees of freedom")
        from scipy.sparse import eye
        residual=self.lift[self.owners]-eye(len(self.owners),format='csc')
        if np.max(np.abs(residual.data),initial=0)>1e-13:
            raise ValueError("Restriction followed by lift must be the owned-coordinate identity")

    def restrict(self, values):
        return np.asarray(values)[self.owners]

    def reconstruct(self, values):
        return self.lift@values

    def restrict_operator(self, full_row_operator):
        """Map native owner inputs to native owner output update rows."""
        return csc_matrix(full_row_operator)[self.owners]@self.lift

    def reconstruction_relative_residual(self, full_values):
        values=np.asarray(full_values)
        return float(np.linalg.norm(values-self.reconstruct(self.restrict(values)))/max(np.linalg.norm(values),1e-300))

    def induced_metric(self, full_weights):
        """Pull back the declared measured-field metric: Wowned=F† Wfull F."""
        from scipy.sparse import diags
        weights=np.asarray(full_weights,dtype=float)
        if weights.shape!=(self.lift.shape[0],) or np.any(weights<0) or not np.all(np.isfinite(weights)):
            raise ValueError("Full-chart weights must be finite nonnegative and match its size")
        return self.lift.T@diags(weights)@self.lift

    def lift_material_derivative(self, material):
        if material.shape!=(len(self.owners),len(self.owners)):
            raise ValueError("Material derivative must use native owned coordinates")
        def action(field):
            return self.reconstruct(material.apply_epsilon_derivative(self.restrict(field)))
        return LinearOperator((self.lift.shape[0],)*2,dtype=float,matvec=action,matmat=action)


@dataclass
class YeeMaterialDerivative:
    """Bulk-parameter derivative of a *discrete* nondispersive material map.

    ``inverse_epsilon`` is K0 and ``inverse_epsilon_derivative`` is dK/dlambda,
    measured on the identical native grid.  Epsilon' = -K0^-1 K' K0^-1.
    Dense inverses are never formed.  Real coefficients are required here;
    absorption, dispersion and complex material parameters need separate laws.
    """
    inverse_epsilon: object
    inverse_epsilon_derivative: object
    material_operator_id: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.inverse_epsilon = csc_matrix(self.inverse_epsilon, dtype=complex)
        self.inverse_epsilon_derivative = csc_matrix(self.inverse_epsilon_derivative, dtype=complex)
        a, b = self.inverse_epsilon, self.inverse_epsilon_derivative
        if a.shape[0] != a.shape[1] or b.shape != a.shape:
            raise ValueError("K and its derivative must be square matrices of the same size")
        if any(not np.all(np.isfinite(m.data)) for m in (a, b)):
            raise ValueError("Material matrices must contain finite coefficients")
        if any(np.max(np.abs(m.data.imag), initial=0) > 1e-13 for m in (a, b)):
            raise ValueError("This nondispersive material derivative requires real matrices")
        measured_id = sparse_operator_id(a)
        if self.material_operator_id and self.material_operator_id != measured_id:
            raise ValueError("Material operator ID does not match the supplied native K")
        self.material_operator_id = measured_id
        self._factor = splu(a)

    @property
    def shape(self):
        return self.inverse_epsilon.shape

    def apply_epsilon(self, electric):
        """Map E to D by solving the actual sparse native constitutive map."""
        return self._factor.solve(np.asarray(electric, dtype=complex))

    def polarization_from_displacement(self, displacement):
        """Return epsilon' E = -K0^-1 K' D for native displacement samples."""
        return -self._factor.solve(self.inverse_epsilon_derivative @ np.asarray(displacement, dtype=complex))

    def apply_epsilon_derivative(self, electric):
        return self.polarization_from_displacement(self.apply_epsilon(electric))

    def as_linear_operator(self):
        """Real constitutive derivative action, without forming dense epsilon'."""
        return LinearOperator(self.shape, dtype=float,
            matvec=self.apply_epsilon_derivative, matmat=self.apply_epsilon_derivative)

    def exact_contrast_action(self, changed_inverse_epsilon):
        """Return epsilon(lambda)-epsilon(0) for an independently sampled Klambda."""
        other = csc_matrix(changed_inverse_epsilon, dtype=complex)
        if other.shape != self.shape:
            raise ValueError("Changed material must use the same native degrees of freedom")
        if not np.all(np.isfinite(other.data)) or np.max(np.abs(other.data.imag), initial=0)>1e-13:
            raise ValueError("Changed material must contain finite real nondispersive coefficients")
        solve_other = splu(other).solve
        def apply(electric):
            electric = np.asarray(electric, dtype=complex)
            return solve_other(electric)-self.apply_epsilon(electric)
        return LinearOperator(self.shape, dtype=float, matvec=apply, matmat=apply)
