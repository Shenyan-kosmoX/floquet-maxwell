"""R2a: native Legume radiation overlaps at one common positive real frequency.

The internal spatial fields/eigenvectors remain fixed. Only the outgoing
reference radiation waves and their density of states are evaluated at the
requested frequency. This is NOT a retarded complex-frequency self-energy:
C(omega) C(omega)^dagger is a real-axis spectral Gram, not an analytic function.

The overlap loop is adapted from GuidedModeExp._compute_rad_components in
legume-gme 1.0.3, https://github.com/fancompute/legume/blob/v1.0.3/legume/gme/gme.py.
Changes: a common explicit positive real radiation frequency, multiple fixed
modal rows, threshold rejection, immutable labeled output and provenance.
The installed library is never patched and gme.freqs is never overwritten.

Upstream license (applies to the adapted overlap loop):
MIT License
Copyright (c) 2019 Fan group
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from importlib.metadata import version
import inspect
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np

from .basis import native_basis_id, validate_mode_selection
from .radiation import (ChannelLabel,FrozenRadiationBath,_labels,_positive_real,
                        _readonly_array,_unitary,_permutation)

SUPPORTED_LEGUME_VERSION = "1.0.3"
# SHA256 of the upstream UTF-8 source with normalized LF line endings. These
# identify the exact audited implementation, rather than trusting a version
# string after local monkey patches or changes to the installed source files.
SUPPORTED_SOURCE_SHA256 = {
    "gme": "6429487a560e5213cb90068aa0ceb39ebd66981d96baf3786f645cbd34d132a7",
    "matrix_elements": "b7e208615d94e2db84363400534480ae3c63cdc80fc725da1b854e97bfc06d04",
    "slab_modes": "5cfe2583d2b1eb27055390af611c2df96f715530b9e5b315c9771f88c819a355",
}


class DiffractionThresholdError(ValueError):
    """A requested sample is at or numerically too close to kz=0."""


class FrozenWindowError(ValueError):
    """A heuristic frozen-frequency window failed, not a malformed operator."""


def _source_hash(obj):
    source = Path(inspect.getfile(obj)).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def source_provenance():
    """Fail closed for unaudited source/version; no installed files are edited."""
    import legume
    from legume.gme import matrix_elements, slab_modes
    if version("legume-gme") != SUPPORTED_LEGUME_VERSION:
        raise RuntimeError("Frequency radiation adapter is pinned to legume-gme==1.0.3")
    objects = {"gme": legume.GuidedModeExp, "matrix_elements": matrix_elements, "slab_modes": slab_modes}
    actual = {name: _source_hash(obj) for name, obj in objects.items()}
    if actual != SUPPORTED_SOURCE_SHA256:
        bad = [name for name in actual if actual[name] != SUPPORTED_SOURCE_SHA256[name]]
        raise RuntimeError(f"Native Legume source hash differs from audited 1.0.3: {bad}")
    return {"legume_version": SUPPORTED_LEGUME_VERSION, "source_sha256_lf_utf8": actual,
            "upstream_license": "MIT; Copyright (c) 2019 Fan group; notice retained in this module",
            "source_url": "https://github.com/fancompute/legume/blob/v1.0.3/legume/gme/gme.py",
            "adapted_function": "GuidedModeExp._compute_rad_components",
            "adapter_source_sha256": _source_hash(FrequencyRadiationAdapter)}


def native_state_hash(gme):
    """Fingerprint the whole in-memory native state to detect any mutation.

    This serializes a live trusted project object; it does not load a pickle or
    execute data from an external source. Hashes are local-process provenance,
    not claimed portable across Python versions.
    """
    return hashlib.sha256(pickle.dumps(gme, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


@dataclass(frozen=True)
class CouplingSample:
    """Common-real-frequency radiation data in native angular a/c units.

    C has modal rows and physical channel columns. R=C C^dagger is the native
    squared-frequency correction; gamma_weak=R/(2*omega) is its weak positive-
    frequency damping approximation. Neither includes a reactive frequency
    shift, negative-frequency outgoing continuation, or pump-dependent bath.
    """
    omega_a_over_c: float
    kpoint: np.ndarray
    basis_id: str
    couplings: np.ndarray
    labels: tuple[ChannelLabel, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        w = _positive_real(self.omega_a_over_c, "omega_a_over_c (positive real axis only)")
        if not np.isfinite(w*w):
            raise ValueError("Squared sample frequency overflows; check omega*a/c units")
        k = _readonly_array(self.kpoint, float, "kpoint", 1)
        c = _readonly_array(self.couplings, complex, "couplings", 2)
        if k.shape != (2,) or c.shape[0] < 1:
            raise ValueError("Require one two-component Bloch k and at least one modal row")
        if not isinstance(self.basis_id, str) or not self.basis_id:
            raise ValueError("A nonempty native basis_id is required")
        labels = _labels(self.labels, c.shape[1])
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        meta = deepcopy(dict(self.metadata))
        units = {"couplings_units": "omega_a_over_c", "omega_units": "omega_a_over_c",
                 "kpoint_units": "k_a", "gamma_weak_units": "omega_a_over_c"}
        for key, expected in units.items():
            if key in meta and meta[key] != expected:
                raise ValueError(f"Invalid {key}; expected {expected}")
            meta[key] = expected
        with np.errstate(over="ignore", invalid="ignore"):
            gram = c @ c.conj().T
        if not np.all(np.isfinite(gram)):
            raise ValueError("Radiation Gram overflows")
        meta["status"] = "R2A_REAL_AXIS_COUPLING_NOT_ANALYTIC_SELF_ENERGY"
        meta["basis_id"] = self.basis_id
        for name, value in [("omega_a_over_c", w), ("kpoint", k), ("couplings", c),
                            ("labels", labels), ("metadata", meta)]:
            object.__setattr__(self, name, value)

    @property
    def omega(self):
        return self.omega_a_over_c

    @property
    def k(self):
        return self.kpoint

    @property
    def radiation_matrix(self):
        value = self.couplings @ self.couplings.conj().T
        value = (value + value.conj().T)/2
        value.setflags(write=False)
        return value

    @property
    def R(self):
        return self.radiation_matrix

    @property
    def gamma_weak(self):
        value = self.radiation_matrix/(2*self.omega_a_over_c)
        value.setflags(write=False)
        return value

    @property
    def native_linewidths(self):
        """Native square-root prescription at this evaluation w, in f*a/c.

        It is an on-shell regression quantity only when w equals the associated
        native modal frequency. It is not an off-shell resonance linewidth.
        """
        strength = np.diag(self.radiation_matrix).real
        w2 = self.omega_a_over_c**2
        real_root = np.sqrt(np.hypot(w2, strength)/2+w2/2)
        return strength/(2*real_root)/(2*np.pi)

    def transform_modes(self, unitary):
        u = _unitary(unitary, self.couplings.shape[0])
        changed = hashlib.sha256(self.basis_id.encode()+np.ascontiguousarray(u,dtype="<c16").tobytes()).hexdigest()
        meta = deepcopy(dict(self.metadata))
        meta.update(source_basis_id=self.basis_id, material_basis_transform_required=True,
                    basis_transform=np.stack([u.real,u.imag],axis=-1).tolist())
        if 'native_omega_a_over_c' in meta:
            values=np.asarray(meta['native_omega_a_over_c'],float)
            operator=u.conj().T@np.diag(values)@u
            diagonal=np.allclose(operator,np.diag(np.diag(operator)),rtol=1e-10,atol=1e-12)
            meta['frequency_basis_is_diagonal']=bool(meta.get('frequency_basis_is_diagonal',True) and diagonal)
            if meta['frequency_basis_is_diagonal']:
                meta['native_omega_a_over_c']=np.diag(operator).real.tolist()
            else:meta['native_omega_over_drive_is_spectrum_only']=True
        return type(self)(self.omega_a_over_c,self.kpoint,changed,u.conj().T@self.couplings,self.labels,meta)

    def permute_channels(self, order):
        permutation = _permutation(order, self.couplings.shape[1])
        return type(self)(self.omega_a_over_c,self.kpoint,self.basis_id,self.couplings[:,permutation],
                          tuple(self.labels[i] for i in permutation),self.metadata)

    def to_frozen_bath(self,drive_omega_a_over_c,*,max_relative_spread=.08,max_loss_ratio=.05,
                       window_policy='strict'):
        """Freeze this actual common-frequency sample as a guarded R1 bath.

        Unlike joining native on-shell rows, this permits a modal window that
        spans a channel threshold: all rows have already been evaluated using
        one common, noncritical channel set. It remains a frequency-frozen,
        positive-frequency weak-radiation approximation, not a solved sideband
        self-energy or a justification to cross the adapter's threshold guard.

        window_policy='trial' permits an explicitly unassessed frozen model
        beyond the heuristic modal-frequency window. It does not bypass units,
        native basis, threshold sampling or collective weak-loss constraints.
        Actual occupied-sideband assessment is a separate result, not a pass
        inferred from the mere existence of the finite-model spectrum.
        """
        drive=_positive_real(drive_omega_a_over_c,'drive_omega_a_over_c')
        spread_limit=_positive_real(max_relative_spread,'max_relative_spread')
        loss_limit=_positive_real(max_loss_ratio,'max_loss_ratio')
        if window_policy not in ('strict','trial'):
            raise ValueError("window_policy must be strict or trial")
        if spread_limit>=1 or loss_limit>=1:
            raise ValueError('Spread and weak-loss limits must be smaller than one')
        if 'native_omega_a_over_c' not in self.metadata:
            raise ValueError('Native modal frequencies are required to guard the common-frequency bath')
        w=_readonly_array(self.metadata['native_omega_a_over_c'],float,'native modal frequencies',1)
        if w.shape!=(self.couplings.shape[0],) or np.any(w<=0):
            raise ValueError('Native positive modal frequencies must match all coupling rows')
        spread=float(np.ptp(w)/np.mean(w))
        distance=float(np.max(abs(w-self.omega_a_over_c))/self.omega_a_over_c)
        window_passed=bool(spread<=spread_limit and distance<=spread_limit)
        if not window_passed and window_policy=='strict':
            raise FrozenWindowError('Modal frequency spread or common evaluation frequency exceeds the frozen-bath window')
        c=self.couplings/np.sqrt(2.)/np.sqrt(self.omega_a_over_c)/np.sqrt(drive)
        gamma=c@c.conj().T
        collective=float(max(0.,np.linalg.eigvalsh(gamma).max())/np.min(w/drive))
        if collective>loss_limit:
            raise ValueError('Collective common-frequency radiation is too strong for the weak-radiation bath')
        weak=np.diag(self.gamma_weak).real
        sqrt_rate=2*np.pi*self.native_linewidths
        error=np.divide(weak-sqrt_rate,sqrt_rate,out=np.zeros_like(weak),where=sqrt_rate>0)
        meta=deepcopy(dict(self.metadata))
        meta.update(couplings_units='sqrt(gamma_over_Omega)',damping_matrix_units='gamma_over_Omega',
            native_couplings_units='omega_a_over_c',native_omega_over_drive=(w/drive).tolist(),
            drive_omega_a_over_c=drive,reference_w=self.omega_a_over_c,reference_w_units='omega_a_over_c',
            reference_omega_a_over_c=self.omega_a_over_c,reference_omega_over_drive=self.omega_a_over_c/drive,
            sample_omega_a_over_c=self.omega_a_over_c,source_common_real_frequency=True,
            bath_source='R2a actual common-positive-real-frequency native overlap sample',
            kpoint=self.kpoint.tolist(),kpoint_units='k_a',on_shell_spread=spread,
            reference_relative_distance=distance,max_relative_spread=spread_limit,max_loss_ratio=loss_limit,
            frequency_window_policy=window_policy,frequency_window_passed=window_passed,
            frozen_approximation_assessment='UNASSESSED_TRIAL' if window_policy=='trial' else 'HEURISTIC_WINDOW_ONLY',
            collective_loss_ratio=collective,sample_weak_sqrt_rate_relative_errors=error.tolist(),
            sample_sqrt_linewidths_fa_over_c=self.native_linewidths.tolist(),
            weak_rate_comparison_scope='At the common sample frequency, not the native on-shell resonance linewidths',
            approximation='Freeze a true common-positive-real-frequency sample; no reactive shift or sideband-resolved radiation',
            approximation_scope='Weak collective radiation and narrowband native modal frequencies at fixed k and pump-independent bath')
        meta.setdefault('frequency_basis_is_diagonal',True)
        return FrozenRadiationBath(c,self.labels,meta)


class FrequencyRadiationAdapter:
    """Fixed native fields coupled to radiation evaluated at a common real w.

    Uses audited native 1.0.3 _get_guided/_get_rad and matrix_elements. The
    minimal assembly port changes only the radiation evaluation frequency.
    No gme frequencies, eigenvectors, geometry, or installed code are modified.
    An external mutation of the bound object invalidates this adapter; construct
    a new adapter after an intentional basis change.
    """
    def __init__(self, gme, kind=0, minds=None, *, threshold_relative_tolerance=1e-8):
        import legume
        from legume.backend import backend
        self.provenance = source_provenance()
        if type(gme) is not legume.GuidedModeExp:
            raise TypeError("A native legume.GuidedModeExp 1.0.3 instance is required; subclasses are not audited")
        if backend.__class__.__name__ != "NumpyBackend":
            raise RuntimeError("Frequency radiation is audited only with Legume's NumPy backend")
        for name in ("_get_guided", "_get_rad"):
            method = getattr(gme,name)
            if getattr(method,"__func__",None) is not getattr(legume.GuidedModeExp,name):
                raise RuntimeError(f"Patched native method {name} is unsupported")
        if minds is None:
            minds = list(range(np.asarray(gme.freqs).shape[1]))
        self.minds = validate_mode_selection(gme,kind,minds)
        self.minds.setflags(write=False)
        self.kind = int(kind)
        self.threshold_relative_tolerance = _positive_real(threshold_relative_tolerance,"threshold_relative_tolerance")
        if self.threshold_relative_tolerance >= .1:
            raise ValueError("Threshold tolerance must be smaller than 0.1")
        self._gme = gme
        self.kpoint = _readonly_array(gme.kpoints[:,kind],float,"native kpoint",1)
        self.native_omega_a_over_c = _readonly_array(2*np.pi*gme.freqs[kind,self.minds],float,"native frequencies",1)
        if np.any(self.native_omega_a_over_c<=0):
            raise ValueError("Selected native frequencies must be positive")
        self.basis_id = native_basis_id(gme,kind,self.minds)
        self.source_state_hash = native_state_hash(gme)
        exterior = np.array([clad.eps_avg for clad in gme.phc.claddings])
        if np.iscomplexobj(exterior) or not np.all(np.isfinite(exterior)) or np.any(exterior<=0):
            raise ValueError("This native adapter requires real positive cladding permittivities")
        self.cladding_epsilon = exterior.astype(float)
        self._gkx = np.asarray(gme.gvec[0])+self.kpoint[0]+gme.delta_gx
        self._gky = np.asarray(gme.gvec[1])+self.kpoint[1]
        self._gk = np.hypot(self._gkx,self._gky)
        if np.any(self._gk==0):
            raise ValueError("Native polarization direction is singular; retain the audited delta_gx regulator")
        px,py = self._gkx/self._gk,self._gky/self._gk
        qx,qy = py,-px
        self._pp = np.outer(px,px)+np.outer(py,py)
        self._pq = np.outer(px,qx)+np.outer(py,qy)
        self._qq = np.outer(qx,qx)+np.outer(qy,qy)

    def _unchanged(self):
        if native_state_hash(self._gme) != self.source_state_hash:
            raise RuntimeError("Bound native GME state changed; reconstruct the frequency adapter for the new basis")

    def evaluate(self, omega_a_over_c):
        w = _positive_real(omega_a_over_c,"omega_a_over_c (positive real axis only)")
        if not np.isfinite(w*w):
            raise ValueError("Evaluation frequency overflows; check omega*a/c units")
        self._unchanged()
        try:
            result = self._evaluate_native_loop(w)
        finally:
            self._unchanged()
        return result

    def _evaluate_native_loop(self,w):
        """Adapted native 1.0.3 overlap loop; retain TE/TM phases and DOS."""
        from legume.gme import matrix_elements as elements
        gme,kind = self._gme,self.kind
        # Exactly at a threshold native <= selection has 1/kz. Do not silently
        # sample it; both open and closed near-threshold approaches are rejected.
        margin = np.sqrt(self.cladding_epsilon)[:,None]*w-self._gk[None,:]
        scale = np.maximum(1.,np.maximum(np.sqrt(self.cladding_epsilon)[:,None]*w,self._gk[None,:]))
        near = abs(margin)<=self.threshold_relative_tolerance*scale
        if np.any(near):
            indices = np.argwhere(near)
            raise DiffractionThresholdError(f"Evaluation is at/near exterior kz=0 diffraction threshold: {indices.tolist()}")
        indmoder = [np.flatnonzero(margin[clad]>0) for clad in (0,1)]
        gkr = [self._gk[index] for index in indmoder]
        raw = {pol:[np.zeros((len(self.minds),len(index)),complex) for index in indmoder] for pol in ("te","tm")}
        xs,ys,chis = ({pol:[] for pol in ("te","tm")} for _ in range(3))
        for clad in (0,1):
            for pol in ("te","tm"):
                x,y,chi = gme._get_rad(gkr[clad],w,pol=pol,clad=clad)
                xs[pol].append(x);ys[pol].append(y);chis[pol].append(chi)
        if gme.gradients=="exact":
            eps,thickness = gme.eps_array,gme.d_array
        elif gme.gradients=="approx":
            eps,thickness = gme.eps_array_val,gme.d_array
        else:
            raise ValueError("Unknown native guided-mode gradient/parameter convention")
        count = 0
        for guided_index,mode in enumerate(gme.gmode_include[kind]):
            indices,oms,a,b,chi = gme._get_guided(self._gk,kind,mode)
            number = gme.modes_numg[kind][guided_index]
            # Native uses conjugated eigenvector weights, hence C'=U^dagger C.
            coefficients = np.asarray(gme.eigvecs[kind])[count:count+number,self.minds].conj().T
            for clad in (0,1):
                omr_array = w*np.ones(len(indmoder[clad]))
                args = (eps,thickness,gme.eps_inv_mat,indices,oms,a,b,chi,
                        indmoder[clad],omr_array)
                if mode%2==0:
                    te = elements.mat_te_te(*args,ys['te'][clad],xs['te'][clad],chis['te'][clad],self._qq)
                else:
                    te = elements.mat_tm_te(*args,ys['te'][clad],xs['te'][clad],chis['te'][clad],self._pq)
                raw['te'][clad] += -1j*(coefficients@np.asarray(te))
                if mode%2==0:
                    tm = elements.mat_te_tm(*args,ys['tm'][clad],xs['tm'][clad],chis['tm'][clad],self._pq.T)
                else:
                    tm = elements.mat_tm_tm(eps,thickness,gme.eps_inv_mat,self._gk,indices,oms,a,b,chi,
                        indmoder[clad],omr_array,ys['tm'][clad],xs['tm'][clad],chis['tm'][clad],self._pp)
                raw['tm'][clad] += coefficients@np.asarray(tm)
            count += number
        if count!=len(gme.eigvecs[kind]):
            raise RuntimeError("Native guided basis block lengths do not match eigenvectors")
        labels,blocks,kzs = [],[],[]
        for clad,side in enumerate(('l','u')):
            kz = np.sqrt(self.cladding_epsilon[clad]*w*w-gkr[clad]**2)
            # Native rad_dos=epsilon/(4*pi*kz), C=sqrt(pi*rad_dos)*overlap.
            rad_dos = self.cladding_epsilon[clad]/kz/(4*np.pi)
            for pol in ('te','tm'):
                blocks.append(np.sqrt(np.pi*rad_dos)[None,:]*raw[pol][clad])
                labels.extend((side,pol,float(gme.gvec[0,ig]),float(gme.gvec[1,ig])) for ig in indmoder[clad])
                kzs.extend(kz.tolist())
        c = np.concatenate(blocks,axis=1)
        order = sorted(range(len(labels)),key=lambda i:labels[i])
        labels = tuple(labels[i] for i in order)
        c = c[:,order]
        metadata = {"provenance":deepcopy(self.provenance),"mode_indices":self.minds.tolist(),
            "kind":self.kind,"native_omega_a_over_c":self.native_omega_a_over_c.tolist(),
            "evaluation_frequency_common_to_all_rows":True,"cladding_epsilon":self.cladding_epsilon.tolist(),
            "spatial_reciprocal_vectors_Ga":np.asarray(gme.gvec).tolist(),
            "frequency_basis_is_diagonal":True,
            "channel_kz_a":[kzs[i] for i in order],"threshold_relative_tolerance":self.threshold_relative_tolerance,
            "source_state_hash":self.source_state_hash,"native_object_mutation_check":"unchanged before/after evaluation",
            "native_delta_gx":float(gme.delta_gx),"native_basis_fields_fixed":True,
            "scope":"positive-real-frequency outgoing reference channels, native perturbative overlap/DOS",
            "not_provided":["reactive self-energy","analytic complex-frequency continuation","negative-frequency retarded map",
                            "pump-dependent system-bath coupling","complete radiation continuum beyond native G cutoff"]}
        return CouplingSample(w,self.kpoint,self.basis_id,c,labels,metadata)
