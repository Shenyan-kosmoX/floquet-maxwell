"""Exact provenance of a native spatial modal coordinate system."""
import hashlib
import json
from numbers import Integral
import numpy as np


def validate_mode_selection(gme, kind, modes):
    frequencies=np.asarray(gme.freqs)
    if frequencies.ndim!=2:
        raise ValueError('Native frequencies must be a k-by-mode array')
    if isinstance(kind,(bool,np.bool_)) or not isinstance(kind,Integral) or not 0<=kind<len(frequencies):
        raise ValueError('kind must be a valid integer Bloch-point index')
    modes=list(modes)
    if not modes or any(isinstance(m,(bool,np.bool_)) or not isinstance(m,Integral)
                        or not 0<=m<frequencies.shape[1] for m in modes):
        raise ValueError('Mode indices must be valid integers, without truncation')
    if len(set(modes))!=len(modes):
        raise ValueError('Duplicate mode indices are not a modal basis')
    return np.asarray(modes,dtype=int)


def native_basis_id(gme, kind, modes):
    """Hash the ordered eigenvectors INCLUDING phase, spatial basis and material.

    It is deliberately not gauge invariant: field and radiation couplings must
    be transformed together, so independently rephased data must not be joined.
    """
    modes=validate_mode_selection(gme,kind,modes)
    digest=hashlib.sha256()
    arrays=[modes,np.asarray(gme.freqs)[kind,modes],np.asarray(gme.kpoints)[:,kind],
            np.asarray(gme.gvec),np.asarray(gme.eigvecs[kind])[:,modes],np.asarray(gme.eps_array)]
    for array in arrays:
        canonical=np.ascontiguousarray(array,dtype='<c16')
        digest.update(str(canonical.shape).encode());digest.update(canonical.tobytes())
    for name in ('gmode_include','ind_modes'):
        if hasattr(gme,name):
            values=getattr(gme,name)[kind]
            for value in values:
                canonical=np.ascontiguousarray(value,dtype='<i8')
                digest.update(str(canonical.shape).encode());digest.update(canonical.tobytes())
    def shape_parameters(shape):
        # Public pretty repr rounds dimensions and can hide small geometry edits.
        values={'type':type(shape).__name__}
        if type(shape).__name__=='Circle' and all(hasattr(shape,key) for key in ('eps','x_cent','y_cent','r')):
            # Preserve existing Circle fingerprints for historical baselines.
            for key in ('eps','x_cent','y_cent','r'):
                value=np.asarray(getattr(shape,key))
                values[key]=[float(value.real),float(value.imag)]
        else:
            from .reference_runner import _state
            values['exact_native_state']=_state(shape,'native_modal_shape')
        return values
    geometry=dict(layers=[dict(d=float(layer.d),background=float(layer.eps_b),
                               shapes=[shape_parameters(shape) for shape in layer.shapes]) for layer in gme.phc.layers],
                  claddings=[float(clad.eps_avg) for clad in gme.phc.claddings],
                  area=float(gme.phc.lattice.ec_area),
                  a1=np.asarray(gme.phc.lattice.a1).tolist(),a2=np.asarray(gme.phc.lattice.a2).tolist())
    digest.update(json.dumps(geometry,sort_keys=True).encode())
    return digest.hexdigest()
