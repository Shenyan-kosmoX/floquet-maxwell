"""Version-pinned native Lorentz sigma(t) adapter; preserves polarization memory.

No installed Meep files are changed. Own C++ code accesses the exact installed
headers/ABI, scales only pre-existing forcing arrays, and retains P/P_prev.
One electric Lorentz pole is required; do not reset or mutate the simulation
while this handle is alive. This is a toolbox extension, not a public Python
Meep feature, and must pass independent homogeneous validation before use.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import subprocess
import sys
import meep as mp

HERE=Path(__file__).resolve().parent


def build():
    if mp.__version__!='1.35.0':raise RuntimeError('Lorentz adapter pinned to Meep 1.35.0')
    prefix=Path(sys.prefix);source=HERE/'native_lorentz_sigma.cpp';out=HERE/'native_build';out.mkdir(exist_ok=True)
    lib=out/'liblorentz_sigma.so';manifest=out/'build_manifest.json'
    inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,prefix/'include/meep.hpp',prefix/'lib/libmeep.so']}
    if lib.exists() and manifest.exists() and json.loads(manifest.read_text()).get('inputs')==inputs:return lib
    command=['g++','-std=c++17','-O2','-fPIC','-shared',str(source),'-I'+str(prefix/'include'),
        '-L'+str(prefix/'lib'),'-Wl,-rpath,'+str(prefix/'lib'),'-lmeep','-o',str(lib)]
    result=subprocess.run(command,text=True,capture_output=True,timeout=60)
    (out/'compiler.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Native Lorentz adapter compilation failed: '+result.stderr)
    manifest.write_text(json.dumps(dict(inputs=inputs,command=command,meep_version=mp.__version__,
        library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest()),indent=2))
    return lib


class LorentzSigmaModulator:
    def __init__(self,sim):
        self.sim=sim;self.api=ctypes.CDLL(str(build()))
        self.api.lorentz_sigma_error.restype=ctypes.c_char_p
        self.api.lorentz_sigma_create.argtypes=[ctypes.c_void_p];self.api.lorentz_sigma_create.restype=ctypes.c_void_p
        self.api.lorentz_sigma_scale.argtypes=[ctypes.c_void_p,ctypes.c_double];self.api.lorentz_sigma_scale.restype=ctypes.c_int
        self.api.lorentz_sigma_destroy.argtypes=[ctypes.c_void_p]
        for name in ['lorentz_sigma_array_count','lorentz_sigma_state_count']:
            getattr(self.api,name).argtypes=[ctypes.c_void_p];getattr(self.api,name).restype=ctypes.c_int
        self.fields_address=int(sim.fields.this)
        self.handle=self.api.lorentz_sigma_create(ctypes.c_void_p(self.fields_address))
        if not self.handle:raise RuntimeError(self.api.lorentz_sigma_error().decode())
        self.metadata=dict(array_count=self.api.lorentz_sigma_array_count(self.handle),
            state_count=self.api.lorentz_sigma_state_count(self.handle),realnum_bytes=self.api.lorentz_realnum_size(),
            adapter='Version-pinned C++ sigma forcing update; native polarization history retained')
    def scale(self,factor):
        if not self.handle:raise RuntimeError('Closed Lorentz modulation handle')
        if self.sim.fields is None or int(self.sim.fields.this)!=self.fields_address:
            raise RuntimeError('Native Meep fields reset after Lorentz adapter creation')
        if self.api.lorentz_sigma_scale(self.handle,float(factor)):
            raise RuntimeError(self.api.lorentz_sigma_error().decode())
    def close(self):
        if self.handle:
            self.api.lorentz_sigma_destroy(self.handle);self.handle=None
