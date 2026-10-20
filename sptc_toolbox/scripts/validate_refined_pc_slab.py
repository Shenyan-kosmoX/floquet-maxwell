"""Continue the real patterned-slab workflow on the independently refined path."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,pickle,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.slab_study import solve_slab_bloch
from sptc_toolbox.mode_package import ModePackage
from sptc_toolbox.slab_model import SlabMaterialModel
from sptc_toolbox.slab_fields import reconstruct_from_spectrum
BASE=ROOT/'sptc_toolbox/results/pc_slab_pipeline'
OUT=BASE/'refined_floquet'
REFINED=BASE/'refined_path_current'


def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,(complex,np.complexfloating)):return [float(x.real),float(x.imag)]
    if isinstance(x,np.generic):return x.item()
    return x


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=REFINED/'native.pkl'
    with source.open('rb') as f:g=pickle.load(f)
    tracking=json.loads((REFINED/'tracking.json').read_text())
    records=tracking['target']['records'];selection=[r['modes'] for r in records]
    identity=[r['identity_supported_from_seed'] for r in records]
    regions=[SlabRegion(0,0,name='circular-hole'),SlabRegion(0,1,name='elliptical-hole')]
    protocol=RegionModulation.sinusoidal(.08,phases=[0,np.pi/2],region_labels=('circular-hole','elliptical-hole'),lattice_basis=np.eye(2))
    drive=4*np.pi*float(np.mean(g.freqs[0,[2,3]]))
    results=solve_slab_bloch(g,regions,protocol,drive,selection,identity_supported=identity,reference_checks=(0,1,2,3))
    saved=[];pointdata=[]
    for row in results:
        item={k:v for k,v in row.items() if k not in ('package','model','spectrum')}
        if 'spectrum' in row:
            folder=OUT/f'k{row["k_index"]:02d}';folder.mkdir(exist_ok=True)
            row['package'].save(folder/'mode_package.npz');row['spectrum'].save(folder/'floquet.npz')
            spectrum=row['spectrum'];v=spectrum.eigenvalues[row['physical_indices']]
            item['quasifrequencies']=v
            pointdata.extend([[row['k_index'],w.real,w.imag] for w in v])
        saved.append(item)
    # Separate physical relative phase from a global origin-of-time change.
    package=results[0]['package'];phis=np.linspace(0,2*np.pi,33);coupling=[]
    spectrum=results[0]['spectrum'];physical=results[0]['physical_indices']
    branch=int(physical[np.argmax(spectrum.eigenvalues[physical].real)])
    times=np.array([0.,np.pi/2,np.pi,3*np.pi/2])
    field_records=[reconstruct_from_spectrum(g,package.slab_projection(),protocol,drive,spectrum,branch,t,[.1,.21,.35],
                   spectrum_basis_id=package.basis_id) for t in times]
    np.savez_compressed(OUT/'floquet_fields.npz',tau=times,z=[.1,.21,.35],
        E=np.asarray([r['E'] for r in field_records]),H=np.asarray([r['H'] for r in field_records]),D=np.asarray([r['D'] for r in field_records]),
        gvec=g.gvec,kpoint=g.kpoints[:,0],quasifrequency=spectrum.eigenvalues[branch],
        spectrum_index=branch,basis_id=package.basis_id)
    for phi in phis:
        mod=RegionModulation.sinusoidal(.08,phases=[0,phi],region_labels=protocol.region_labels)
        material=SlabMaterialModel(package.slab_projection(),mod,drive)
        tau=2*np.pi*np.arange(64)/64
        block=np.fft.ifft(np.asarray([material.stiffness(t) for t in tau]),axis=0)[1]
        coupling.append([abs(block[0,1]),abs(block[0,0]),abs(block[1,1])])
    np.savez_compressed(OUT/'overview.npz',kpoint_ka=g.kpoints,native_frequency_fa_over_c=g.freqs,
        path_position=np.r_[0,np.cumsum(np.linalg.norm(np.diff(g.kpoints,axis=1),axis=0))],
        identity_supported=identity,selected_modes_flat=np.concatenate(selection),
        selected_modes_offsets=np.r_[0,np.cumsum([len(ids) for ids in selection])],
        floquet_points=np.asarray(pointdata),relative_phase=phis,stiffness_harmonic_amplitudes=coupling)
    summary=dict(status='REFINED_PATTERNED_SLAB_FLOQUET_WORKFLOW_EXECUTED_WITH_SCOPE_GATES',
        actual_k_points=len(g.freqs),seed_supported_points=sum(identity),
        solved_points=sum('spectrum' in r for r in results),fixed_drive_omega_a_over_c=drive,
        results=saved,whole_bz_certified=False,physical_bic_ep_topology_certified=False,
        spatial_refinement_reference='../summary.json',
        notes=['Band identity and frozen-bath suitability are independent gates','Local solves retain positive/negative Floquet states','Native bath frequency is common across retained modes at each k; sideband radiative self-energy is not supplied','Weak positive rates alone do not certify physical amplification against spatial/radiative approximation error'],
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),source,REFINED/'tracking.json',ROOT/'sptc_toolbox/slab_model.py',ROOT/'sptc_toolbox/slab_projection.py',ROOT/'sptc_toolbox/mode_package.py',ROOT/'sptc_toolbox/slab_study.py',ROOT/'sptc_toolbox/modulation.py']})
    (OUT/'summary.json').write_text(json.dumps(plain(summary),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','actual_k_points','seed_supported_points','solved_points']},indent=2))


if __name__=='__main__':main()
