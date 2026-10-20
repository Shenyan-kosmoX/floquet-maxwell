"""Actual Legume slab fields for Jin2025 Supplementary Figure 5.

Static geometry is the stated finite AlGaAs slab, with silica/air claddings.
This script never substitutes a fitted coupling or an analytical nodal map.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import time
import legume
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'sptc_toolbox/results/jin2025_legume_tensor_20261019'


def write_json(path,data):
    def conv(x):
        if isinstance(x,np.ndarray):return conv(x.tolist())
        if isinstance(x,(complex,np.complexfloating)):return [float(x.real),float(x.imag)]
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,dict):return {str(k):conv(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [conv(v) for v in x]
        return x
    Path(path).write_text(json.dumps(conv(data),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def geometry(index):
    phc=legume.PhotCryst(legume.Lattice('square'),eps_l=1.46**2,eps_u=1.)
    phc.add_layer(d=85/465,eps_b=index**2)
    phc.add_shape(legume.Circle(eps=1.,r=50/465))
    phc.layers[0].eps_eff=index**2-(index**2-1)*np.pi*(50/465)**2
    for clad,eps in zip(phc.claddings,[1.46**2,1.]):clad.eps_eff=eps
    return phc


def native(points,cutoff,index,numeig=16):
    g=legume.GuidedModeExp(geometry(index),gmax=cutoff,truncate_g='abs')
    g.run(kpoints=-2*np.pi*np.asarray(points).T,gmode_inds=list(range(6)),
          numeig=numeig,eps_eff='custom',gmode_compute='exact',compute_im=False,verbose=False)
    return g


def probe(args):
    from sptc_toolbox.traveling_slab import physical_field_coefficients
    args.output.mkdir(parents=True,exist_ok=True)
    points=np.array([[.001,.001],[.02,0],[0,.02],[.02,.02],[.04,.04]])
    g=native(points,args.gmax,args.index)
    (args.output/'probe_native.pkl').write_bytes(pickle.dumps(g))
    rows=[]
    for kind,k in enumerate(points):
        character=[]
        for mode in range(len(g.freqs[kind])):
            ez=physical_field_coefficients(g,'E',kind,mode,85/465/2)
            hz=physical_field_coefficients(g,'H',kind,mode,85/465/2)
            character.append(dict(mode=mode,frequency=float(g.freqs[kind,mode]),
                wavelength_nm=float(465/g.freqs[kind,mode]),
                Ez_fraction=float(np.sum(abs(ez[2])**2)/np.sum(abs(ez)**2)),
                Hz_fraction=float(np.sum(abs(hz[2])**2)/np.sum(abs(hz)**2))))
        rows.append(dict(k=k,character=character))
    write_json(args.output/'mode_probe.json',dict(index=args.index,gmax=args.gmax,rows=rows))
    print(json.dumps(rows,default=lambda x:x.tolist()),flush=True)


def select_modes(g,kind):
    from sptc_toolbox.traveling_slab import physical_field_coefficients
    character=[]
    for mode,f in enumerate(g.freqs[kind]):
        e=physical_field_coefficients(g,'E',kind,mode,85/465/2)
        h=physical_field_coefficients(g,'H',kind,mode,85/465/2)
        character.append(dict(mode=mode,frequency=float(f),
            Ez_fraction=float(np.sum(abs(e[2])**2)/np.sum(abs(e)**2)),
            Hz_fraction=float(np.sum(abs(h[2])**2)/np.sum(abs(h)**2))))
    te=[r['mode'] for r in character if .4<r['frequency']<.53 and r['Hz_fraction']>.8]
    tm=[r['mode'] for r in character if .54<r['frequency']<.74 and r['Ez_fraction']>.4]
    if len(te)<2 or not tm:raise ValueError(f'Identified source TE/TM windows not supplied at k{kind}: {character}')
    chosen=[*te[-2:],tm[0]]
    return chosen,character


def scan(args):
    from sptc_toolbox.tensor_projection import project_tensor_slab
    from sptc_toolbox.slab_projection import SlabRegion
    start=time.perf_counter()
    dest=args.output/f'n{args.index:g}_g{args.gmax:g}_N{args.grid_size}'
    dest.mkdir(parents=True,exist_ok=True)
    axis=np.linspace(0,.04,args.grid_size)
    allpoints=np.array([(x,y) for x in axis for y in axis])
    # The paper notes an individual TE mode is undefined at Gamma degeneracy.
    valid=np.linalg.norm(allpoints,axis=1)>0
    points=allpoints[valid]
    path=dest/'native.pkl'
    if path.exists():g=pickle.loads(path.read_bytes())
    else:
        g=native(points,args.gmax,args.index)
        path.write_bytes(pickle.dumps(g))
    records=[];out=[];modes=[];freq=[];mblocks=[]
    tx=np.zeros((1,3,3),complex);tx[0,1,2]=tx[0,2,1]=1
    ty=np.zeros((1,3,3),complex);ty[0,0,2]=ty[0,2,0]=1
    for kind,k in enumerate(points):
        selected,character=select_modes(g,kind)
        p=project_tensor_slab(g,kind,selected,[SlabRegion(0,kind='background')],z_order=args.z_order)
        x=p.mass_fourier({1:tx,-1:tx.conj()})[1]
        y=p.mass_fourier({1:ty,-1:ty.conj()})[1]
        # TM row, higher-TE column, same actual native mode gauge for both pumps.
        out.append([x[2,1],y[2,1]])
        mblocks.append(np.stack([x,y]));modes.append(selected);freq.append(np.asarray(g.freqs)[kind,selected])
        records.append(dict(k=k,modes=selected,character=[character[i] for i in selected],
            minimum_TE_splitting=float(g.freqs[kind,selected[1]]-g.freqs[kind,selected[0]])))
        if (kind+1)%50==0:print(json.dumps(dict(done=kind+1,total=len(points),seconds=time.perf_counter()-start)),flush=True)
    out=np.asarray(out)
    shaped=np.full((len(axis),len(axis),2),np.nan+1j*np.nan)
    shaped.reshape(-1,2)[valid]=out
    np.savez_compressed(dest/'coupling.npz',axis=axis,points=points,valid=valid,
        Vxy=shaped,raw_pairs=out,selected_modes=modes,frequencies=freq,
        tensor_mass_blocks=np.asarray(mblocks),n_AlGaAs=args.index,gmax=args.gmax,z_order=args.z_order)
    write_json(dest/'summary.json',dict(status='ACTUAL_LEGUME_TENSOR_FIELDS_FOR_PUBLISHED_SFIG5',completed=True,
        paper='10.1038/s41565-025-02003-1',figure='Official Supplementary Figure5(a-d)',
        geometry_nm=dict(period=465,thickness=85,hole_diameter=100),
        index=args.index,substrate_index=1.46,top_index=1,gmax=args.gmax,z_order=args.z_order,
        grid_size=args.grid_size,selected_modes=records,seconds=time.perf_counter()-start,
        native_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        couplings_sha256=hashlib.sha256((dest/'coupling.npz').read_bytes()).hexdigest(),
        tensor_scale='unit Fourier tensor yz+zy or xz+zx in AlGaAs; all polarization combinations same mode gauge',
        normalization='Native Legume E-field normalization; plotted node structure independent of common pump scale; no fit',
        limits=['Nondispersive index from stated preprint material value, not a full verified final-paper dispersion curve.',
            'This is field-derived modulation coupling, not full open QNM spectrum or disorder validation.',
            'Gamma individual TE mode intentionally not selected; zero-point coupling is NaN.']))
    print(json.dumps(dict(completed=str(dest),seconds=time.perf_counter()-start)),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--gmax',type=float,default=3.1)
    p.add_argument('--index',type=float,default=3.47)
    p.add_argument('--probe',action='store_true')
    p.add_argument('--grid-size',type=int,default=25)
    p.add_argument('--z-order',type=int,default=16)
    args=p.parse_args()
    if args.grid_size < 5:
        p.error('--grid-size must be at least 5 for the axis/diagonal comparison')
    expected = args.output/'probe_native.pkl' if args.probe else args.output/f'n{args.index:g}_g{args.gmax:g}_N{args.grid_size}'
    if expected.exists():
        p.error('The selected result already exists; choose a new --output directory to preserve previous calculations')
    if args.probe:probe(args)
    else:scan(args)


if __name__=='__main__':main()
