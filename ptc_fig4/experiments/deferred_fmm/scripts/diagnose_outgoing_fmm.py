"""Direct complex poles of actual cylinders: independent of Legume/GME.

Gamma A2g is continued by its eigenvector along M-Gamma-X-M. This is a
branch diagnostic, not a proof of full-spectrum completeness. Every pole
and all nullvectors/finite-layer fields are checkpointed; no peak scans.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from ptc_fig4.outgoing_fmm import OutgoingFMM,gamma_a2_basis,embed_center_field


def path_grid(segments):
    points=[np.array([np.pi,np.pi]),np.zeros(2),np.array([np.pi,0.]),np.array([np.pi,np.pi])]
    k=np.concatenate([np.linspace(points[i],points[i+1],segments,endpoint=False) for i in range(3)]+[[points[3]]])
    return k,[0,segments,2*segments,3*segments]


def run(cutoff,index,segments,output,previous=None,tolerance=2e-8):
    output.mkdir(parents=True,exist_ok=True)
    model=OutgoingFMM.cylinder(cutoff,epsilon=index**2)
    k,vertices=path_grid(segments)
    n=len(k)
    gamma_index=segments
    conversion=299792458/(2*np.pi*3296e-9)/1e12
    source_hash=hashlib.sha256(Path(__import__('ptc_fig4.outgoing_fmm',fromlist=['x']).__file__).read_bytes()).hexdigest()
    metadata=dict(cutoff=cutoff,square_G_count=model.gvec.shape[1],index=index,
                  epsilon=index**2,height_over_a=model.height,k_points=n,source_sha256=source_hash,
                  approach="direct complex-frequency boundary zeros; exact outgoing air; Fourier-truncated xy",
                  branch="Gamma A2g, Et even in z; continuous nullvector continuation",
                  full_spectrum_complete=False,tolerance=tolerance)
    manifest=output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=metadata:
        raise RuntimeError("Checkpoint configuration or source differs; use a fresh directory")
    manifest.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    seed=4.7-.001j
    v=None
    if previous:
        with np.load(previous/'pole_gamma_seed.npz') as p:
            seed=complex(p['omega'])
            v=gamma_a2_basis(model.gvec).conj().T@embed_center_field(p['e_center'],p['gvec'],model.gvec)
    seedfile=output/'pole_gamma_seed.npz'
    if seedfile.exists():
        with np.load(seedfile) as p:
            gw=complex(p['omega']);gv=p['e_center']
    else:
        gm=model.find_mode(seed,a2_gamma=True,initial_vector=v,tolerance=tolerance,search_radius=.8)
        gw=gm.frequency;gv=gamma_a2_basis(model.gvec)@gm.right
        np.savez_compressed(seedfile,omega=gw,e_center=gv,gvec=model.gvec,
                            left_reduced=gm.left,residual=gm.relative_residual,iterations=gm.iterations)
    nodes,weights=np.polynomial.legendre.leggauss(24)
    z=nodes*model.height/2;weights=weights*model.height/2
    frequencies=np.zeros(n,complex);qs=np.zeros(n);residuals=np.zeros(n);overlaps=np.ones(n)
    snapshots={}
    route=[gamma_index]+list(range(gamma_index-1,-1,-1))+list(range(gamma_index+1,n))
    previous_w,previous_v,previous_fields=gw,gv,None
    t0=time.perf_counter()
    for ik in route:
        if ik==gamma_index+1:
            previous_w,previous_v=gw,gv
            previous_fields=snapshots[gamma_index]
        checkpoint=output/f'pole_k{ik:03d}.npz'
        if checkpoint.exists():
            with np.load(checkpoint) as p:
                w=complex(p['omega']);v=p['right'];e=p['E'];h=p['H'];res=float(p['residual'])
        else:
            mode=model.find_mode(previous_w,k[ik],initial_vector=previous_v,
                                 tolerance=tolerance,search_radius=.22)
            w,v,res=mode.frequency,mode.right,mode.relative_residual
            if w.imag>2e-7:
                raise RuntimeError(f"Passive structure yielded growing pole {w}; inspect outgoing sheet")
            e,h=model.even_fields(w,k[ik],v,z)
            np.savez_compressed(checkpoint,omega=w,right=v,left=mode.left,kpoint_ka=k[ik],
                                gvec=model.gvec,E=e,H=h,z_over_a=z,z_weights=weights,
                                residual=res,iterations=mode.iterations,
                                derivative_pairing=mode.derivative_pairing)
        field=(h*np.sqrt(weights)[:,None,None]).ravel()
        field/=np.linalg.norm(field)
        if previous_fields is not None:
            overlaps[ik]=abs(np.vdot(previous_fields,field))**2
        snapshots[ik]=field
        frequencies[ik]=w;residuals[ik]=res
        qs[ik]=w.real/(-2*w.imag) if w.imag< -1e-10 else np.inf
        previous_w,previous_v,previous_fields=w,v,field
        print(f'N={cutoff} k={ik:02d}/{n-1} f={w.real*conversion:.8f} THz Im={w.imag:.3g} overlap={overlaps[ik]:.4f}',flush=True)
    np.savez_compressed(output/'bands.npz',kpoints_ka=k,omega_a_over_c=frequencies,
                        frequency_thz=frequencies.real*conversion,q_radiation=qs,
                        residuals=residuals,layer_H_overlap_squared=overlaps,path_indexes=vertices)
    report=dict(status="APPROXIMATE_FOURIER_TRUNCATION_DIRECT_OUTGOING_POLES",
                **metadata,vertex_frequency_thz={label:float(frequencies[i].real*conversion) for label,i in zip(['M_start','Gamma','X','M_end'],vertices)},
                bandwidth_thz=float(np.ptp(frequencies.real)*conversion),
                relative_bandwidth=float(np.ptp(frequencies.real)/frequencies[gamma_index].real),
                maximum_residual=float(residuals.max()),minimum_layer_H_overlap_squared=float(overlaps.min()),
                M_closure_error=float(abs(frequencies[0]-frequencies[-1])),
                elapsed_seconds=time.perf_counter()-t0,
                limitations=["square Fourier cutoff requires convergence", "single tracked pole family, not full spectrum",
                             "constant real material", "matrix propagation conditioning at high evanescent order"])
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs',type=int,nargs='+',default=[2,3,4,5])
    parser.add_argument('--index',type=float,default=4.)
    parser.add_argument('--segments',type=int,default=8)
    parser.add_argument('--output',type=Path,default=Path('results/material_vs_solver/outgoing_fmm'))
    args=parser.parse_args()
    previous=None
    for cutoff in args.cutoffs:
        output=args.output/f'n{args.index:.6f}_G{cutoff}_k{3*args.segments+1}'
        report=run(cutoff,args.index,args.segments,output,previous=previous,tolerance=2e-7 if cutoff>=5 else 2e-8)
        print(json.dumps(report,indent=2),flush=True)
        previous=output


if __name__=='__main__': main()
