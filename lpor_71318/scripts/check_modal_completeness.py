"""Check omitted high-frequency A2g modes in the native Legume projection."""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legume_basis import solve_native, classify_gamma_h, MetasurfaceGeometry
from field_projection import project_cylinder, weak_modulation_coefficient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gmax',type=float,default=7.1)
    parser.add_argument('--vertical-modes',type=int,default=8)
    parser.add_argument('--numeig',type=int,default=80)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=args.output or ROOT/'results/modal_completeness'/f'custom_g{args.gmax:g}_v{args.vertical_modes}_m{args.numeig}'
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    geometry=MetasurfaceGeometry()
    gme=solve_native(gmax=args.gmax,vertical_modes=args.vertical_modes,reference='custom',
                     reference_epsilon=16.,compute_im=False,numeig=args.numeig)
    nodes,weights=np.polynomial.legendre.leggauss(6)
    z,ww=[],[]
    for left,right in [(-.3,0.),(0.,geometry.height),(geometry.height,geometry.height+.3)]:
        z.extend((left+right)/2+(right-left)*nodes/2)
        ww.extend((right-left)*weights/2)
    z,ww=np.asarray(z),np.asarray(ww)
    h=np.asarray([[gme.ft_field_xy('H',0,im,zi) for zi in z] for im in range(args.numeig)])
    omega=2*np.pi*np.asarray(gme.freqs[0])
    classification=classify_gamma_h(h,gme.gvec,ww,omega)
    modes=[v['mode_indices'][0] for v in classification if v['irrep']=='A2g' and v['dimension']==1]
    if not modes or modes[0]!=2:
        raise RuntimeError('Lowest ring A2g seed identity changed; inspect before coefficient comparison')
    (output/'gamma_symmetries.json').write_text(json.dumps(classification,indent=2,allow_nan=False),encoding='utf-8')
    projection=project_cylinder(gme,modes,z_order=24)
    conversion=299792458./(geometry.period_nm*1e-9)/geometry.modulation_angular_frequency_per_second
    w=omega[modes]*conversion
    np.savez_compressed(output/'modal_projection.npz',all_omega_a_over_c=omega,all_frequencies_thz=gme.freqs[0]*299792458./(geometry.period_nm*1e3),
                        A2g_modes=np.asarray(modes),A2g_omega_over_Omega=w,
                        eigvals_chi=projection['eigvals_chi'],displacement_weights=projection['displacement_weights'],
                        electric_overlap=projection['electric_overlap'],epsilon=projection['epsilon'])
    counts=sorted(set([1,min(3,len(modes)),len(modes)]))
    convergence=[]
    for count in counts:
        small=dict(projection)
        small['modes']=projection['modes'][:count]
        small['displacement_weights']=projection['displacement_weights'][:,:count,:count]
        small['electric_overlap']=projection['electric_overlap'][:count,:count]
        c=weak_modulation_coefficient(w[:count],small,target=0)
        convergence.append({'count':count,'included_GME_modes':modes[:count],
                            'highest_omega_over_Omega':float(w[count-1]),'coefficient':c})
    base3=next(v['coefficient'] for v in convergence if v['count']==min(3,len(modes)))
    full=convergence[-1]['coefficient']
    report={'status':'VERIFIED_WITHIN_NATIVE_LEGUME_MODAL_PROJECTION',
            'does_not_verify':'Open-QNM normalization or radiation continuum completeness',
            'gmax':args.gmax,'vertical_orders':args.vertical_modes,'stored_native_modes':args.numeig,
            'basis_dimension':len(gme.eigvecs[0]),'reference_epsilon':16.,'physical_epsilon':16.,
            'A2g_modes':modes,'A2g_frequencies_thz':(gme.freqs[0,modes]*299792458./(geometry.period_nm*1e3)).tolist(),
            'modal_convergence':convergence,'three_to_all_absolute_change':full-base3,
            'three_to_all_relative_change':(full-base3)/base3,
            'paper_coefficient_approximate_readback':.27,'paper_to_projected_coefficient_ratio':.27/full,
            'dielectric_reconstruction_error':projection['dielectric_reconstruction_error'],
            'z_quadrature_order':24,'elapsed_seconds':time.perf_counter()-started,
            'interpretation':'Tests omission of the extra stored A2g eigenmodes at fixed GME basis; it does not prove completeness of the guided-mode approximation.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Native modal-window completeness check','','The physical geometry and epsilon=16 are fixed. Native Legume custom reference epsilon=16 is fixed. No independent spatial solver is used.','','| Retained A2g modes | Native mode indices | C in Δ(ω/Ω)=C ε_mod² |','|---:|---|---:|']
    for v in convergence:lines.append(f"| {v['count']} | {v['included_GME_modes']} | {v['coefficient']:.12f} |")
    lines+=['',f"Expanding from three A2g modes to all {len(modes)} among {args.numeig} saved native eigenmodes changes C by {100*report['three_to_all_relative_change']:.6f}%. The paper curve coefficient estimated near 0.27 is still {report['paper_to_projected_coefficient_ratio']:.3f} times the calculated value.",'','This controls the high-frequency modal window within the selected guided reference. It does not establish completeness of the guided radiation basis or validate outgoing resonant-state normalization.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
