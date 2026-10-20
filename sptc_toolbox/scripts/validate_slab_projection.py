"""Native fixed-reference static controls for general slab-region projection."""
from __future__ import annotations
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
import argparse,hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
import legume
import numpy as np

from sptc_toolbox.slab_projection import project_slab,SlabProjection,SlabRegion
from lpor_71318.legume_basis import solve_native
from lpor_71318.field_projection import project_cylinder,projected_static_frequencies


def geometry(multilayer,delta):
    p=legume.PhotCryst(legume.Lattice('square'),eps_l=1.,eps_u=1.)
    p.add_layer(d=.42,eps_b=11.5+delta[2]+delta[3])
    p.add_shape(legume.Circle(eps=1+delta[0]+delta[3],x_cent=-.22,r=.12))
    p.add_shape(legume.Ellipse(eps=1+delta[1]+delta[3],x_cent=.22,rx=.12,ry=.09))
    if multilayer:
        p.add_layer(d=.18,eps_b=6.+delta[5])
        p.add_shape(legume.Poly(eps=2.+delta[4],x_edges=[-.14,.14,.14,-.14],y_edges=[-.1,-.1,.1,.1]),layer=1)
    for layer,eps in zip(p.layers,[10.,5.5]):layer.eps_eff=eps
    for clad in p.claddings:clad.eps_eff=1.
    return p


def native(multilayer,delta=None,numeig=10):
    delta=np.zeros(6) if delta is None else np.asarray(delta)
    g=legume.GuidedModeExp(geometry(multilayer,delta),gmax=2.1,truncate_g='abs')
    g.run(kpoints=np.array([[.41],[.19]]),gmode_inds=[0,1,2,3],numeig=numeig,
        eps_eff='custom',gmode_compute='exact',gmode_tol=1e-10,compute_im=False,verbose=False)
    return g


def physical_h(g,modes):
    # Identical physical z/G grid across native perturbations; this is sampled
    # field identity, not a QNM norm or an all-space integration certificate.
    points=np.linspace(-.25,.85,23)
    return np.asarray([[g.ft_field_xy('H',0,m,float(z)) for z in points] for m in modes])


def overlaps(first,second):
    a=first.reshape(len(first),-1);b=second.reshape(len(second),-1)
    return abs(np.sum(a.conj()*b,axis=1))**2/(np.sum(abs(a)**2,axis=1)*np.sum(abs(b)**2,axis=1))


def case(multilayer,out):
    modes=[2,3,4,5];g=native(multilayer);h=physical_h(g,modes)
    rs=[SlabRegion(0,0,name='circle'),SlabRegion(0,1,name='ellipse'),
        SlabRegion(0,kind='background',name='background'),SlabRegion(0,name='whole layer')]
    if multilayer:rs.extend([SlabRegion(1,0,name='polygon'),SlabRegion(1,kind='background',name='upper background')])
    projection=project_slab(g,modes,rs,z_order=24)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'projection.npz',**projection.arrays)
    (out/'projection_metadata.json').write_text(json.dumps(projection.metadata_dict(),indent=2),encoding='utf-8')
    derivative_records=[];raw={}
    for ir,region in enumerate(rs):
        delta=np.zeros(6);delta[ir]=.002
        plus,minus=native(multilayer,delta),native(multilayer,-delta)
        actual=(plus.freqs[0,modes]-minus.freqs[0,modes])/.004
        expected=-g.freqs[0,modes]*np.diag(projection.electric_overlap_regions[ir]).real/2
        identity=np.minimum(overlaps(h,physical_h(plus,modes)),overlaps(h,physical_h(minus,modes)))
        error=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(actual),1e-300))
        derivative_records.append(dict(region=region.name,step=.002,native_frequency_derivative=actual.tolist(),
            projected_frequency_derivative=expected.tolist(),relative_error=error,
            minimum_sampled_H_overlap_squared=float(identity.min())))
        raw[f'region{ir}_plus_frequencies']=plus.freqs[0];raw[f'region{ir}_minus_frequencies']=minus.freqs[0]
    change=np.array([.08,-.04,.025,.015,.05,-.03])[:len(rs)]
    padded=np.zeros(6);padded[:len(rs)]=change
    complete=native(multilayer,padded,numeig=len(g.eigvecs[0]))
    eigenvectors=complete.eigvecs[0];baselinevectors=g.eigvecs[0][:,modes]
    independent=(eigenvectors*(2*np.pi*complete.freqs[0])**2)@eigenvectors.conj().T
    independent=baselinevectors.conj().T@independent@baselinevectors
    omega=2*np.pi*g.freqs[0,modes]
    projected=projection.stiffness(omega,change)
    matrix_error=float(np.linalg.norm(projected-independent)/np.linalg.norm(independent))
    eta_error=float(np.linalg.norm(projection.inverse_dielectric(change)-np.asarray(complete.eps_inv_mat)[1:-1]))
    reduced_f=np.sqrt(np.linalg.eigvalsh(projected))/(2*np.pi)
    full_f=complete.freqs[0,modes]
    zero_error=float(np.linalg.norm(projection.stiffness(omega,np.zeros(len(rs)))-np.diag(omega**2)))
    half_f=[]
    for factor in [1.,.5,.25]:
        changed=native(multilayer,padded*factor)
        reduced=np.sqrt(np.linalg.eigvalsh(projection.stiffness(omega,change*factor)))/(2*np.pi)
        half_f.append(dict(perturbation_factor=factor,
            reduced_vs_full_native_max_absolute_frequency_error=float(np.max(abs(reduced-changed.freqs[0,modes])))))
    fine=projection;coarse=project_slab(g,modes,rs,z_order=16)
    quadrature=float(np.linalg.norm(fine.stiffness(omega,change)-coarse.stiffness(omega,change))/np.linalg.norm(projected))
    dot=np.linspace(-.03,.04,len(rs));step=1e-5
    numerical=(projection.stiffness(omega,change+step*dot)-projection.stiffness(omega,change-step*dot))/(2*step)
    analytic=projection.stiffness_derivative(omega,change,dot)
    derivative_error=float(np.linalg.norm(numerical-analytic)/max(np.linalg.norm(analytic),1e-300))
    restored=SlabProjection.from_arrays(dict(projection.arrays),projection.metadata_dict())
    roundtrip=float(np.linalg.norm(restored.stiffness(omega,change)-projected))
    np.savez_compressed(out/'native_comparison.npz',baseline_frequencies=g.freqs[0],
        baseline_vectors=g.eigvecs[0],baseline_H=h,gvec=g.gvec,kpoint=g.kpoints,
        delta=change,native_complete_frequencies=complete.freqs[0],native_complete_vectors=eigenvectors,
        native_projected_matrix=independent,adapter_projected_matrix=projected,
        reduced_frequencies=reduced_f,native_selected_frequencies=full_f,**raw)
    report=dict(geometry='two-layer Circle/Ellipse holes plus Poly inclusion' if multilayer else 'single-layer two independent off-center Circle/Ellipse holes',
        gmax=2.1,guided_reference_permittivity=[10.,5.5] if multilayer else [10.],
        guided_mode_indices=[0,1,2,3],kpoint_ka=[.41,.19],selected_modes=modes,
        native_guided_basis_dimension=len(g.eigvecs[0]),retained_modal_dimension=len(modes),
        native_basis_id=projection.basis_id,physical_signature=projection.physical_signature,
        dielectric_reconstruction=[r['dielectric_reconstruction_relative_error'] for r in projection.metadata['layer_reports']],
        region_derivatives=derivative_records,finite_delta=change.tolist(),
        finite_projected_operator_relative_error=matrix_error,
        finite_native_inverse_dielectric_absolute_error=eta_error,
        reduced_vs_full_finite_native_frequency_difference=(reduced_f-full_f).tolist(),
        retained_mode_truncation_diagnostic=half_f,
        z16_to_z24_relative_stiffness_change=quadrature,
        time_derivative_finite_difference_relative_error=derivative_error,
        unmodulated_stiffness_error=zero_error,serialization_roundtrip_error=roundtrip)
    checks=dict(native_dielectric=max(report['dielectric_reconstruction'])<2e-8,
        region_static_derivatives=max(r['relative_error'] for r in derivative_records)<2e-5,
        sampled_field_identity=min(r['minimum_sampled_H_overlap_squared'] for r in derivative_records)>.9999,
        finite_native_operator=matrix_error<2e-9,finite_native_dielectric=eta_error<2e-9,
        quadrature=quadrature<2e-11,time_derivative=derivative_error<1e-6,
        zero_perturbation=zero_error<2e-13,serialization=roundtrip==0.)
    report['checks']=checks
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/slab_projection');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    records=[case(False,a.output/'two_inclusions'),case(True,a.output/'multilayer')]
    g=solve_native(gmax=2.1,vertical_modes=4,reference='custom',reference_epsilon=16.,compute_im=False,numeig=8)
    old=project_cylinder(g,[2,3],z_order=16);new=project_slab(g,[2,3],[dict(layer=0,shape_index=0)],z_order=16)
    olderrors=[]
    for delta in [-.2,0,.1]:
        oldfreq=2*np.pi*projected_static_frequencies(g.freqs[0,[2,3]],old,delta)
        newfreq=np.sqrt(np.linalg.eigvalsh(new.stiffness(2*np.pi*g.freqs[0,[2,3]],[delta])))
        olderrors.append(float(np.max(abs(oldfreq-newfreq))))
    checks=all(all(r['checks'].values()) for r in records) and max(olderrors)<3e-14
    result=dict(status='VERIFIED_NATIVE_FINITE_BASIS_MULTIREGION_SLAB_PROJECTION' if checks else 'FAILED_CONTROL',
        all_checks_pass=checks,cases=records,old_centered_circle_frequency_errors=olderrors,
        old_centered_circle_overlap_error=float(np.linalg.norm(new.electric_overlap_regions[0]-old['electric_overlap'])),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'sptc_toolbox/slab_projection.py',ROOT/'sptc_toolbox/tests/test_slab_projection.py',Path(__file__)]},
        scope='Native Legume finite-layer material projection; actual independent fixed-reference static solves, not paper tuning or a new spatial solver',
        limitations=['The retained native spatial/modal basis is finite and unrefined here; exact operator projection does not make truncated modal eigenvalues exact.',
            'The direct native reconstruction uses the same fixed guided reference, so this validates adapter algebra/physical masks and field normalization, not independent spatial-basis convergence.',
            'Geometry overlap certification is conservative: intersecting noncircular bounding rectangles are rejected even when actual shapes might be disjoint.',
            'Physical field overlap uses a common finite z/G sample metric, not a QNM residue or all-space norm.',
            'Cladding modulation, complex/negative instantaneous epsilon, dispersive memory and open Floquet self-energy are outside this material-projection contract.'])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Native multi-region slab projection validation','',
        f"Status: {result['status']}. Native geometry Fourier coefficients and native D fields are reused; installed Legume is unchanged.",'',
        'The adapter handles native Circle, Ellipse and Poly-derived shapes in multiple finite layers. Shape, whole-layer and background modulation contributions add before a single exact finite dielectric-matrix inverse. Physical geometry must satisfy the declared disjointness guard; background excludes every shape, including unmodulated ones.', '',
        'Legume1.0.3 uses epsilon[row,column]=FT(G_column−G_row). Off-center asymmetric shapes expose the sign; the regression verifies the full complex projected matrix, not only frequencies of a centered symmetric structure. Native D already includes1/omega_mode and1/sqrt(cellarea); the quadrature factors add only sqrt(cellarea*weight_z).', '',
        '| Case | Max native dielectric error | Max region derivative error | Minimum sampled H overlap² | Finite operator error |',
        '|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['geometry']} | {max(r['dielectric_reconstruction']):.4g} | {max(v['relative_error'] for v in r['region_derivatives']):.4g} | {min(v['minimum_sampled_H_overlap_squared'] for v in r['region_derivatives']):.10f} | {r['finite_projected_operator_relative_error']:.4g} |")
    lines+=['',f'Historical centered-cylinder finite-amplitude frequency match errors are {olderrors}; electric-overlap error is {result["old_centered_circle_overlap_error"]:.4g}.', '',
        'Finite multi-region perturbations are independently recomputed by native GME with its complete returned guided-basis eigenpairs. That spatial matrix is reconstructed and projected onto the original selected modes, without the adapter D/mask formulas. Differences between the four-mode projected eigenfrequencies and the full new native eigenfrequencies are retained separately as modal-truncation diagnostics; they are not fitted away.', '',
        'The state is a scaled displacement coordinate u, with physical D=sum omega_i D_i u_i. The equation is not simply an H-mode amplitude equation with a time-dependent coefficient; material time derivatives belong to the chosen Maxwell variable formulation.', '',
        'Before/after native state hashes, ordered basis identity and physical signature are unchanged. Known generic-runner lineage is checked before export. Native eigenvector orthonormality is required rather than repaired by silent reweighting.', '', 'Limitations:', '']+['- '+s for s in result['limitations']]
    lines+=['','Independent scientific contract: `sptc_toolbox/docs/slab_projection_contract.md`. Machine-readable native comparisons and raw projection arrays are stored in each case folder.','']
    (a.output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],old_match=max(olderrors),case_checks=[r['checks'] for r in records],finite_errors=[r['finite_projected_operator_relative_error'] for r in records]),indent=2))
    if not checks:raise RuntimeError('Native slab projection validation failed')


if __name__=='__main__':main()
