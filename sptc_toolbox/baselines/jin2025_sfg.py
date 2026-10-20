"""Jin et al. Nature Nanotechnology 2025: open modal SFG and pulse benchmark.

Published measured static modes and cooperativity calibration are inputs.
This is a reproduction of the explicitly stated two-mode approximation, not
a Legume Maxwell/tensor reconstruction of the AlGaAs spatial structure.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from pathlib import Path
import argparse,csv,hashlib,json,re
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq,minimize_scalar
from scipy.integrate import simpson,solve_ivp
from sptc_toolbox.scattering import PeriodicStateSpace,solve_floquet_scattering

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'sptc_toolbox/literature/sptc_baselines/sources/jin_followup'
OUTPUT=ROOT/'sptc_toolbox/results/sptc_baselines/jin2025'
PORTS=('lower_top','lower_bottom','upper_top','upper_bottom')


def save_json(path,value):
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=convert),encoding='utf-8')


def published_inputs():
    books=json.loads((SOURCE/'source_data_parsed.json').read_text(encoding='utf-8'))
    rows=books['41565_2025_2003_MOESM3_ESM.xlsx']['Sheet1'][3:]
    def band(columns):
        points=[]
        for row in rows:
            row={re.sub('[0-9]','',k):v for k,v in row.items()}
            if all(row.get(k) is not None for k in columns):points.append([float(row[k]) for k in columns])
        a=np.array(points);return a[np.argsort(a[:,0])]
    upper,lower=band('BCD'),band('HIJ')
    # f=c/lambda in THz, k in 2pi/a; no optical frequency offsets fitted.
    c_nm_ps=299792.458;pump_nm=3050.
    high_f=interp1d(upper[:,0],c_nm_ps/upper[:,1]);low_f=interp1d(lower[:,0],c_nm_ps/lower[:,1])
    root=brentq(lambda k:float(high_f(k)-low_f(k)-c_nm_ps/pump_nm),.005,.01)
    linewidths=np.array([np.interp(root,lower[:,0],lower[:,2]),np.interp(root,upper[:,0],upper[:,2])])
    author=np.genfromtxt(SOURCE/'figure4_author_data.csv',delimiter=',',names=True)
    slopes=author['cooperativity'][1:]/author['pump_mW'][1:]
    assert np.ptp(slopes)<1e-12
    return dict(k_a_over_2pi=root,linewidths_GHz=linewidths,decay_per_ps=2*np.pi*linewidths/1000,
        lower_frequency_THz=float(low_f(root)),upper_frequency_THz=float(high_f(root)),
        pump_wavelength_nm=pump_nm,drive_per_ps=2*np.pi*c_nm_ps/pump_nm,C_per_mW=float(slopes.mean()),
        author=author,static_method='Linear interpolation of Fig2 TE_y/TM measured modes at f_TM-f_TE=c/3050nm; not fit to Fig4',
        coupling_method='Published Fig4 C/P calibration used as input; g=sqrt(C*kappa1*kappa2) uses independently selected Fig2 decays',
        linewidth_scope='Selected Fig2 decay pair is reproducible but not explicitly identified by authors as their Fig4/SFig7 pair')


def port_matrix(decay):
    k=np.asarray(decay,float)
    if k.shape!=(2,) or not np.isfinite(k).all() or np.any(k<=0):raise ValueError('Two positive amplitude decay rates required')
    matrix=np.zeros((4,2));matrix[:2,0]=np.sqrt(k[0]);matrix[2:,1]=np.sqrt(k[1])
    return matrix


def frequency_system(decay,g,drive,phase=0.):
    if not np.isfinite(g) or g<0 or not np.isfinite(drive) or drive<=0:raise ValueError('Finite nonnegative g and positive drive required')
    k=np.asarray(decay);ports=port_matrix(k)
    h0=np.diag([-1j*k[0],drive-1j*k[1]])
    hp=np.zeros((2,2),complex);hp[1,0]=-g*np.exp(-1j*phase)
    hm=hp.conj().T
    return PeriodicStateSpace({0:h0/drive,1:hp/drive,-1:hm/drive},
        {0:1j*ports.T/drive},{0:ports},{0:-np.eye(4)},PORTS,PORTS,
        metadata=dict(source='10.1038/s41565-025-02003-1, SI S39-S42',
            normalization_status='Photon amplitudes; b_j=a_j/sqrt(omega_j), S41 symmetric two-face radiative model',
            loss_ownership='All specified amplitude decay assigned to two equal external faces; no additional decay',
            physical_scope='Paper two-mode RWA with measured decay inputs; not a Maxwell geometry or QNM normalization claim',
            state_units='Time ps; carrier-removed lower frequency0, upper frequencyOmega; tau=Omega*t',
            reference_planes='Abstract paper TCMT lower/upper-frequency top/bottom ports'))


def floquet_response(decay,g,drive,detuning=0.,phase=0.,n=1):
    system=frequency_system(decay,g,drive,phase)
    result=solve_floquet_scattering(system,detuning/drive,n_harmonics=n,guard_harmonics=1,boundary_tolerance=1e-10)
    amplitudes=result.column('lower_top')
    upper=np.where(result.output_harmonics==1)[0][0]
    converted=float(np.sum(abs(amplitudes[upper,2:])**2))
    return converted,float(np.sum(abs(amplitudes)**2)),result


def analytic_efficiency(decay,g,detuning=0.):
    k1,k2=decay;d=np.asarray(detuning)
    return 2*k1*k2*g*g/abs((k1-1j*d)*(k2-1j*d)+g*g)**2


def pulse_response(decay,C0,tau_ps,delay_ps,*,probe_ps=.169,step=.012,rtol=2e-9,atol=2e-11,return_trace=False):
    from sptc_toolbox.transient import solve_transient_state_space,integrated_flux
    k=np.asarray(decay);g=np.sqrt(C0*np.prod(k));ports=port_matrix(k)
    envelope=lambda t:np.exp(-2*np.log(2)*((t-delay_ps)/tau_ps)**2)
    def h(t):return np.array([[-1j*k[0],-g*envelope(t)],[-g*envelope(t),-1j*k[1]]])
    def source(t):return np.array([np.exp(-2*np.log(2)*(t/probe_ps)**2),0.,0.,0.],complex)
    # Start seven probe intensity widths before excitation. Before this the
    # empty state has no pump-only source; extending far into its Gaussian
    # underflow tail can trigger meaningless RK error-norm underflow.
    start=-7*probe_ps;stop=max(7*probe_ps,delay_ps+4*tau_ps,18/min(k))
    times=np.linspace(start,stop,int(np.ceil((stop-start)/step))+1)
    result=solve_transient_state_space(h,1j*ports.T,ports,-np.eye(4),source,(start,stop),times,
        initial_state=np.zeros(2,complex),rtol=rtol,atol=atol,max_step=probe_ps/8)
    # Explicit spectral bin versus pulse-integrated photon numbers.
    inputs=result.inputs;outputs=result.outputs;states=result.states
    flux=integrated_flux(result,np.ones(4),np.ones(4),quantity='photon_flux',
        provenance='Explicit symmetric two-face SI S41 photon-normalized port model')
    upper_flux=integrated_flux(result,np.ones(4),np.array([0,0,1,1]),quantity='photon_flux',
        provenance='Collect the two upper-frequency photon channels in the same S41 model')
    input_total=flux['input_integral'];output_total=flux['output_integral'];converted=upper_flux['output_integral']
    input_dc=simpson(inputs[:,0],x=times);output_dc=simpson(outputs[:,2:],x=times,axis=0)
    spectral=float(np.sum(abs(output_dc)**2)/abs(input_dc)**2)
    residual=abs(output_total+float(np.sum(abs(states[-1])**2))-input_total)/input_total
    record=dict(eta_integrated=converted/input_total,eta_spectral=spectral,
        photon_balance_relative=residual,remaining_state_norm=float(np.sum(abs(states[-1])**2)),
        interval_ps=[start,stop],samples=len(times))
    if return_trace:record.update(time_ps=times,states=states,input=inputs,output=outputs)
    return record


def optimize_delay(decay,tau,observable='eta_spectral'):
    optimum=minimize_scalar(lambda d:-pulse_response(decay,1e-3,tau,d,step=.018)[observable],
        bounds=(-2.,4.),method='bounded',options={'xatol':2e-4})
    if not optimum.success:raise RuntimeError('Low-pump delay optimization failed')
    return float(optimum.x)


def independent_cw_ode(decay,g):
    # Primitive physical ps equations, no Floquet or transient-module assembly.
    k1,k2=decay
    def rhs(t,a):return [-k1*a[0]+1j*g*a[1]+np.sqrt(k1),-k2*a[1]+1j*g*a[0]]
    horizon=35/min(decay)
    sol=solve_ivp(rhs,(0,horizon),np.zeros(2,complex),method='DOP853',rtol=2e-11,atol=2e-13)
    return np.sqrt(k2)*sol.y[1,-1]


def run_cw():
    OUTPUT.mkdir(parents=True,exist_ok=True);p=published_inputs();k=p['decay_per_ps'];drive=p['drive_per_ps']
    power=np.linspace(0,45,301);coop=power*p['C_per_mW'];g=np.sqrt(coop*np.prod(k))
    eta=[];balance=[];analytic_error=[];boundary=[]
    for coupling in g:
        value,total,result=floquet_response(k,coupling,drive)
        eta.append(value);balance.append(abs(total-1));analytic_error.append(abs(value-analytic_efficiency(k,coupling)))
        boundary.append(result.diagnostics['maximum_boundary_relative_residual'])
    author=p.pop('author')
    np.savez_compressed(OUTPUT/'cw_scan.npz',power_mW=power,C=coop,g_GHz=g/(2*np.pi)*1000,
        eta_photon=eta,input_loss_GHz=p['linewidths_GHz'],author_power=author['pump_mW'],
        author_I_normalized=author['sfg_intensity_au']/author['sfg_intensity_au'].max(),
        author_C=author['cooperativity'],author_g_GHz=author['g_over_2pi_GHz'])
    controls=[]
    for C in (0.,.2,1.,2.062060762981566):
        coupling=np.sqrt(C*np.prod(k));value,total,result=floquet_response(k,coupling,drive)
        native=result.column('lower_top')[np.where(result.output_harmonics==1)[0][0],2]
        primitive=independent_cw_ode(k,coupling)
        # Retained harmonic at n=1 is upper output; compare complex amplitude.
        shifted=floquet_response(k,coupling,drive,phase=.43)[2].column('lower_top')
        base=result.column('lower_top');expected=base*np.exp(-1j*result.output_harmonics[:,None]*.43)
        controls.append(dict(C=C,primitive_complex_amplitude_error=float(abs(native-primitive)),
            phase_covariance_error=float(np.max(abs(shifted-expected)))))
    # Detuned spectra: independent rational expression, not only peak shape.
    detunings=np.linspace(-4*max(k),4*max(k),301);coupling=np.sqrt(2*np.prod(k))
    spectrum=np.array([floquet_response(k,coupling,drive,d)[0] for d in detunings])
    analytic=analytic_efficiency(k,coupling,detunings)
    poles=np.linalg.eigvals(np.array([[-1j*k[0],-coupling],[-coupling,-1j*k[1]]]))
    np.savez_compressed(OUTPUT/'detuned_scattering.npz',detuning_per_ps=detunings,eta=spectrum,analytic=analytic,poles=poles)
    summary=dict(status='CALCULATED_PUBLISHED_TWO_MODE_RESPONSE_WITH_DECLARED_MEASURED_INPUTS',parameters=p,
        max_CW_analytic_error=max(analytic_error),max_photon_balance_error=max(balance),
        max_omitted_harmonic_residual=max(boundary),controls=controls,
        max_detuned_analytic_error=float(np.max(abs(spectrum-analytic))),
        intensity_peak_power_mW=float(power[np.argmax(eta)]),g45_GHz=float(g[-1]/(2*np.pi)*1000),
        author_source_g45_GHz=float(author['g_over_2pi_GHz'][-1]),author_caption_g45_GHz=220.,
        all_controls_pass=bool(max(analytic_error)<1e-10 and max(balance)<1e-10 and
                              max(c['primitive_complex_amplitude_error'] for c in controls)<1e-9),
        comparison_scope='Fig4a conditional on author C/P calibration, normalized source intensity. Fig4b independently selected static decays yield g; not a first-principles prediction of C(P).',
        physical_scope='SI two-mode rotating-wave photon model with symmetric two-face radiative closure; real slab is asymmetric',
        not_certified=['Legume tensor/dissipative full Maxwell slab','Chern number','actual modal radiation branching fractions'])
    source_files=['source_data_parsed.json','figure4_author_data.csv','41565_2025_2003_MOESM1_ESM.pdf']
    summary['source_sha256']={f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in source_files}
    save_json(OUTPUT/'cw_summary.json',summary);print(json.dumps(summary,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a),flush=True)


def run_pulses():
    OUTPUT.mkdir(parents=True,exist_ok=True);p=published_inputs();k=p['decay_per_ps'];C=np.linspace(0,4,81);taus=[1.5,6.]
    spectral=[];integrated=[];delays=[];max_balance=0.;controls=[]
    for tau in taus:
        delay=optimize_delay(k,tau);delays.append(delay);eta_s=[];eta_i=[]
        for j,c in enumerate(C):
            r=pulse_response(k,float(c),tau,delay);eta_s.append(r['eta_spectral']);eta_i.append(r['eta_integrated'])
            max_balance=max(max_balance,r['photon_balance_relative'])
            if j%20==0:print('pulse',tau,j+1,len(C),r['eta_spectral'],flush=True)
        spectral.append(eta_s);integrated.append(eta_i)
        for c in (1.,2.,4.):
            base=pulse_response(k,c,tau,delay);fine=pulse_response(k,c,tau,delay,step=.006,rtol=2e-11,atol=2e-13)
            controls.append(dict(tau=tau,C0=c,spectral_change=abs(base['eta_spectral']-fine['eta_spectral']),
                integrated_change=abs(base['eta_integrated']-fine['eta_integrated'])))
    np.savez_compressed(OUTPUT/'pulse_scan.npz',C0=C,tau_ps=taus,eta_spectral=spectral,eta_integrated=integrated,delay_ps=delays)
    trace=pulse_response(k,2.,1.5,delays[0],return_trace=True)
    np.savez_compressed(OUTPUT/'pulse_trace.npz',**{k:v for k,v in trace.items() if isinstance(v,np.ndarray)})
    summary=dict(scope='SI Fig7(a) two-mode pulse model; selected static measured decay pair and low-pump optimized delay',
        decay_GHz=p['linewidths_GHz'],probe_intensity_FWHM_ps=.169,pump_intensity_FWHM_ps=taus,
        delay_ps=delays,delay_method='Maximize on-resonance spectral conversion at C0=.001, then fix delay for entire C0 scan',
        spectral_efficiency='sum_faces |integral s_out_upper(t)dt|^2 / |integral s_in_lower(t)dt|^2 at the resonant spectral bin',
        pulse_photon_efficiency='integral sum_faces |s_out_upper(t)|^2 dt / integral |s_in_lower(t)|^2 dt',
        peaks=[dict(tau=t,C0=float(C[np.argmax(s)]),spectral=float(max(s)),max_integrated=float(max(i))) for t,s,i in zip(taus,spectral,integrated)],
        max_photon_balance_error=max_balance,refinement=controls,
        limits='Published SI does not specify exact scalar decay pair, delay or integration formula for Fig7. The two observables are kept separate; no vertical fit or amplitude rescaling.',
        total_photon_bound='Symmetric two-face single-sided passive photon model converts at most one half of total pulse photons; spectral-bin ratio can exceed one half without violating this bound')
    save_json(OUTPUT/'pulse_summary.json',summary);print(json.dumps(summary,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['cw','pulse'],default='cw');args=parser.parse_args()
    (run_cw if args.stage=='cw' else run_pulses)()
