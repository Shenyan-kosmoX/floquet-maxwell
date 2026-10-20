"""Independent periodic-sphere Floquet scattering reference using native treams.

This is a validation backend for open spatially and temporally periodic scalar
materials. It does not replace the Legume patterned-slab foundation. Lengths
and angular frequencies use a caller-declared common c=1 unit system.
"""
from pathlib import Path
import sys
import warnings
import numpy as np
from scipy.linalg import block_diag,solve,LinAlgWarning,matrix_balance

from toolbox_experiments.nphoton_2024_01563.sphere_lattice_smin import basis_modes,lattice_coupling,reality_transform,treams


def sphere_matrix(material,omega,drive,radius,n_harmonics,lmax):
    modes=basis_modes(lmax);ell,em,pol=modes.T;nsp=len(modes);nh=2*n_harmonics+1
    matrix=np.zeros((nsp*nh,nsp*nh),complex);diagnostics=[]
    for order in range(1,lmax+1):
        t=material.tmatrix(omega,drive,radius=radius,n_harmonics=n_harmonics,ell=order)
        diagnostics.append(t.diagnostics)
        for j in np.flatnonzero(ell==order):
            ids=np.arange(nh)*nsp+j;matrix[np.ix_(ids,ids)]=t.blocks[int(pol[j])]
    return matrix,omega+drive*np.arange(-n_harmonics,n_harmonics+1),diagnostics


def incident_spherical(omega,kparallel,lmax,polarization=1,direction=1):
    if omega<=0:raise ValueError('This reference injects a positive-frequency plane wave')
    if polarization not in (0,1) or direction not in (-1,1):raise ValueError('Invalid native parity/propagation direction')
    kparallel=np.asarray(kparallel,float);kz2=omega**2-kparallel@kparallel
    if kz2<=1e-12*max(1.,omega**2):raise ValueError('Incident channel must be propagating and away from threshold')
    l,m,p=basis_modes(lmax).T
    return np.asarray(treams.pw.to_sw(l,m,p,kparallel[0],kparallel[1],direction*np.sqrt(kz2),np.int32(polarization),poltype='parity'))


def outgoing_plane_map(frequency,kparallel,lmax,area,direction):
    """Return both native parity amplitudes in a declared signed-frequency chart.

    Negative frequencies use the exact electric-VSW reality transform and the
    conjugate positive-frequency plane-wave basis at -kparallel. This preserves
    outgoing energy direction. Raw negative k is never fed to the native Ewald
    continuation. Amplitudes of opposite-frequency collisions require coherent
    physical-polarization reconstruction before computing real-field powers.
    """
    if frequency==0:raise ValueError('Zero-frequency outgoing basis not implemented')
    kparallel=np.asarray(kparallel,float);w=abs(float(frequency));kz2=w*w-kparallel@kparallel
    if kz2<=1e-12*max(1.,w*w):raise ValueError('Output must be propagating and away from threshold')
    kp=kparallel if frequency>0 else -kparallel
    l,m,p=basis_modes(lmax).T
    positive=np.asarray(treams.sw.periodic_to_pw(kp[0],kp[1],direction*np.sqrt(kz2),
        np.array([0,1],dtype='l')[:,None],l,m,p,area,poltype='parity'))
    return positive if frequency>0 else positive.conj()@reality_transform(lmax)


def open_orders(frequency,kparallel,period):
    w=abs(float(frequency));reciprocal=2*np.pi/period
    bound=int(np.ceil((w+np.linalg.norm(kparallel))/reciprocal))+1
    orders=[]
    for gx in range(-bound,bound+1):
        for gy in range(-bound,bound+1):
            transverse=np.asarray(kparallel)+reciprocal*np.array([gx,gy]);margin=w*w-transverse@transverse
            if abs(margin)<=1e-12*max(1.,w*w):raise ValueError('Outgoing diffraction threshold needs a separate limiting treatment')
            if margin>0:orders.append((gx,gy,transverse,np.sqrt(margin)/w))
    return orders


def scatter_sphere_array(material,omega,drive,*,radius,period,n_harmonics=5,lmax=3,
                         kparallel=(0.,0.),polarization=1,eta=0.,allow_ill_conditioned=False):
    """All propagating outgoing orders for one positive-frequency input.

    The central harmonic carries the incident wave. Totals sum real-field mean
    power only when distinct signed output frequencies have distinct absolute
    values. Half-zone collisions and zero sidebands are explicitly rejected;
    no eigenfrequency or stability claim follows from this forced response.
    """
    if not np.isreal(omega) or not np.isreal(drive) or not np.isfinite(omega) or not np.isfinite(drive) or omega<=0 or drive<=0:
        raise ValueError('Positive real incident and drive frequencies required')
    for value,name in [(n_harmonics,'n_harmonics'),(lmax,'lmax')]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<(0 if name=='n_harmonics' else 1):
            raise ValueError(f'{name} must be a valid integer')
    if not np.isreal(radius) or not np.isreal(period) or not np.isfinite(radius) or not np.isfinite(period) or radius<=0 or period<=2*radius:
        raise ValueError('Disjoint positive-radius spheres and positive period required')
    kp=np.asarray(kparallel,float)
    if kp.shape!=(2,) or not np.isfinite(kp).all():raise ValueError('kparallel requires two finite angular components')
    with warnings.catch_warnings(record=True) as sphere_warnings:
        warnings.simplefilter('always',LinAlgWarning)
        t,frequencies,sphere_checks=sphere_matrix(material,float(omega),float(drive),radius,n_harmonics,lmax)
    numerical_warnings=['sphere boundary: '+str(w.message) for w in sphere_warnings]
    if numerical_warnings and not allow_ill_conditioned:
        raise np.linalg.LinAlgError('Periodic-sphere solve was ill-conditioned: '+numerical_warnings[0])
    if np.any(abs(frequencies)<1e-10*max(1.,drive)):raise ValueError('Zero-frequency sideband is unsupported')
    if np.any(abs(frequencies[:,None]+frequencies[None,:])<1e-10*max(1.,drive)):
        raise ValueError('Opposite-frequency collision: coherent real-field power requires an explicit polarization/input phase contract')
    nsp=len(basis_modes(lmax));lattice=np.eye(2)*period
    g=block_diag(*[lattice_coupling(f,kp,lmax=lmax,eta=eta,lattice=lattice) for f in frequencies])
    source=np.zeros(len(t),complex)
    source[n_harmonics*nsp:(n_harmonics+1)*nsp]=incident_spherical(omega,kp,lmax,polarization)
    a=np.eye(len(t))-t@g;rhs=t@source
    # Multipole coefficients at very different signed frequencies have wildly
    # different natural magnitudes. LAPACK similarity equilibration changes
    # coordinates only: A_b=S^-1 A S, b=S y. Export the original coefficients.
    balanced,(scales,_)=matrix_balance(a,permute=False,separate=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',LinAlgWarning)
        scattered=scales*solve(balanced,rhs/scales,assume_a='gen')
    numerical_warnings.extend('array: '+str(w.message) for w in caught)
    if numerical_warnings and not allow_ill_conditioned:
        raise np.linalg.LinAlgError('Periodic-sphere solve was ill-conditioned: '+numerical_warnings[0])
    residual=float(np.linalg.norm(a@scattered-rhs)/max(np.linalg.norm(a)*np.linalg.norm(scattered)+np.linalg.norm(rhs),1e-300))
    componentwise_residual=float(np.max(abs(a@scattered-rhs)/np.maximum(abs(a)@abs(scattered)+abs(rhs),1e-300)))
    input_cosine=np.sqrt(omega**2-kp@kp)/omega
    channels=[];r=0.;transmission=0.;rpositive=0.;tpositive=0.;byharm=[]
    coefficients=scattered.reshape(len(frequencies),nsp)
    for j,f in enumerate(frequencies):
        hr=0.;ht=0.
        for gx,gy,kout,cosine in open_orders(f,kp,period):
            for direction in (-1,1):
                amp=outgoing_plane_map(f,kout,lmax,period**2,direction)@coefficients[j]
                if j==n_harmonics and gx==gy==0 and direction==1:amp[polarization]+=1.
                power=cosine/input_cosine*abs(amp)**2
                if direction==-1:r+=float(power.sum());hr+=float(power.sum())
                else:transmission+=float(power.sum());ht+=float(power.sum())
                if f>0:
                    if direction==-1:rpositive+=float(power.sum())
                    else:tpositive+=float(power.sum())
                channels.append(dict(harmonic=j-n_harmonics,frequency=float(f),G=[gx,gy],direction=direction,
                    amplitude_real=amp.real.tolist(),amplitude_imag=amp.imag.tolist(),relative_power=power.tolist()))
        byharm.append([j-n_harmonics,f,hr,ht])
    edge=float((np.linalg.norm(coefficients[0])**2+np.linalg.norm(coefficients[-1])**2)/max(np.linalg.norm(coefficients)**2,1e-300))
    return dict(reflectivity=r,transmissivity=transmission,net_exchange=1-r-transmission,
        positive_frequency_only_R=rpositive,positive_frequency_only_T=tpositive,
        negative_frequency_output_power=r+transmission-rpositive-tpositive,
        by_harmonic=np.array(byharm),channels=channels,scattered_coefficients=scattered,
        signed_frequencies=frequencies,basis_modes=basis_modes(lmax),linear_residual=residual,
        componentwise_linear_residual=componentwise_residual,
        similarity_scale_range=[float(scales.min()),float(scales.max())],
        numerical_warnings=numerical_warnings,numerically_qualified=not bool(numerical_warnings),
        spherical_edge_coefficient_fraction=edge,sphere_checks=sphere_checks,
        scope='Specified finite-harmonic/multipole periodic-sphere forced scattering; net_exchange includes pump work and is not pure material absorption',
        negative_frequency_power='Each distinct absolute frequency counted once in real-field mean; opposite-frequency coincidences rejected',
        discarded_spatial_output_orders=0,all_propagating_orders_in_retained_frequency_window=True,
        causal_stability_verified=False)
