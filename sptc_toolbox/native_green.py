"""Static native port response plus modulation in one fixed coordinate chart.

This adapter improves composition and exact finite-model reduction. Its static
response still has the stated GME/frozen-radiation/homogeneous-background
approximation of NativeSlabPortModel; no new continuum or QNM input is invented.
"""
from copy import deepcopy
from dataclasses import replace
import numpy as np

from .slab_ports import NativeSlabPortModel, homogeneous_slab_background
from .modal_adapter import _sample_blocks
from .green_scattering import (StaticGreenResponse, StaticGreenSystem,
                                ModulationKernel, solve_green_floquet)


class NativeGreenModel:
    """Use the pump-off native response at every signed sideband frequency.

    ``reference_ports`` must be a static model in exactly the same modal,
    radiation and physical port coordinates as ``driven_ports``. A DC shift
    and nonlinear-in-permittivity mean shift belong to the modulation kernel,
    not silently to a redefined pump-off reference.
    """
    def __init__(self,reference_ports,driven_ports,*,max_modulation_order=6,n_samples=128):
        if not isinstance(reference_ports,NativeSlabPortModel) or not isinstance(driven_ports,NativeSlabPortModel):
            raise TypeError('Both references must be NativeSlabPortModel instances')
        a,b=reference_ports,driven_ports
        for name in ('drive','polarization','background_epsilon','thickness','reference_distance'):
            if getattr(a,name)!=getattr(b,name):raise ValueError('Static/driven physical port contract differs: '+name)
        if a.metadata['basis_id']!=b.metadata['basis_id']:
            raise ValueError('Static and driven responses need the same ordered native basis')
        if not np.array_equal(a.internal.bath.couplings,b.internal.bath.couplings):
            raise ValueError('This adapter modulates internal material only; static and driven radiation must agree')
        static_blocks,static_check=_sample_blocks(a.internal.matrix,0,n_samples)
        h0=np.array(static_blocks[0],copy=True)
        delta,delta_check=_sample_blocks(lambda t:b.internal.matrix(t)-h0,
                                       max_modulation_order,n_samples)
        # Snapshot matrices and port parameters; later mutations of the caller
        # cannot silently change a previously built Green response.
        self.drive=float(a.drive);self.n_modes=a.internal.n_modes
        coupling=np.array(a.internal.bath.couplings,copy=True)
        epsilon,thickness,gap=float(a.background_epsilon),float(a.thickness),float(a.reference_distance)
        n=self.n_modes
        reciprocal=np.asarray(a.sample.metadata.get('spatial_reciprocal_vectors_Ga',np.empty((2,0))))
        lengths=np.linalg.norm(reciprocal,axis=0) if reciprocal.ndim==2 and reciprocal.shape[0]==2 else np.array([])
        positive=lengths[lengths>1e-10]
        self.first_diffraction_threshold=float(positive.min()) if len(positive) else None
        output=np.hstack((np.zeros((2,n)),np.sqrt(2)*coupling.conj().T))
        metadata={**deepcopy(a.metadata),
            'basis':a.metadata['basis_id'],
            'source':'Native GME fields and common-frequency radiation, declared homogeneous direct background',
            'reference_planes':f'Outside slab by gap={gap:g} on both sides',
            'normalization':'Declared native two-port flux amplitudes; inherited physical accuracy limits',
            'frequency_units':'omega_over_drive',
            'drive_omega_a_over_c':self.drive,
            'static_reference':'Pump-off model; nonzero modulation mean belongs to V_0',
            'static_fourier_check':static_check,'modulation_fourier_check':delta_check,
            'radiation':'Same frozen common-frequency bath for static and driven systems',
            'not_provided':['Full patterned-scattering background','Retarded native continuum completion',
                            'Pump modulation of direct continuum response','General non-Gamma ports']}
        def sample(frequency):
            if not np.isfinite(frequency) or abs(complex(frequency).imag)>0 or frequency==0:
                raise ValueError('Native background adapter evaluates nonzero signed real frequencies; DC needs a separate port contract')
            frequency=float(np.real(frequency));omega=frequency*self.drive
            background=homogeneous_slab_background(omega,epsilon,thickness)
            phase=np.exp(1j*omega*gap)
            source=np.vstack((np.zeros((n,2)),-2j*np.sqrt(2)*coupling@background))*phase
            return StaticGreenResponse(frequency*np.eye(2*n)-h0,source,
                output*phase,background*phase**2)
        self.system=StaticGreenSystem(sample,
            tuple([f'q{j}' for j in range(n)]+[f'v{j}' for j in range(n)]),
            ('bottom','top'),('bottom','top'),metadata)
        self.modulation=ModulationKernel(tuple(sorted(delta)),
            lambda wr,wc,p:delta[p],
            {'basis':metadata['basis'],'frequency_units':'omega_over_drive',
             'origin':'Driven native Maxwell generator minus the explicitly static generator',
             'sign':'A_F=diag(z+n-H_static)-delta_H_F',
             'fourier_check':delta_check})
        self.metadata=metadata

    def solve(self,incident_omega,*,n_harmonics=3,incident_harmonics=(0,),
              retained_modes=None,**kwargs):
        """Solve at physical omega*a/c; result spectral coordinates use omega/drive.

        ``retained_modes`` selects physical native modal coordinates; q and v
        are retained together. Every eliminated coordinate is reconstructed.
        Physical frequencies are also supplied in the returned metadata.
        """
        if not np.isscalar(incident_omega) or not np.isreal(incident_omega) or not np.isfinite(incident_omega) or incident_omega==0:
            raise ValueError('Supply a nonzero finite real incident angular omega*a/c')
        retained=None
        if retained_modes is not None:
            modes=list(retained_modes)
            if (not modes or any(isinstance(j,(bool,np.bool_)) or not isinstance(j,(int,np.integer))
                    or j<0 or j>=self.n_modes for j in modes) or len(set(modes))!=len(modes)):
                raise ValueError('retained_modes must be unique valid modal indices')
            retained=modes+[self.n_modes+j for j in modes]
        response=solve_green_floquet(self.system,self.modulation,float(incident_omega)/self.drive,
            drive_frequency=1.,n_harmonics=n_harmonics,incident_harmonics=incident_harmonics,
            retained_indices=retained,**kwargs)
        return replace(response,metadata={**response.metadata,
            'physical_output_omega_a_over_c':((float(incident_omega)/self.drive+response.output_harmonics)*self.drive).tolist(),
            'physical_input_omega_a_over_c':((float(incident_omega)/self.drive+response.incident_harmonics)*self.drive).tolist(),
            'first_omitted_diffraction_threshold_omega_a_over_c':self.first_diffraction_threshold,
            'outside_two_port_harmonics':None if self.first_diffraction_threshold is None else
                response.output_harmonics[abs(float(incident_omega)+response.output_harmonics*self.drive)>=self.first_diffraction_threshold].tolist(),
            'native_scope':'Exact composition/reduction of declared finite native port model; not full physical Green function'})
