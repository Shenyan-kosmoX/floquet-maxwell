"""Independent direct outgoing Fourier-modal eigenfrequency diagnostic.

Single patterned layer, isotropic nonmagnetic dielectric, air claddings.
This solves boundary-matching zeros in complex omega, NOT spectral peaks.
The xy Fourier truncation is approximate; z propagation and air DtN are exact
for that truncated system. No GME guided-mode reference is used.
Units: a=c0=1; exp(i*k.r-i*omega*t). The slab is centered at z=0.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eig, solve, eigh
from scipy.special import j1
from .static_modes import solve_nonlinear_mode


def reciprocal_grid(cutoff):
    if cutoff < 0 or int(cutoff) != cutoff:
        raise ValueError("Require nonnegative integer square Fourier cutoff")
    return 2*np.pi*np.array([(x,y) for x in range(-cutoff,cutoff+1)
                            for y in range(-cutoff,cutoff+1)],float).T


def cylinder_epsilon_matrix(gvec, epsilon=16., radius=1/3.2):
    delta = gvec[:,:,None]-gvec[:,None,:]
    arg = radius*np.linalg.norm(delta,axis=0)
    mask = np.full(arg.shape,np.pi*radius**2)
    nonzero = arg > 1e-14
    mask[nonzero] *= 2*j1(arg[nonzero])/arg[nonzero]
    return np.eye(gvec.shape[1])+(epsilon-1)*mask


def air_outgoing_q(omega, transverse_squared):
    """Retarded sheet for positive Re omega, continued through the real axis.

    q=sqrt(omega-K)*sqrt(omega+K) with principal individual square roots.
    A tiny negative Im omega would flip evanescent waves to the other sheet;
    fix this by defining each channel's propagation/evanescence sheet from
    Re omega. The resulting function is locally analytic away from Rayleigh
    thresholds (roots must remain in one threshold interval).
    """
    w = complex(omega)
    if w.real <= 0:
        raise ValueError("This diagnostic is restricted to positive real frequencies")
    q = np.sqrt((w*w-np.asarray(transverse_squared))+0j)
    propagating = w.real*w.real > transverse_squared
    q = np.where(propagating & (q.real<0),-q,q)
    q = np.where(~propagating & (q.imag<0),-q,q)
    if np.min(abs(q)) < 1e-10:
        raise ValueError("Rayleigh threshold requires a separate branch treatment")
    return q


def air_admittance(omega,kx,ky):
    q = air_outgoing_q(omega,kx*kx+ky*ky)
    return np.block([[np.diag(-kx*ky/(omega*q)),np.diag(-(omega*omega-kx*kx)/(omega*q))],
                     [np.diag((omega*omega-ky*ky)/(omega*q)),np.diag(kx*ky/(omega*q))]])


def gamma_a2_basis(gvec, axial=False):
    """C4=+1, sigma_x=-1 transverse vectors on a C4-complete Fourier grid."""
    labels = np.rint(gvec.T/(2*np.pi)).astype(int)
    lookup = {tuple(v):j for j,v in enumerate(labels)}
    n = len(labels)
    operators = []
    for rotation in [np.array([[0.,-1.],[1.,0.]]),np.diag([-1.,1.])]:
        source = labels@rotation
        permutation = [lookup[tuple(np.rint(v).astype(int))] for v in source]
        components = np.linalg.det(rotation)*rotation if axial else rotation
        operator = np.zeros((2*n,2*n),complex)
        for a in range(2):
            for b in range(2):
                operator[a*n+np.arange(n),b*n+np.asarray(permutation)] = components[a,b]
        operators.append(operator)
    r,m = operators
    ident = np.eye(2*n)
    projector = (ident+r+r@r+r@r@r)@(ident-m)/8
    projector = (projector+projector.conj().T)/2
    values,vectors = eigh(projector)
    return vectors[:,values>.5]


@dataclass
class OutgoingFMM:
    gvec: np.ndarray
    epsilon_matrix: np.ndarray
    height: float

    def __post_init__(self):
        self.inverse_epsilon = np.linalg.inv(self.epsilon_matrix)

    @classmethod
    def cylinder(cls,cutoff,epsilon=16.,radius=1/3.2,height=1/(.703*3.2)):
        gvec = reciprocal_grid(cutoff)
        return cls(gvec,cylinder_epsilon_matrix(gvec,epsilon,radius),height)

    @classmethod
    def uniform(cls,index=2.,height=.7):
        return cls(np.zeros((2,1)),np.array([[index**2]],complex),height)

    def maxwell_blocks(self,omega,kpoint):
        kx,ky = self.gvec+np.asarray(kpoint)[:,None]
        x,y = np.diag(kx),np.diag(ky)
        ident = np.eye(len(kx))
        eta,eps,w = self.inverse_epsilon,self.epsilon_matrix,complex(omega)
        a = np.block([[x@eta@y/w,w*ident-x@eta@x/w],
                      [-w*ident+y@eta@y/w,-y@eta@x/w]])
        b = np.block([[-x@y/w,x@x/w-w*eps],
                      [w*eps-y@y/w,y@x/w]])
        return a,b

    def boundary_matrix(self,omega,kpoint=(0.,0.),parity="even",a2_gamma=False):
        """Analytic matrix sine/cosine matching Et at center to outgoing air.

        even means tangential E even under z reflection (A2g has this parity).
        This representation is independent of internal eigenvector ordering,
        phases and q sign, since cos(sqrt(lambda)L) and sin(sqrt(lambda)L)/
        sqrt(lambda) are entire functions of lambda. Large evanescent orders
        may still cause transfer conditioning; monitor cutoff convergence.
        """
        a,b = self.maxwell_blocks(omega,kpoint)
        values,vectors = eig(a@b)
        q = np.sqrt(values+0j)
        length = self.height/2
        cos = np.cos(q*length)
        sin_over_q = length*np.sinc(q*length/np.pi)
        cmat = solve(vectors.T,(vectors*cos).T).T
        smat = solve(vectors.T,(vectors*sin_over_q).T).T
        kx,ky = self.gvec+np.asarray(kpoint)[:,None]
        admittance = air_admittance(omega,kx,ky)
        if parity == "even":
            result = 1j*b@smat-admittance@cmat
        elif parity == "odd":
            # Use an independent initial H amplitude at z=0 for odd E.
            # Propagation from H0: E=i*A*sinc(sqrt(BA)L)*L, H=cos(sqrt(BA)L).
            vals,hvec = eig(b@a)
            hq = np.sqrt(vals+0j)
            hc = solve(hvec.T,(hvec*np.cos(hq*length)).T).T
            hs = solve(hvec.T,(hvec*(length*np.sinc(hq*length/np.pi))).T).T
            result = hc-1j*admittance@a@hs
        else:
            raise ValueError("parity must be even or odd")
        if a2_gamma:
            if np.linalg.norm(kpoint)>1e-14 or parity!="even":
                raise ValueError("A2g reduction applies only to Gamma and even Et")
            self._last_e_basis = gamma_a2_basis(self.gvec,axial=False)
            left = gamma_a2_basis(self.gvec,axial=True)
            result = left.conj().T@result@self._last_e_basis
        return result

    def find_mode(self,seed,kpoint=(0.,0.),initial_vector=None,parity="even",
                  a2_gamma=False,tolerance=2e-9,search_radius=.5):
        def operator(w):
            return self.boundary_matrix(w,kpoint,parity,a2_gamma)
        return solve_nonlinear_mode(operator,seed,initial_vector=initial_vector,
                                    tolerance=tolerance,search_radius=search_radius,
                                    max_iterations=40,residual_scale=1.)

    def even_fields(self,omega,kpoint,e_center,z_samples):
        """Reconstruct E,H in the patterned layer from an even-E pole vector.

        Fields at complex poles have arbitrary normalization. Finite-layer
        overlap is an identity diagnostic, not a dispersive QNM norm.
        """
        a,b = self.maxwell_blocks(omega,kpoint)
        values,vectors = eig(a@b)
        q = np.sqrt(values+0j)
        amplitudes = solve(vectors,e_center)
        n = self.gvec.shape[1]
        kx,ky = self.gvec+np.asarray(kpoint)[:,None]
        electric,magnetic = [],[]
        for z in z_samples:
            et = vectors@(np.cos(q*z)*amplitudes)
            ht = 1j*b@vectors@(z*np.sinc(q*z/np.pi)*amplitudes)
            ez = self.inverse_epsilon@(ky*ht[:n]-kx*ht[n:])/omega
            hz = (kx*et[n:]-ky*et[:n])/omega
            electric.append(np.vstack([et[:n],et[n:],ez]))
            magnetic.append(np.vstack([ht[:n],ht[n:],hz]))
        return np.asarray(electric),np.asarray(magnetic)


def embed_center_field(vector,source_gvec,target_gvec):
    """Zero-pad a tangential Fourier field when increasing square cutoff."""
    lookup = {tuple(np.rint(g/(2*np.pi)).astype(int)):i for i,g in enumerate(target_gvec.T)}
    target = np.zeros(2*target_gvec.shape[1],complex)
    for i,g in enumerate(source_gvec.T):
        key=tuple(np.rint(g/(2*np.pi)).astype(int))
        if key in lookup:
            for c in range(2):
                target[c*target_gvec.shape[1]+lookup[key]]=vector[c*source_gvec.shape[1]+i]
    return target
