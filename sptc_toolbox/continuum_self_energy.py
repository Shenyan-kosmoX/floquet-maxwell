"""Unbroadened finite-window matrix continuum with explicit logarithmic cuts.

Gamma(nu) is a DECLARED PSD spectral density, linear on supplied intervals.
Sigma(z)=integral Gamma(nu)/(pi*(z-nu)) dnu. On the upper bank,
-Im_H Sigma=Gamma. For second-order coordinates D(z)=2i Sigma(z).
This removes quadrature-pole/broadening artifacts, not missing Maxwell tails,
reference-slab approximations, QNM normalization, or unknown reactive terms.
"""
import numpy as np


class PiecewiseLinearContinuum:
    def __init__(self, intervals, endpoint_damping, *, metadata=None):
        raw=np.asarray(intervals)
        if np.iscomplexobj(raw):raise ValueError('Intervals must be real')
        edges=np.array(raw,float,copy=True)
        values=np.array(endpoint_damping,complex,copy=True)
        if (edges.ndim!=2 or edges.shape[1]!=2 or not len(edges) or not np.isfinite(edges).all()
                or np.any(edges[:,1]<=edges[:,0]) or np.any(edges[1:,0]<edges[:-1,1])):
            raise ValueError('Intervals must be finite ordered, positive-width and nonoverlapping')
        if (values.ndim!=4 or values.shape[:2]!=edges.shape or values.shape[2]!=values.shape[3]
                or values.shape[2]==0 or not np.isfinite(values).all()):
            raise ValueError('Supply two finite modal damping matrices per interval')
        for a in values.reshape(-1,*values.shape[-2:]):
            scale=max(np.linalg.norm(a),1e-300)
            if np.linalg.norm(a-a.conj().T)>1e-10*scale or np.linalg.eigvalsh((a+a.conj().T)/2).min() < -1e-10*scale:
                raise ValueError('Every spectral endpoint must be Hermitian positive semidefinite')
        self.intervals=edges;self.values=(values+values.conj().transpose(0,1,3,2))/2
        self.slopes=np.diff(self.values,axis=1)[:,0]/np.diff(edges,axis=1)[:,0,None,None]
        self.n_modes=values.shape[2]
        self.metadata=dict(metadata or {})
        self.metadata.update(scope='DECLARED_FINITE_WINDOW_CONTINUUM_NOT_FULL_MAXWELL',
            outside_window='Zero by declared truncation, not inferred physical absence',
            broadening=0,interpolation='Piecewise linear PSD damping density',
            outgoing_sheet='Explicit interval crossed from upper bank; no automatic C(z)C(z)^H')
        for a in (self.intervals,self.values,self.slopes):a.setflags(write=False)

    @classmethod
    def from_nodes(cls, frequencies, damping, **kwargs):
        f=np.asarray(frequencies);d=np.asarray(damping)
        if f.ndim!=1 or len(f)<2 or d.ndim<1 or d.shape[0]!=len(f):
            raise ValueError('At least two frequency nodes and their matrices are required')
        return cls(np.stack((f[:-1],f[1:]),axis=1),np.stack((d[:-1],d[1:]),axis=1),**kwargs)

    def _location(self,z):
        z=complex(z)
        if not np.isfinite(z):raise ValueError('Frequency must be finite')
        if np.any(abs(z-self.intervals)<1e-13*np.maximum(1,abs(self.intervals))):
            raise ValueError('At a spectral knot/threshold; choose a bank limit or refine with an explicit threshold model')
        return z

    def density(self,frequency):
        if np.iscomplexobj(frequency) or not np.isfinite(frequency):raise ValueError('Density is real-axis data only')
        x=float(frequency);ids=np.flatnonzero((self.intervals[:,0]<x)&(x<self.intervals[:,1]))
        if np.any(x==self.intervals):raise ValueError('Density at a knot requires a one-sided convention')
        if not len(ids):return np.zeros((self.n_modes,self.n_modes),complex)
        i=ids[0]
        return self.values[i,0]+self.slopes[i]*(x-self.intervals[i,0])

    def crossing_interval(self,frequency):
        """Declare a cut crossing at a real seed; keep this index fixed in Newton."""
        x=float(frequency)
        ids=np.flatnonzero((self.intervals[:,0]<x)&(x<self.intervals[:,1]))
        if len(ids)!=1:raise ValueError('Crossing must lie strictly inside one supplied interval')
        return int(ids[0])

    def self_energy(self,frequency, *, sheet='retarded', crossed_interval=None, derivative=False):
        """Analytic integral and its derivative; exp(-i z t) convention.

        retarded: upper half plane / upper real bank. physical: either half
        plane of the Cauchy transform. outgoing: continue from the upper bank
        through exactly the declared interval into Im(z)<0. At a complex pole
        the sheet must come from the path, not by conjugating real-axis data.
        """
        z=self._location(frequency)
        if sheet not in ('retarded','physical','outgoing'):raise ValueError('Unknown analytic sheet')
        if sheet=='retarded' and z.imag<0:raise ValueError('Lower-half-plane roots need an explicit outgoing sheet')
        if sheet!='outgoing' and crossed_interval is not None:raise ValueError('Crossing is only valid on the outgoing sheet')
        if sheet=='outgoing':
            if isinstance(crossed_interval,bool) or not isinstance(crossed_interval,(int,np.integer)) or not 0<=crossed_interval<len(self.intervals):
                raise ValueError('Outgoing continuation requires an explicit interval index')
            if z.imag>0:raise ValueError('Outgoing sheet here is defined in the lower half plane')
        a,b=self.intervals.T
        # Retarded/physical real-axis calls use the upper bank. The outgoing
        # real-axis value must instead be the limit from BELOW on its declared
        # sheet, even if Re(z) is no longer inside the crossed interval.
        if sheet=='outgoing' and z.imag==0:
            def lower_log(v):
                return np.log(abs(v))-1j*np.pi*(v<0)
            logs=lower_log(z.real-a)-lower_log(z.real-b)
        else:
            # Separate logs retain the cut and avoid signed-zero ratio issues.
            logs=np.log(z-a+0j)-np.log(z-b+0j)
        line=self.values[:,0]+self.slopes*(z-a)[:,None,None]
        if derivative:
            integral=np.sum(self.slopes*logs[:,None,None]+line*(1/(z-a)-1/(z-b))[:,None,None],axis=0)
        else:
            integral=np.sum(line*logs[:,None,None]-self.slopes*(b-a)[:,None,None],axis=0)
        # The chosen outgoing sheet is continued through a FIXED interval;
        # its discontinuity is not re-selected by Re(z) during a pole search.
        if sheet=='outgoing':
            i=crossed_interval
            integral-=2j*np.pi*(self.slopes[i] if derivative else line[i])
        return integral/np.pi

    def friction_kernel(self,frequency,**kwargs):
        return 2j*self.self_energy(frequency,**kwargs)

    def kernel_derivative(self,frequency,**kwargs):
        return 2j*self.self_energy(frequency,derivative=True,**kwargs)


def floquet_self_energy(continuum,z,drive,harmonics,*,coupling_harmonics=None, sheet_for_sideband=None):
    """First-order Schur feedback including a time-modulated bath coupling.

    B(t)=sum B_p exp(-ip Omega t); Sigma_ll'=sum_r B_(l-r)
    Sigma(z+r Omega) B_(l'-r)^H. Intermediate bath r includes ALL harmonics
    reached by B_p, even outside the retained internal window. A static B
    produces diagonal blocks. No intrinsic modal loss is added here.
    """
    hs=np.asarray(harmonics)
    if hs.ndim!=1 or not len(hs) or not np.issubdtype(hs.dtype,np.integer) or len(set(hs))!=len(hs):
        raise ValueError('Unique integer harmonics required')
    if not np.isreal(drive) or not np.isfinite(drive) or drive<=0:raise ValueError('Positive real drive required')
    n=continuum.n_modes
    blocks={0:np.eye(n)} if coupling_harmonics is None else coupling_harmonics
    if not blocks:raise ValueError('Provide nonempty coupling harmonics')
    for p,v in blocks.items():
        if isinstance(p,bool) or not isinstance(p,(int,np.integer)) or np.shape(v)!=(n,n) or not np.isfinite(v).all():
            raise ValueError('Coupling Fourier blocks require integer keys and finite modal matrices')
    out=np.zeros((n*len(hs),n*len(hs)),complex)
    reached=sorted({int(l-p) for l in hs for p in blocks})
    for r in reached:
        options={} if sheet_for_sideband is None else sheet_for_sideband(r)
        sigma=continuum.self_energy(z+r*drive,**options)
        for i,l in enumerate(hs):
            if l-r not in blocks:continue
            for j,m in enumerate(hs):
                if m-r in blocks:
                    out[i*n:(i+1)*n,j*n:(j+1)*n]+=blocks[l-r]@sigma@np.asarray(blocks[m-r]).conj().T
    return out


class ContinuumFloquetPencil:
    """Second-order K(t) with stationary specified D(z); no frozen diagonal Q.

    Units must be shared (angular frequency). K Fourier coefficients use
    exp(-ip Omega t). This is an explicitly supplied coordinate model, not
    an automatic mapping from arbitrary electric coefficients or QNM fields.
    """
    def __init__(self,continuum,stiffness_blocks,drive,half_harmonics,*,sheet_for_sideband=None):
        if isinstance(half_harmonics,bool) or not isinstance(half_harmonics,int) or half_harmonics<0:raise ValueError('Invalid harmonic cutoff')
        if not np.isreal(drive) or not np.isfinite(drive) or drive<=0:raise ValueError('Positive drive required')
        self.bath=continuum;self.blocks={p:np.array(v,complex,copy=True) for p,v in stiffness_blocks.items()}
        self.n_modes=continuum.n_modes;self.drive=drive
        self.harmonics=np.arange(-half_harmonics,half_harmonics+1)
        self.dimension=len(self.harmonics)*self.n_modes
        self.sheets=sheet_for_sideband
        if 0 not in self.blocks:raise ValueError('Static stiffness block is required')
        for p,v in self.blocks.items():
            if isinstance(p,bool) or not isinstance(p,(int,np.integer)) or v.shape!=(self.n_modes,self.n_modes) or not np.isfinite(v).all():raise ValueError('Invalid stiffness Fourier block')

    def _matrix(self,z,derivative):
        n=self.n_modes;out=np.zeros((self.dimension,self.dimension),complex)
        for i,l in enumerate(self.harmonics):
            if not derivative:
                for j,m in enumerate(self.harmonics):
                    if l-m in self.blocks:out[i*n:(i+1)*n,j*n:(j+1)*n]=self.blocks[l-m]
            w=z+l*self.drive;kw={} if self.sheets is None else self.sheets(int(l))
            d=self.bath.friction_kernel(w,**kw)
            if derivative:diagonal=-2*w*np.eye(n)-1j*d-1j*w*self.bath.kernel_derivative(w,**kw)
            else:diagonal=-w*w*np.eye(n)-1j*w*d
            out[i*n:(i+1)*n,i*n:(i+1)*n]+=diagonal
        return out

    def matrix(self,z):return self._matrix(z,False)
    def derivative(self,z):return self._matrix(z,True)
