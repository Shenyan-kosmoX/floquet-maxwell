"""Independent normal-incidence Maxwell boundary reference for open slabs.

This analytic fixture validates poles/residues/scattering of a specified
homogeneous slab, not a replacement for Legume's spatial solver. Units c=1.
The local positive-frequency refractive-index branch is explicitly checked;
no global continuation through a material pole or branch cut is claimed.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class OutgoingSlab:
    permittivity: object
    permittivity_derivative: object
    thickness: float=1.

    def __post_init__(self):
        if not np.isreal(self.thickness) or not np.isfinite(self.thickness) or self.thickness<=0:
            raise ValueError('Positive finite slab thickness required')

    def index(self,z):
        eps=complex(self.permittivity(z))
        if not np.isfinite(eps) or abs(eps)<1e-12:raise ValueError('Material pole or zero not supported in this local branch')
        n=np.sqrt(eps)
        if n.real<1e-8:raise ValueError('Local positive-index branch is unresolved; specify a different analytic fixture')
        return n

    def matrix(self,z):
        n=self.index(z);e2=np.exp(2j*n*z*self.thickness)
        return np.array([[n+1,1-n],[(n-1)*e2,-(n+1)]],complex)

    def derivative(self,z):
        n=self.index(z);dn=complex(self.permittivity_derivative(z))/(2*n)
        e2=np.exp(2j*n*z*self.thickness)
        return np.array([[dn,-dn],[(dn+2j*self.thickness*(n-1)*(n+z*dn))*e2,-dn]],complex)

    def ports(self,z):
        e=np.exp(1j*self.index(z)*z*self.thickness)
        incident=np.diag([2.,-2*e]).astype(complex)
        outgoing=np.array([[1.,1.],[e,1/e]],complex)
        return incident,outgoing,-np.eye(2)

    def scattering(self,z):
        b,c,d=self.ports(z)
        return d+c@np.linalg.solve(self.matrix(z),b)

    def fresnel_scattering(self,z):
        """Independent multiple-reflection formula at the two slab faces."""
        n=self.index(z);r=(1-n)/(1+n);e=np.exp(1j*n*z*self.thickness)
        denominator=1-r*r*e*e
        reflection=r*(1-e*e)/denominator
        transmission=(1-r*r)*e/denominator
        return np.array([[reflection,transmission],[transmission,reflection]])

    def scattering_residue(self,pole):
        """Nonlinear pole is normalized with A', including material dispersion."""
        from .qnm import simple_pole_residue
        result=simple_pole_residue(pole,self.matrix,self.derivative)
        b,c,_=self.ports(pole.frequency)
        return c@result.residue@b,result
