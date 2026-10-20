"""Publication-sized figures; every title states the model being displayed."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,
                         "axes.labelsize":9,"axes.titlesize":9,
                         "xtick.labelsize":8,"ytick.labelsize":8,
                         "legend.fontsize":7,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})


def _save(fig, path):
    path = Path(path)
    fig.savefig(path.with_suffix(".png"),dpi=220,bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)


def plot_homogeneous_bands(data, gap_edges, config, output):
    style()
    k,w,ref = data["kappa"],data["quasifrequencies"],data["reference"]
    # Display in [0,1) so both branches near the temporal zone edge are visible.
    real = np.mod(w.real,1)
    real_ref = np.mod(ref.real,1)
    fig,axes = plt.subplots(2,2,figsize=(7.0,5.0),layout="constrained")
    for j in range(w.shape[1]):
        axes[0,0].scatter(k,real[:,j],s=7,color="#0072B2",zorder=3,
                          label="Fourier eigenproblem" if j==0 else None)
        axes[0,0].plot(k,real_ref[:,j],color="#E69F00",lw=1,
                       label="Independent monodromy" if j==0 else None)
        axes[0,1].scatter(k,w[:,j].imag,s=7,color="#0072B2",zorder=3)
        axes[0,1].plot(k,ref[:,j].imag,color="#E69F00",lw=1)
    for ax in axes[0]:
        if gap_edges is not None:
            ax.axvspan(*gap_edges,color="#009E73",alpha=.1)
        ax.set_xlabel(r"$\kappa=c_0 k/\Omega$")
    axes[0,0].set_ylabel(r"$\mathrm{Re}\,\widetilde\omega/\Omega$ (zone $[0,1)$)")
    axes[0,1].set_ylabel(r"$\mathrm{Im}\,\widetilde\omega/\Omega$")
    axes[0,0].legend(loc="best")
    axes[0,1].axhline(0,color=".6",lw=.6)
    axes[1,0].semilogy(k,np.maximum(data["spectral_error"],1e-17),color="#CC79A7")
    axes[1,0].set_xlabel(r"$\kappa=c_0 k/\Omega$")
    axes[1,0].set_ylabel("Spectral error versus monodromy")
    axes[1,1].semilogy(k,np.maximum(data["edge_weight"],1e-20),label="Edge harmonic weight",color="#009E73")
    axes[1,1].semilogy(k,np.maximum(data["residual"],1e-20),label="Matrix residual",color="#D55E00")
    axes[1,1].set_xlabel(r"$\kappa=c_0 k/\Omega$")
    axes[1,1].set_ylabel("Numerical diagnostic")
    axes[1,1].legend()
    for label,ax in zip("abcd",axes.ravel()):
        ax.set_title(label,loc="left",fontweight="bold")
    fig.suptitle(f"Homogeneous PTC benchmark (not Ge Fig. 4): m={config['modulation']}, N={config['n_harmonics']}",fontsize=10)
    _save(fig,Path(output)/"homogeneous_bands")


def plot_convergence(records, weak_data, output):
    style()
    fig,ax = plt.subplots(1,2,figsize=(7.,2.8),layout="constrained")
    valid = [r for r in records if r.get("spectral_error") is not None]
    ax[0].semilogy([r["n_harmonics"] for r in valid],
                   [max(r["spectral_error"],1e-17) for r in valid],"o-",color="#0072B2")
    ax[0].set_xlabel(r"Temporal harmonic cutoff $N_F$")
    ax[0].set_ylabel("Error versus independent monodromy")
    m,growth,leading = weak_data.T
    ax[1].loglog(m,np.abs(growth/leading-1),"o-",color="#D55E00",label="Computed relative error")
    ax[1].loglog(m,.1*m*m,"--",color=".5",label=r"Reference slope $m^2$")
    ax[1].set_xlabel("Modulation depth m")
    ax[1].set_ylabel("Error of leading Mathieu growth")
    ax[1].legend()
    ax[0].set_title("a  Harmonic convergence",loc="left")
    ax[1].set_title("b  Weak-modulation asymptotic check",loc="left")
    _save(fig,Path(output)/"homogeneous_convergence")
