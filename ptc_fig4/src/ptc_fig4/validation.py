"""Reproducible CPU benchmark, raw spectra, atomic checkpoints and evidence gate."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
from scipy.optimize import brentq
import yaml
from .materials import SinusoidalDielectric, InverseSinusoidalDielectric, LorentzDielectric
from .floquet_maxwell import solve_floquet
from .references import (monodromy_dielectric, monodromy_lorentz,
    quasifrequencies_from_monodromy, matched_spectral_error,
    weak_modulation_prediction, mathieu_first_gap)
from .mode_tracking import track_modes
from .convergence import harmonic_convergence
from .plotting import plot_homogeneous_bands, plot_convergence


def _json_write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def _signature(config):
    h = hashlib.sha256(json.dumps(config,sort_keys=True).encode())
    source_names = ["materials.py","floquet_maxwell.py","references.py",
                    "mode_tracking.py","convergence.py","validation.py","plotting.py"]
    for p in [Path(__file__).parent/name for name in source_names]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def first_gap_from_monodromy(model, bracket):
    """Locate the first instability region using tr(U)=-2, det(U)=1.

    Returns boundaries in kappa. The fundamental matrix is real for this
    lossless benchmark. This is an independent check, not used in F assembly.
    """
    def discriminant(k):
        return float(np.trace(monodromy_dielectric(model,k,rtol=8e-13,atol=8e-15))+2)
    grid = np.linspace(*bracket,161)
    val = [discriminant(k) for k in grid]
    roots = [brentq(discriminant,grid[i],grid[i+1],xtol=2e-12)
             for i in range(len(grid)-1) if val[i]*val[i+1] < 0]
    if len(roots) != 2:
        raise RuntimeError(f"Expected one resolved first gap, found {len(roots)} boundaries")
    return roots


def run_validation(config_path, output_directory, resume=True):
    started = time.perf_counter()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output_directory)
    output.mkdir(parents=True,exist_ok=True)
    raw = output/"raw"
    raw.mkdir(exist_ok=True)
    signature = _signature(config)
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise RuntimeError("Checkpoint config/code signature mismatch; choose a new output directory")
    versions = {name:importlib.metadata.version(name)
                for name in ["numpy","scipy","matplotlib","PyYAML","ptc-fig4"]}
    _json_write(manifest_path,dict(signature=signature,config=config,versions=versions,
                                 python=sys.version,platform=platform.platform(),
                                 units="tau=Omega*t; kappa=c0*k/Omega; w=omega/Omega",
                                 convention="exp(+ikx-iwt); Im(w)>0 means growth"))
    model = SinusoidalDielectric(config["epsilon_base"],config["modulation"],
                                  config["dc_shift"],config["phase"])
    ks = np.linspace(config["k_min"],config["k_max"],config["k_points"])
    fields = {name:[] for name in ["quasifrequencies","reference","spectral_error",
                                  "edge_weight","residual","condition_number",
                                  "tracking_ambiguous","tracking_overlap"]}
    previous,previous_idx = None,None
    for number,k in enumerate(ks):
        # Eigendecomposition is cheap here; resume reuses the expensive ODE
        # reference while independently reconstructing tracking order.
        checkpoint = raw/f"k_{number:04d}.npz"
        s = solve_floquet(model,float(k),config["n_harmonics"],config["n_samples"])
        idx = s.physical_modes()
        ambiguity,overlap = False,1.
        if previous is not None:
            tracking = track_modes(previous,previous_idx,s,idx)
            idx = idx[tracking.permutation]
            ambiguity,overlap = tracking.ambiguous,float(tracking.overlaps.min())
        if resume and checkpoint.exists():
            with np.load(checkpoint,allow_pickle=False) as saved:
                if str(saved["signature"]) != signature or abs(float(saved["kappa"])-k)>1e-15:
                    raise RuntimeError("Corrupt or incompatible checkpoint")
                u = saved["monodromy"]
        else:
            u = monodromy_dielectric(model,k,rtol=config["monodromy_rtol"],
                                    atol=config["monodromy_atol"])
        reference = quasifrequencies_from_monodromy(u)
        error,assignment = matched_spectral_error(s.eigenvalues[idx],reference)
        reference = reference[assignment]
        temp = checkpoint.with_suffix(".tmp.npz")
        np.savez_compressed(temp,signature=signature,kappa=k,eigenvalues=s.eigenvalues,
                            right=s.right,left=s.left,selected_indices=idx,
                            residuals=s.residuals,left_residuals=s.left_residuals,
                            edge_weights=s.edge_weights,mean_harmonics=s.mean_harmonics,
                            condition_numbers=s.condition_numbers,monodromy=u,
                            reference=reference,tracking_ambiguous=ambiguity)
        temp.replace(checkpoint)
        for name,value in dict(quasifrequencies=s.eigenvalues[idx],reference=reference,
                               spectral_error=error,edge_weight=s.edge_weights[idx].max(),
                               residual=max(s.residuals.max(),s.left_residuals.max()),
                               condition_number=s.condition_numbers[idx].max(),
                               tracking_ambiguous=ambiguity,tracking_overlap=overlap).items():
            fields[name].append(value)
        previous,previous_idx = s,idx
    data = {key:np.asarray(value) for key,value in fields.items()}
    data["kappa"] = ks
    np.savez_compressed(output/"bands.npz",**data)
    gap = first_gap_from_monodromy(model,(config["k_min"],config["k_max"]))
    convergence = harmonic_convergence(model,np.sqrt(model.mean_epsilon)/2,
                                        config["convergence_harmonics"],raw)
    weak = []
    for m in [.04,.02,.01,.005,.0025,.00125]:
        material = SinusoidalDielectric(4,m,True)
        k = np.sqrt(material.mean_epsilon)/2
        s = solve_floquet(material,k,7)
        s.save(raw/f"weak_m_{m:.5f}.npz")
        growth = float(s.eigenvalues[s.physical_modes()].imag.max())
        weak.append([m,growth,weak_modulation_prediction(material,k)])
    weak = np.asarray(weak)
    np.savetxt(output/"weak_mathieu.csv",weak,delimiter=",",
               header="modulation,computed_growth,leading_order_growth",comments="")
    _json_write(output/"convergence.json",convergence)
    inverse = InverseSinusoidalDielectric(4,.12)
    exact_edges = mathieu_first_gap(4,.12)
    ode_edges = first_gap_from_monodromy(inverse,(.8,1.2))
    dispersive = LorentzDielectric(modulation=.13,phase=.43)
    ds = solve_floquet(dispersive,1.07,9)
    ds.save(raw/"lorentz_auxiliary.npz")
    du = monodromy_lorentz(dispersive,1.07,rtol=5e-13,atol=5e-15)
    lorentz_error,_ = matched_spectral_error(ds.eigenvalues[ds.physical_modes()],
                                             quasifrequencies_from_monodromy(du))
    np.savez_compressed(raw/"lorentz_reference.npz",monodromy=du)
    last_error = convergence[-1].get("spectral_error") if convergence else None
    checks = {
        "full_two_mode_spectrum_at_every_k":bool(data["quasifrequencies"].shape==(len(ks),2)),
        "monodromy_spectral_error_below_3e-9":bool(data["spectral_error"].max()<3e-9),
        "left_and_right_residual_below_2e-14":bool(data["residual"].max()<2e-14),
        "edge_weight_below_1e-6":bool(data["edge_weight"].max()<1e-6),
        "harmonic_convergence_below_2e-10":bool(last_error is not None and np.isfinite(last_error)
                                                and last_error<2e-10),
        "exact_mathieu_gap_edges_error_below_2e-9":bool(np.max(abs(exact_edges-ode_edges))<2e-9),
        "weak_mathieu_relative_error_below_2e-5":bool(abs(weak[-1,1]/weak[-1,2]-1)<2e-5),
        "four_mode_lorentz_spectral_error_below_2e-9":bool(lorentz_error<2e-9),
        "growth_and_decay_both_present":bool(np.max(data["quasifrequencies"].imag)>.001
                                              and np.min(data["quasifrequencies"].imag)<-.001),
    }
    report = dict(schema_version=1,status="VERIFIED" if all(checks.values()) else "FAILED",
                  scope="Homogeneous transverse Maxwell, specified smooth periodic media; not Ge Fig.4",
                  checks=checks,signature=signature,k_points=len(ks),
                  max_spectral_error=float(data["spectral_error"].max()),
                  max_eigen_residual=float(data["residual"].max()),
                  max_edge_weight=float(data["edge_weight"].max()),
                  max_eigenvector_condition_number=float(data["condition_number"].max()),
                  ambiguous_tracking_points=int(data["tracking_ambiguous"].sum()),
                  first_gap_kappa=gap,first_gap_width=float(gap[1]-gap[0]),
                  max_growth=float(data["quasifrequencies"].imag.max()),
                  mathieu_exact_gap_kappa=exact_edges.tolist(),mathieu_ode_gap_kappa=ode_edges,
                  lorentz_spectral_error=float(lorentz_error),
                  elapsed_seconds=time.perf_counter()-started,
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  limits=["Finite test campaign, not a universal proof",
                          "Exact defective degeneracies require invariant-subspace analysis",
                          "No spatial radiation, Ge coefficients, or Fig.4 reproduction validated"])
    _json_write(output/"validation_report.json",report)
    plot_homogeneous_bands(data,gap,config,output)
    plot_convergence(convergence,weak,output)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/homogeneous_quick.yaml")
    p.add_argument("--output",default="results/homogeneous_quick")
    p.add_argument("--no-resume",action="store_true")
    args = p.parse_args()
    report = run_validation(args.config,args.output,not args.no_resume)
    print(json.dumps(report,indent=2))
    if report["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
