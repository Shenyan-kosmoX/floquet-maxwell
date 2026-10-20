"""Static weak-modulation counterparts at the fixed paper half-drive.

Roots are checked with new native MPB solves. They are not direct dynamic
Floquet gap edges: the static counterpart construction is first order in m.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from mpb_nonresonant_reference import run


def main(output):
    output = Path(output)
    q0, epsilon0, resolution, lz = 0.2818, 11.68, 12, 64
    target = 91.5e12 * 631.8e-9 / 299792458

    def read_frequency(q, epsilon):
        folder = output / f"q{q:.12g}_eps{epsilon:.12g}_r{resolution}_Lz{lz:g}"
        return json.loads((folder / "summary.json").read_text())["modes"][0]["frequency"]

    def solve(q, epsilon):
        run(q, epsilon, resolution, lz, 3, output)
        return read_frequency(q, epsilon)

    slope = (read_frequency(q0 + .001, epsilon0)
             - read_frequency(q0 - .001, epsilon0)) / .002
    roots = []
    for epsilon in [11.6266, 11.68, 11.7334]:
        previous_q = q0
        previous_residual = read_frequency(q0, epsilon) - target
        current_q = q0 - previous_residual / slope
        history = [dict(q=q0, frequency=target + previous_residual)]
        for _ in range(4):
            frequency = solve(current_q, epsilon)
            residual = frequency - target
            history.append(dict(q=current_q, frequency=frequency))
            if abs(residual) < 2e-9:
                break
            derivative = (residual - previous_residual) / (current_q - previous_q)
            previous_q, previous_residual, current_q = current_q, residual, current_q - residual / derivative
        else:
            raise RuntimeError("Bounded MPB root did not converge")
        roots.append(dict(epsilon=epsilon, q=current_q, frequency=frequency,
                          frequency_residual=residual, evaluations=history))
    fm = read_frequency(q0, 11.6266)
    fp = read_frequency(q0, 11.7334)
    carrier = read_frequency(q0, 11.68)
    width = roots[2]["q"] - roots[0]["q"]
    result = dict(scope="Static weak-modulation counterpart crossing; not a direct dynamic Floquet gap",
        reference_backend="Native MPB EVEN_Z periodic-z supercell",
        resolution=resolution, lz=lz, m=.01, chi0=10.68, target_frequency=target,
        target_half_drive_THz=91.5, roots=roots, delta_q=width,
        relative_momentum_width_percent=100 * width / roots[1]["q"],
        local_q=q0, local_carrier=carrier, local_df_dq=slope,
        local_df_depsilon=(fp - fm) / .1068,
        local_width_from_secants=(fm - fp) / slope,
        local_relative_width_percent=100 * (fm - fp) / slope / q0,
        local_predicted_gain_per_meep_time=float(np.pi * (fm - fp)),
        local_predicted_amplitude_efold_pump_periods=float(2 * carrier / (np.pi * (fm - fp))),
        limitations=["Spatial resolution 12 has not yet been refined",
                     "Static counterparts approximate the first parametric tongue to leading modulation order",
                     "MPB and Meep have different spatial discretizations; do not use this drive without Meep calibration"])
    (output / "static_gap_reference.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
