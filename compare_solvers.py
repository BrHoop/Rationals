#!/usr/bin/env python3
"""Run solver scripts, time them, and compare their outputs with error plots."""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import tomllib

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class SolverResult:
    name: str
    solver_path: Path
    parfile: Path
    runtime: float
    times: np.ndarray
    final_time: float
    grid: np.ndarray
    variables: tuple[str, ...]
    l2_history: Dict[str, np.ndarray]
    linf_history: Dict[str, np.ndarray]


def resolve_paths(solver: str, parfile: str | None) -> Tuple[Path, Path]:
    solver_path = Path(solver).expanduser().resolve()
    if not solver_path.exists():
        raise FileNotFoundError(f"Solver script not found: {solver}")
    if parfile is not None:
        parfile_path = Path(parfile).expanduser().resolve()
    else:
        parfile_path = solver_path.with_name("parfile")
    if not parfile_path.exists():
        raise FileNotFoundError(f"Parfile not found for solver {solver_path}: {parfile_path}")
    return solver_path, parfile_path


def parse_curve_file(path: Path) -> Tuple[float, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    final_time: float | None = None
    variables: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    current_name: str | None = None
    xs: list[float] = []
    values: list[float] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line.upper().startswith("# TIME"):
                parts = line.split()
                if len(parts) >= 3:
                    final_time = float(parts[2])
            else:
                if current_name is not None:
                    variables[current_name] = (np.array(xs), np.array(values))
                current_name = line[1:].strip()
                xs = []
                values = []
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            xs.append(float(parts[0]))
            values.append(float(parts[1]))

    if current_name is not None:
        variables[current_name] = (np.array(xs), np.array(values))

    if final_time is None:
        raise ValueError(f"No time metadata found in {path}")

    return final_time, variables


def l2_norm(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values ** 2)))


def linf_norm(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def analytic_wave_solution(x: np.ndarray, time: float, params: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Analytic solution for the 1D wave equation with *periodic* boundary conditions.

    The initial data is a Gaussian centered at x0, periodized over a domain of length L.
    Left- and right-traveling components move at speed c.

    Expected (optional) parameters in `params` with reasonable fallbacks:
      - id_amp (float): amplitude of the Gaussian (default 1.0)
      - id_omega (float): Gaussian sharpness parameter (default 1.0)
      - id_x0 (float): initial center (default 0.5)
      - domain_length or L or id_L (float): periodic domain length (default 1.0)
      - wave_speed or c or id_c (float): wave speed (default 1.0)
      - periodic_images (int): number of image copies on each side for periodization (default 2)
    """
    amp = float(params.get("id_amp", 1.0))
    omega = float(params.get("id_omega", 1.0))
    x0 = float(params.get("id_x0", 0.5))

    # Try multiple common keys for domain length and speed; fall back to 1.0
    L = float(params.get("x_max", 1.0) - float(params.get("x_min", -1.0)))
    c = float(
        params.get("wave_speed",
        # params.get("cfl",
        params.get("id_c", 1.0))#
    )

    # Periodization control: either sum a few image copies ("sum") or use
    # a nearest-image wrap via modular arithmetic ("nearest"). The latter avoids
    # the costly image loop and is exact for a compactly supported profile and a
    # very accurate approximation for Gaussians when tails are small relative to L.
    periodization_mode = str(params.get("periodization", "nearest")).lower()
    images = int(params.get("periodic_images", 3))  # only used if mode == "sum"

    x = np.asarray(x, dtype=float)

    def periodized_profile(x_values: np.ndarray) -> np.ndarray:
        xv = np.asarray(x_values, dtype=float)
        if periodization_mode == "sum":
            out = np.zeros_like(xv, dtype=float)
            # Sum over a small number of images to enforce periodicity
            for m in range(-images, images + 1):
                out += amp * np.exp(-omega * (xv - x0 - m * L) ** 2)
            return out
        # "nearest" mode: wrap coordinate into the base cell using minimal-image distance
        # This avoids the O(images) loop and is typically sufficient for validation plots.
        d = ((xv - x0 + 0.5 * L) % L) - 0.5 * L
        return amp * np.exp(-omega * d ** 2)

    # Left- and right-moving waves with speed c on a periodic domain
    left_travel = periodized_profile(x - c * time)
    right_travel = periodized_profile(x + c * time)

    # Standard first-order reduction variables (Phi, Pi) for the wave equation
    phi = 0.5 * (left_travel + right_travel)
    pi = -0.5 * (left_travel - right_travel)

    return {"Phi": phi, "Pi": pi}

def create_grid(params):
    x_min = params["x_min"]
    x_max = params["x_max"]
    nx = params["nx"]
    dx = (x_max - x_min) / (nx -1)
    x = np.zeros(nx)
    for i in range(nx):
        x[i] = x_min + i*dx
    return x, dx

def output_curves_for_analytica(params: Dict[str, float]):
    nt = params['nt']
    x, dx = create_grid(params)
    time = 0.0
    dt = params["cfl"] * dx
    freq = params.get("output_frequency", 1)
    for i in range(nt+1):
        values = analytic_wave_solution(x, time, params)
        if (i % freq) == 0:
            fname = f"data_{i:04d}.curve"
            write_curve(fname, time, x, ["Phi","Pi"], list(values.values()))
        time += dt




def write_curve(filename, time, x, u_names, u):
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        for m in range(len(u_names)):
            f.write(f"# {u_names[m]}\n")
            for xi, di in zip(x, u[m]):
                f.write(f"{xi:.8e} {di:.8e}\n")

def aggregate_error(history: Dict[str, np.ndarray]) -> np.ndarray:
    if not history:
        raise ValueError("Cannot aggregate empty history")
    stacked = np.vstack([errors for _, errors in sorted(history.items())])
    return np.sqrt(np.mean(stacked ** 2, axis=0))


def max_error(history: Dict[str, np.ndarray]) -> np.ndarray:
    if not history:
        raise ValueError("Cannot aggregate empty history")
    stacked = np.vstack([errors for _, errors in sorted(history.items())])
    return np.max(stacked, axis=0)


def run_solver(name: str, solver_path: Path, parfile: Path) -> SolverResult:
    with tempfile.TemporaryDirectory(prefix=f"{solver_path.stem}_output_") as tmpdir:
        output_dir = Path(tmpdir)
        output_arg = str(output_dir) + os.sep
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(solver_path), str(parfile), output_arg],
            text=True,
            check=False,
            capture_output=True,
        )
        runtime = time.perf_counter() - start

        if proc.returncode != 0:
            raise RuntimeError(
                f"Solver {solver_path} failed with exit code {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        curve_files = sorted(output_dir.glob("*.curve"))
        if not curve_files:
            raise FileNotFoundError(f"No curve files produced by solver {solver_path}")

        with parfile.open("rb") as pf:
            params = tomllib.load(pf)

        times: list[float] = []
        l2_history: Dict[str, list[float]] = {}
        linf_history: Dict[str, list[float]] = {}
        tracked_vars: tuple[str, ...] | None = None
        reference_x: np.ndarray | None = None

        for curve_file in curve_files:
            current_time, variables = parse_curve_file(curve_file)
            if not variables:
                continue

            file_reference_x = next(iter(variables.values()))[0]
            if reference_x is None:
                reference_x = file_reference_x
            else:
                if len(file_reference_x) != len(reference_x) or not np.allclose(
                    file_reference_x, reference_x
                ):
                    raise ValueError(
                        f"Grid mismatch detected in solver output for {solver_path} ({curve_file})"
                    )

            if reference_x is None:
                continue

            analytic = analytic_wave_solution(reference_x, current_time, params)
            available_vars = sorted(set(variables.keys()) & set(analytic.keys()))

            if tracked_vars is None:
                if not available_vars:
                    continue
                tracked_vars = tuple(available_vars)
                for key in tracked_vars:
                    l2_history[key] = []
                    linf_history[key] = []
            else:
                missing = [var for var in tracked_vars if var not in variables]
                if missing:
                    raise ValueError(
                        f"Variables {missing} missing in solver output {curve_file} from {solver_path}"
                    )

            for var in tracked_vars or ():
                if var not in analytic:
                    raise ValueError(
                        f"Analytic solution does not provide variable {var}"
                    )
                x_values, data_values = variables[var]
                if len(x_values) != len(reference_x) or not np.allclose(x_values, reference_x):
                    raise ValueError(
                        f"Variable {var} from {solver_path} has grid mismatch for analytic comparison"
                    )
                diff = data_values - analytic[var]
                l2_history[var].append(l2_norm(diff))
                linf_history[var].append(linf_norm(diff))

            if tracked_vars is not None:
                times.append(current_time)

        if not times:
            raise ValueError(f"No usable curve data produced by solver {solver_path}")
        if tracked_vars is None or reference_x is None:
            raise ValueError(
                f"No analytic comparison could be performed for solver {solver_path}"
            )

        times_array = np.array(times)
        sort_idx = np.argsort(times_array)
        times_array = times_array[sort_idx]
        l2_history_array = {var: np.array(values)[sort_idx] for var, values in l2_history.items()}
        linf_history_array = {var: np.array(values)[sort_idx] for var, values in linf_history.items()}
        final_time = float(times_array[-1])

    return SolverResult(
        name=name,
        solver_path=solver_path,
        parfile=parfile,
        runtime=runtime,
        times=times_array,
        final_time=final_time,
        grid=reference_x,
        variables=tracked_vars,
        l2_history=l2_history_array,
        linf_history=linf_history_array,
    )


def plot_error_history(
    results: list[SolverResult],
    histories: Dict[str, np.ndarray],
    *,
    ylabel: str,
    filename: str,
) -> Path:
    if not histories:
        raise ValueError("No error data available to plot.")

    fig, ax = plt.subplots()
    styles = [".",".","."]
    for result, style in zip(results, styles):
        values = histories.get(result.name)
        if values is None:
            continue
        if values.size != result.times.size:
            raise ValueError(
                f"Error history for solver {result.name} does not match its time samples"
            )
        ax.plot(result.times, values, style, label=f"Solver {result.name}")

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} over Time")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()
    fig.tight_layout()

    output_path = Path(filename).resolve()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run solver scripts, time them, and compare their outputs.",
    )
    parser.add_argument(
        "solvers",
        nargs="+",
        help="Paths to the solver scripts (provide two or three).",
    )
    parser.add_argument("--parfile", dest="parfile", help="Override parfile for both solvers")
    args = parser.parse_args()

    solver_count = len(args.solvers)
    if solver_count < 2 or solver_count > 3:
        parser.error("Please provide two or three solver scripts to compare.")

    solver_specs: list[tuple[str, Path, Path]] = []
    for idx, solver_arg in enumerate(args.solvers):
        name = chr(ord("A") + idx)
        solver_path, parfile_path = resolve_paths(solver_arg, args.parfile)
        solver_specs.append((name, solver_path, parfile_path))

    results: list[SolverResult] = []
    for name, solver_path, parfile_path in solver_specs:
        print(f"[Info] Running Solver {name}: {solver_path}")
        results.append(run_solver(name, solver_path, parfile_path))
    #output_curves_for_analytica(params)
    label_suffix = "".join(result.name for result in results)

    variable_sets = [set(result.l2_history.keys()) for result in results if result.l2_history]
    if not variable_sets:
        raise ValueError("No variables available for analytic comparison across solvers")

    common_vars = tuple(sorted(set.intersection(*variable_sets)))
    if not common_vars:
        raise ValueError("No common variables across solvers for analytic comparison")

    phi_l2 = {
        result.name: aggregate_error({ "Phi" :result.l2_history["Phi"]})
        for result in results
    }
    pi_l2 = {
        result.name: aggregate_error({ "Pi" :result.l2_history["Pi"]})
        for result in results
    }
    phi_linf = {
        result.name: max_error({ "Phi" :result.l2_history["Phi"]})
        for result in results
    }
    pi_linf = {
        result.name: max_error({ "Pi" :result.l2_history["Pi"]})
        for result in results
    }
    #TODO make it so the analytical solution can output wave files so I can check if everything is working correctly.
    l2_plot_phi = plot_error_history(
        results,
        phi_l2,
        ylabel="L2 Error vs Analytic [phi]",
        filename=f"solver_l2_errors_phi.png",
    )
    l2_plot_pi = plot_error_history(
        results,
        pi_l2,
        ylabel="L2 Error vs Analytic [pi]",
        filename=f"solver_l2_errors_pi.png",
    )
    linf_plot_phi = plot_error_history(
        results,
        phi_linf,
        ylabel="Linf Error vs Analytic [phi]",
        filename=f"solver_linf_errors_phi.png",
    )
    linf_plot_pi = plot_error_history(
        results,
        pi_linf,
        ylabel="Linf Error vs Analytic [pi]",
        filename=f"solver_linf_errors_pi.png",
    )

    def report(result: SolverResult) -> None:
        print(f"Solver {result.name}: {result.solver_path}")
        print(f"  Parfile: {result.parfile}")
        print(f"  Runtime: {result.runtime:.3f} s")
        print(f"  Final time: {result.final_time:.6e}")
        if common_vars:
            print(f"  Variables compared: {', '.join(common_vars)}")
        final_l2 = phi_l2[result.name][-1]
        final_linf = phi_linf[result.name][-1]
        print(f"  Final combined L2 error: {final_l2:.6e}")
        print(f"  Final combined Linf error: {final_linf:.6e}")
        print()

    for result in results:
        report(result)

    print("[Info] Errors computed against the analytic solution.")
    print(f"[Info] L2 error plot saved to: {l2_plot_pi}")
    print(f"[Info] Linf error plot saved to: {linf_plot_pi}")
    print(f"[Info] L2 error plot saved to: {l2_plot_phi}")
    print(f"[Info] Linf error plot saved to: {linf_plot_phi}")


if __name__ == "__main__":
    main()