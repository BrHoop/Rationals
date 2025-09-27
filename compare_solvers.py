#!/usr/bin/env python3
"""Run solver scripts, time them, and compare their outputs with error plots."""

import argparse
import ast
import itertools
import os
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    data_history: Dict[str, np.ndarray]
    refinement_level: int = 0
    l2_history: Dict[str, np.ndarray] = field(default_factory=dict)
    linf_history: Dict[str, np.ndarray] = field(default_factory=dict)


def resolve_paths(solver: str, parfile: Optional[str]) -> Tuple[Path, Path]:
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
    final_time: Optional[float] = None
    variables: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    current_name: Optional[str] = None
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


def read_parfile(path: Path) -> "OrderedDict[str, Any]":
    params: "OrderedDict[str, Any]" = OrderedDict()
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.split("#", 1)[0].strip()
        if not value:
            continue
        try:
            params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            params[key] = value
    return params


def write_parfile(path: Path, params: "OrderedDict[str, Any]") -> None:
    with path.open("w") as fh:
        for key, value in params.items():
            fh.write(f"{key} = {repr(value)}\n")


def refined_parameters(
    base_params: "OrderedDict[str, Any]",
    *,
    level: int,
    factor: int,
) -> "OrderedDict[str, Any]":
    if level < 0:
        raise ValueError("Refinement level must be non-negative")
    if factor < 1:
        raise ValueError("Refinement factor must be at least 1")

    params = OrderedDict(base_params.items())
    if level == 0:
        return params

    intervals = int(base_params["nx"]) - 1
    if intervals <= 0:
        raise ValueError("nx must be at least 2 in the base parfile")

    scaled = factor ** level
    params["nx"] = intervals * scaled + 1

    base_nt = int(base_params["nt"])
    params["nt"] = base_nt * scaled

    base_freq = int(base_params.get("output_frequency", 1))
    params["output_frequency"] = max(1, base_freq * scaled)

    return params


def grid_refinement_ratio(fine_size: int, coarse_size: int) -> int:
    if fine_size < coarse_size:
        raise ValueError("Reference grid must be at least as fine as target grid")
    fine_intervals = fine_size - 1
    coarse_intervals = max(1, coarse_size - 1)
    if fine_intervals % coarse_intervals != 0:
        raise ValueError("Grids are not nested by an integer refinement factor")
    return fine_intervals // coarse_intervals


def compare_histories_between_levels(
    coarse: SolverResult,
    fine: SolverResult,
    *,
    ratio: int,
    variables: Tuple[str, ...],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if ratio < 1:
        raise ValueError("Ratio must be a positive integer")

    if coarse.times.size == 0 or fine.times.size == 0:
        raise ValueError("No time samples available for comparison")

    shared = min(coarse.times.size, fine.times.size)
    if shared == 0:
        raise ValueError("No overlapping time samples between refinement levels")

    if not np.allclose(
        coarse.times[:shared],
        fine.times[:shared],
        rtol=1e-5,
        atol=1e-8,
    ):
        max_diff = float(
            np.max(np.abs(coarse.times[:shared] - fine.times[:shared]))
        )
        raise ValueError(
            "Time samples do not align between refinement levels "
            f"(max difference {max_diff:.3e})"
        )

    l2_histories: Dict[str, np.ndarray] = {}
    linf_histories: Dict[str, np.ndarray] = {}

    for var in variables:
        coarse_data = coarse.data_history[var]
        fine_data = fine.data_history[var]
        if ratio == 1:
            downsampled = fine_data
        else:
            downsampled = fine_data[:, ::ratio]

        min_len = min(coarse_data.shape[0], downsampled.shape[0])
        if min_len == 0:
            raise ValueError(
                f"Variable {var} has no overlapping time samples after downsampling"
            )

        coarse_slice = coarse_data[:min_len]
        fine_slice = downsampled[:min_len]
        if coarse_slice.shape != fine_slice.shape:
            raise ValueError(
                f"Variable {var} does not downsample cleanly (ratio {ratio})"
            )
        diff = coarse_slice - fine_slice
        l2_histories[var] = np.sqrt(np.mean(diff ** 2, axis=1))
        linf_histories[var] = np.max(np.abs(diff), axis=1)

    return l2_histories, linf_histories


def run_solver(
    name: str,
    solver_path: Path,
    parfile: Path,
    *,
    refinement_level: int = 0,
) -> SolverResult:
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

        times: list[float] = []
        variable_samples: Dict[str, list[np.ndarray]] = {}
        tracked_vars: Optional[Tuple[str, ...]] = None
        reference_x: Optional[np.ndarray] = None

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

            available_vars = sorted(variables.keys())
            if tracked_vars is None:
                if not available_vars:
                    continue
                tracked_vars = tuple(available_vars)
                variable_samples = {var: [] for var in tracked_vars}
            else:
                missing = [var for var in tracked_vars if var not in variables]
                if missing:
                    raise ValueError(
                        f"Variables {missing} missing in solver output {curve_file} from {solver_path}"
                    )

            for var in tracked_vars or ():
                x_values, data_values = variables[var]
                if reference_x is None:
                    continue
                if len(x_values) != len(reference_x) or not np.allclose(x_values, reference_x):
                    raise ValueError(
                        f"Variable {var} from {solver_path} has grid mismatch across outputs"
                    )
                variable_samples[var].append(np.array(data_values, copy=True))

            if tracked_vars is not None:
                times.append(current_time)

        if not times:
            raise ValueError(f"No usable curve data produced by solver {solver_path}")
        if tracked_vars is None or reference_x is None:
            raise ValueError(
                f"No variable data could be collected for solver {solver_path}"
            )

        times_array = np.array(times)
        sort_idx = np.argsort(times_array)
        times_array = times_array[sort_idx]
        data_history = {
            var: np.stack(values, axis=0)[sort_idx]
            for var, values in variable_samples.items()
        }
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
        refinement_level=refinement_level,
        data_history=data_history,
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
    marker_cycle = itertools.cycle(["o", "s", "^", "d", "x", "*", "."])
    for result in results:
        values = histories.get(result.name)
        if values is None:
            continue
        if values.size != result.times.size:
            raise ValueError(
                f"Error history for solver {result.name} does not match its time samples"
            )
        marker = next(marker_cycle)
        ax.plot(
            result.times,
            values,
            marker=marker,
            label=f"Solver {result.name}",
        )

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
    parser.add_argument("--parfile", dest="parfile", help="Override parfile for all solvers")
    parser.add_argument(
        "--refine-factor",
        type=int,
        default=2,
        help="Grid refinement multiplier per level for the reference solver (>= 2).",
    )
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=1e-4,
        help="Target aggregate L2 difference between successive refinements.",
    )
    parser.add_argument(
        "--max-refinements",
        type=int,
        default=4,
        help="Maximum additional refinement levels for the reference solver.",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=0,
        help="Zero-based index of the solver to refine for the reference solution.",
    )
    args = parser.parse_args()

    solver_count = len(args.solvers)
    if solver_count < 2 or solver_count > 3:
        parser.error("Please provide two or three solver scripts to compare.")

    if args.reference_index < 0 or args.reference_index >= solver_count:
        parser.error(
            f"Reference index must be between 0 and {solver_count - 1} (received {args.reference_index})."
        )

    if args.max_refinements < 0:
        parser.error("Maximum refinements must be non-negative.")

    if args.max_refinements > 0 and args.refine_factor < 2:
        parser.error("Refinement factor must be >= 2 when refinements are requested.")

    solver_specs: List[Tuple[str, Path, Path]] = []
    base_params: List[OrderedDict[str, Any]] = []
    for idx, solver_arg in enumerate(args.solvers):
        name = chr(ord("A") + idx)
        solver_path, parfile_path = resolve_paths(solver_arg, args.parfile)
        solver_specs.append((name, solver_path, parfile_path))
        base_params.append(read_parfile(parfile_path))

    results: List[SolverResult] = []
    for name, solver_path, parfile_path in solver_specs:
        print(f"[Info] Running Solver {name}: {solver_path}")
        results.append(
            run_solver(name, solver_path, parfile_path, refinement_level=0)
        )

    if not results:
        raise ValueError("No solver results were produced.")

    reference_index = args.reference_index
    reference_spec = solver_specs[reference_index]
    reference_base_params = base_params[reference_index]
    reference_result = results[reference_index]
    previous_result = reference_result
    reference_refinement_level = 0
    convergence_reached = False
    last_difference = 0.0

    for level in range(1, args.max_refinements + 1):
        params_level = refined_parameters(
            reference_base_params,
            level=level,
            factor=args.refine_factor,
        )
        with tempfile.TemporaryDirectory(prefix="refine_parfile_") as tmpdir:
            refined_parfile = Path(tmpdir) / "parfile"
            write_parfile(refined_parfile, params_level)
            refined_result = run_solver(
                reference_spec[0],
                reference_spec[1],
                refined_parfile,
                refinement_level=level,
            )

        shared_vars = tuple(
            sorted(
                set(previous_result.data_history.keys())
                & set(refined_result.data_history.keys())
            )
        )
        if not shared_vars:
            raise ValueError(
                f"No common variables between refinement levels for solver {reference_spec[0]}"
            )

        ratio = grid_refinement_ratio(refined_result.grid.size, previous_result.grid.size)
        l2_diff, linf_diff = compare_histories_between_levels(
            previous_result,
            refined_result,
            ratio=ratio,
            variables=shared_vars,
        )
        aggregate = aggregate_error(l2_diff)
        linf_aggregate = max_error(linf_diff)
        last_difference = float(aggregate[-1])
        last_linf = float(linf_aggregate[-1])
        print(
            f"[Info] Solver {reference_spec[0]} refinement level {level}: "
            f"aggregate L2 diff = {last_difference:.3e}, "
            f"aggregate Linf diff = {last_linf:.3e}"
        )

        reference_result = refined_result
        reference_refinement_level = level
        if last_difference <= args.convergence_tol:
            convergence_reached = True
            break

        previous_result = refined_result

    reference_result.parfile = reference_spec[2]

    variable_sets = [set(result.data_history.keys()) for result in results if result.data_history]
    variable_sets.append(set(reference_result.data_history.keys()))
    if not variable_sets:
        raise ValueError("No variables available for convergence comparison across solvers")

    common_vars = tuple(sorted(set.intersection(*variable_sets)))
    if not common_vars:
        raise ValueError("No common variables across solvers for convergence comparison")

    per_var_l2 = {var: {} for var in common_vars}
    per_var_linf = {var: {} for var in common_vars}
    combined_l2: Dict[str, np.ndarray] = {}
    combined_linf: Dict[str, np.ndarray] = {}
    grid_ratios: Dict[str, int] = {}

    for result in results:
        ratio = grid_refinement_ratio(reference_result.grid.size, result.grid.size)
        grid_ratios[result.name] = ratio
        l2_hist, linf_hist = compare_histories_between_levels(
            result,
            reference_result,
            ratio=ratio,
            variables=common_vars,
        )
        result.l2_history = l2_hist
        result.linf_history = linf_hist
        if l2_hist:
            first_key = next(iter(l2_hist))
            aligned_length = l2_hist[first_key].shape[0]
            result.times = result.times[:aligned_length]
        combined_l2[result.name] = aggregate_error(result.l2_history)
        combined_linf[result.name] = max_error(result.linf_history)
        for var in common_vars:
            per_var_l2[var][result.name] = result.l2_history[var]
            per_var_linf[var][result.name] = result.linf_history[var]

    phi_l2_plot = phi_linf_plot = None
    if "Phi" in per_var_l2:
        phi_l2_plot = plot_error_history(
            results,
            per_var_l2["Phi"],
            ylabel="L2 Error vs Reference [Phi]",
            filename="solver_l2_errors_phi.png",
        )
        phi_linf_plot = plot_error_history(
            results,
            per_var_linf["Phi"],
            ylabel="Linf Error vs Reference [Phi]",
            filename="solver_linf_errors_phi.png",
        )

    pi_l2_plot = pi_linf_plot = None
    if "Pi" in per_var_l2:
        pi_l2_plot = plot_error_history(
            results,
            per_var_l2["Pi"],
            ylabel="L2 Error vs Reference [Pi]",
            filename="solver_l2_errors_pi.png",
        )
        pi_linf_plot = plot_error_history(
            results,
            per_var_linf["Pi"],
            ylabel="Linf Error vs Reference [Pi]",
            filename="solver_linf_errors_pi.png",
        )

    def report(result: SolverResult) -> None:
        print(f"Solver {result.name}: {result.solver_path}")
        print(f"  Parfile: {result.parfile}")
        print(f"  Runtime: {result.runtime:.3f} s")
        print(f"  Final time: {result.final_time:.6e}")
        print(f"  Grid refinement ratio vs reference: {grid_ratios[result.name]:d}x")
        print(f"  Variables compared: {', '.join(common_vars)}")
        final_l2 = combined_l2[result.name][-1]
        final_linf = combined_linf[result.name][-1]
        print(f"  Final combined L2 error: {final_l2:.6e}")
        print(f"  Final combined Linf error: {final_linf:.6e}")
        print()

    for result in results:
        report(result)

    if args.max_refinements == 0:
        print(
            f"[Info] Reference solver {reference_spec[0]} uses the base resolution (no refinements requested)."
        )
    elif convergence_reached:
        print(
            f"[Info] Reference solver {reference_spec[0]} converged at level {reference_refinement_level} "
            f"with aggregate L2 difference {last_difference:.3e} (tolerance {args.convergence_tol:.3e})."
        )
    else:
        print(
            f"[Warn] Reference solver {reference_spec[0]} did not reach the tolerance within "
            f"{args.max_refinements} refinements; using level {reference_refinement_level} "
            f"with final aggregate L2 difference {last_difference:.3e}."
        )

    if phi_l2_plot is not None:
        print(f"[Info] Phi L2 error plot saved to: {phi_l2_plot}")
    if phi_linf_plot is not None:
        print(f"[Info] Phi Linf error plot saved to: {phi_linf_plot}")
    if pi_l2_plot is not None:
        print(f"[Info] Pi L2 error plot saved to: {pi_l2_plot}")
    if pi_linf_plot is not None:
        print(f"[Info] Pi Linf error plot saved to: {pi_linf_plot}")


if __name__ == "__main__":
    main()
