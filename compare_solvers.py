#!/usr/bin/env python3
"""Run two solver scripts, time them, and compare their outputs."""

import argparse
import os
import subprocess
import sys
import tempfile
import tomllib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class SolverResult:
    name: str
    solver_path: Path
    parfile: Path
    runtime: float
    final_time: float
    variables: Dict[str, Tuple[np.ndarray, np.ndarray]]
    l2_errors: Dict[str, float]
    linf_errors: Dict[str, float]


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
    amp = params.get("id_amp", 1.0)
    omega = params.get("id_omega", 1.0)
    x0 = params.get("id_x0", 0.5)

    def profile(x_values: np.ndarray) -> np.ndarray:
        return amp * np.exp(-omega * (x_values - x0) ** 2)

    left_travel = profile(x - time)
    right_travel = profile(x + time)

    phi = 0.5 * (left_travel + right_travel)
    pi = 0.5 * (left_travel - right_travel)

    return {"Phi": phi, "Pi": pi}


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

        final_time, variables = parse_curve_file(curve_files[-1])

        with parfile.open("rb") as pf:
            params = tomllib.load(pf)

        if not variables:
            raise ValueError(f"No variables parsed from solver output for {solver_path}")

        reference_x = next(iter(variables.values()))[0]
        analytic = analytic_wave_solution(reference_x, final_time, params)

        l2_errors: Dict[str, float] = {}
        linf_errors: Dict[str, float] = {}

        for var, (x_values, data_values) in variables.items():
            if var not in analytic:
                continue
            if len(x_values) != len(reference_x) or not np.allclose(x_values, reference_x):
                raise ValueError(
                    f"Variable {var} from {solver_path} has grid mismatch for analytic comparison"
                )
            diff = data_values - analytic[var]
            l2_errors[var] = l2_norm(diff)
            linf_errors[var] = linf_norm(diff)

    return SolverResult(
        name=name,
        solver_path=solver_path,
        parfile=parfile,
        runtime=runtime,
        final_time=final_time,
        variables=variables,
        l2_errors=l2_errors,
        linf_errors=linf_errors,
    )



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run two solver scripts, time them, and compare their outputs.",
    )
    parser.add_argument("solver_a", help="Path to the first solver script")
    parser.add_argument("solver_b", help="Path to the second solver script")
    parser.add_argument("--parfile", dest="parfile", help="Override parfile for both solvers")
    args = parser.parse_args()

    solver_a_path, parfile_a = resolve_paths(args.solver_a, args.parfile)
    solver_b_path, parfile_b = resolve_paths(args.solver_b, args.parfile)

    result_a = run_solver("A", solver_a_path, parfile_a)
    result_b = run_solver("B", solver_b_path, parfile_b)

    def report(result: SolverResult) -> None:
        print(f"Solver {result.name}: {result.solver_path}")
        print(f"  Parfile: {result.parfile}")
        print(f"  Runtime: {result.runtime:.3f} s")
        print(f"  Final time: {result.final_time:.6e}")
        if result.l2_errors:
            print("  L2 errors vs analytic:")
            for var, err in sorted(result.l2_errors.items()):
                print(f"    {var}: {err:.6e}")
        if result.linf_errors:
            print("  Linf errors vs analytic:")
            for var, err in sorted(result.linf_errors.items()):
                print(f"    {var}: {err:.6e}")
        print()

    report(result_a)
    report(result_b)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
