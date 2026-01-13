import sys
import os
import time
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

# --- CONFIGURATION ---
PYTHON_EXE = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parent

# Path to Solvers
REF_SOLVER_PATH = PROJECT_ROOT / "FP_64_Jax" / "newSolver.py"
BFP_SOLVER_PATH = PROJECT_ROOT / "Block_FP" / "JaxSolver.py"
DATA_DIR = PROJECT_ROOT / "data_compare"

# --- DATA STRUCTURES ---
@dataclass
class SolverRunResult:
    name: str
    script_path: Path
    output_dir: Path
    runtime: float
    dt: float = 0.0
    time_series: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_series: np.ndarray = field(default_factory=lambda: np.array([]))
    trace_series: np.ndarray = field(default_factory=lambda: np.array([]))
    final_fields: np.ndarray = field(default_factory=lambda: np.array([]))

def create_params_file(params: dict, path: Path):
    with open(path, "w") as f:
        for k, v in params.items():
            if isinstance(v, str):
                f.write(f'{k} = "{v}"\n')
            else:
                f.write(f'{k} = {v}\n')

def execute_solver(spec: dict, params: dict) -> SolverRunResult:
    name = spec['name']
    script_path = spec['solver']
    
    if not script_path.exists():
        raise FileNotFoundError(f"Solver script not found at: {script_path}")

    # Setup Run Directory
    run_dir = DATA_DIR / name.replace(" ", "_")
    if run_dir.exists(): shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    param_file = run_dir / "params.toml"
    create_params_file(params, param_file)
    
    print(f"  [Exec] {name}...")
    start_time = time.perf_counter()
    
    cmd = [PYTHON_EXE, str(script_path), str(param_file), str(run_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    runtime = time.perf_counter() - start_time
    
    if proc.returncode != 0:
        print(f"  [ERROR] {name} failed!")
        print("  STDOUT:", proc.stdout)
        print("  STDERR:", proc.stderr)
        raise RuntimeError(f"{name} crashed (Exit Code {proc.returncode})")

    # --- RESULTS PROCESSING ---
    result = SolverRunResult(name, script_path, run_dir, runtime)
    h5_files = sorted(list(run_dir.glob("*.h5")))
    
    times, energies, traces = [], [], []
    
    # Grid params for Energy
    nx, ny = params["Nx"], params["Ny"]
    dx = (params["Xmax"] - params["Xmin"]) / (nx - 1)
    
    for f in h5_files:
        with h5py.File(f, 'r') as hf:
            # [FIX] Read 'time' attribute safely
            t_val = hf.attrs.get('time', 0.0)
            if isinstance(t_val, bytes): t_val = float(t_val) # Handle legacy encoding
            
            # [FIX] Read Datasets by Name ("phi", "chi") instead of "data"
            if "phi" in hf and "chi" in hf:
                phi = hf["phi"][:]
                chi = hf["chi"][:]
            elif "data" in hf:
                # Fallback if someone changed the writer back
                u = hf['data'][:]
                phi, chi = u[0], u[1]
            else:
                # Fallback: Read first two keys found
                keys = list(hf.keys())
                phi = hf[keys[0]][:]
                chi = hf[keys[1]][:]

            # Linear Energy: T + 0.5 * (grad_phi)^2
            T = 0.5 * chi**2
            dphi_x = np.gradient(phi, dx, axis=0)
            dphi_y = np.gradient(phi, dx, axis=1)
            E = np.sum(T + 0.5*(dphi_x**2 + dphi_y**2)) * dx * dx
            
            energies.append(E)
            traces.append(phi[nx//2, ny//2])
            times.append(t_val)
            
    result.time_series = np.array(times)
    result.energy_series = np.array(energies)
    result.trace_series = np.array(traces)
    
    if len(times) > 1:
        result.dt = times[1] - times[0]
        
    # Store Final Field
    if h5_files:
        with h5py.File(h5_files[-1], 'r') as hf:
            if "phi" in hf:
                result.final_fields = hf["phi"][:]
            elif "data" in hf:
                result.final_fields = hf["data"][0]
            else:
                result.final_fields = hf[list(hf.keys())[0]][:]
            
    return result

def compare_runs():
    grid_size = 128
    shared_params = {
        "Nx": grid_size, "Ny": grid_size,
        "Xmin": -5.0, "Xmax": 5.0,
        "Ymin": -5.0, "Ymax": 5.0,
        "Nt": 1000,
        "output_interval": 20,
        "cfl": 0.05,
        "id_x0": 0.0, "id_y0": 0.0,
        "id_sigma": 0.5, "id_amp": 1.0,
        "ko_sigma": 0.05
    }

    solvers = [
        {"name": "Ref_Float64", "solver": REF_SOLVER_PATH},
        {"name": "BFP_50bit",   "solver": BFP_SOLVER_PATH},
    ]

    print(f"--- Comparison Suite (Grid {grid_size}x{grid_size}) ---")
    runs = []
    
    for spec in solvers:
        try:
            res = execute_solver(spec, shared_params)
            runs.append(res)
            print(f"  -> Success: {len(res.time_series)} frames")
        except Exception as e:
            print(f"  -> Failed: {e}")

    if len(runs) < 2:
        print("Comparison aborted: need 2 successful runs.")
        return

    ref, bfp = runs[0], runs[1]
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    # 1. Energy
    ax = axes[0]
    ax.plot(ref.time_series, ref.energy_series / ref.energy_series[0], label="Ref")
    ax.plot(bfp.time_series, bfp.energy_series / bfp.energy_series[0], label="BFP", linestyle="--")
    ax.set_title("Energy Stability (Normalized)")
    ax.set_ylabel("E / E0")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Trace
    ax = axes[1]
    ax.plot(ref.time_series, ref.trace_series, label="Ref")
    ax.plot(bfp.time_series, bfp.trace_series, label="BFP", linestyle="--")
    ax.set_title("Central Trace")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Lag Error
    ax = axes[2]
    corr = np.correlate(ref.trace_series, bfp.trace_series, mode='full')
    lag = corr.argmax() - (len(ref.trace_series) - 1)
    
    bfp_shifted = np.roll(bfp.trace_series, lag)
    diff = np.abs(ref.trace_series - bfp_shifted)
    if lag > 0: diff[:lag] = 0
    elif lag < 0: diff[lag:] = 0
        
    ax.plot(ref.time_series, diff, color="red", label=f"Error (Lag={lag})")
    ax.set_title("Phase Lag Error")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Field Diff
    ax = axes[3]
    im = ax.imshow(np.abs(ref.final_fields - bfp.final_fields), cmap='inferno', origin='lower')
    ax.set_title("Final State Difference")
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig("comparison_metrics.png")
    print("\nDone. Saved 'comparison_metrics.png'")

if __name__ == "__main__":
    compare_runs()