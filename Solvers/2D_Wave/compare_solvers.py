import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import time

def load_module_from_path(module_name, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find solver file: {file_path}")
       
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def main():
    # ==========================================================================
    # 1. SETUP
    # ==========================================================================
    ref_path = os.path.join("Solvers","2D_Wave","FP_64_Jax", "newSolver.py")
    ozaki_path = os.path.join("Solvers","2D_Wave","Ozaki_Scheme_II", "ozaki_solver.py")
   
    print(f"Loading Reference Solver from: {ref_path}")
    RefModule = load_module_from_path("ref_solver", ref_path)
   
    print(f"Loading Ozaki Solver from: {ozaki_path}")
    OzakiModule = load_module_from_path("ozaki_solver", ozaki_path)

    os.makedirs("comparison_plots", exist_ok=True)

    # ==========================================================================
    # 2. PARAMETERS
    # ==========================================================================
    params = {
        "Nx": 128, "Ny": 128,
        "Nt": 1500,
        "output_interval": 10,
        "Xmin": -5.0, "Xmax": 5.0,
        "Ymin": -5.0, "Ymax": 5.0,
        "cfl": 0.1,
        "ko_sigma": 0.75,
        "id_x0": 0.0, "id_y0": 0.0,
        "id_sigma": 0.5, "id_amp": 1.0
    }

    print("-" * 60)
    print(f"Comparing Solvers: L2 Error & Spatial Difference")
    print("-" * 60)

    # ==========================================================================
    # 3. INITIALIZATION
    # ==========================================================================
    sim_ref = RefModule.LinearWaveSolver(params['Nx'], params['Ny'], params)
    sim_ref.initialize()
    step_ref = sim_ref.get_stepper(1)

    sim_ozaki = OzakiModule.LinearWaveSolver(params['Nx'], params['Ny'], params)
    sim_ozaki.initialize()
    step_ozaki = sim_ozaki.get_stepper(1)

    # Sync Start
    u_ref_start = np.array(sim_ref.u)
    sim_ozaki.u = u_ref_start.copy()

    history_time = []
    history_l2_error = []

    start_time = time.time()

    # ==========================================================================
    # 4. TIME LOOP
    # ==========================================================================
    for s in range(1, params['Nt'] + 1):
        sim_ref.u = step_ref(sim_ref.u)
        sim_ozaki.u = step_ozaki(sim_ozaki.u)
       
        u_ref_np = np.array(sim_ref.u)
        u_ozaki_np = sim_ozaki.u
       
        # --- Metrics ---
        # Relative L2 Error
        diff = u_ozaki_np[0] - u_ref_np[0]
        ref_norm = np.linalg.norm(u_ref_np[0])
        err_norm = np.linalg.norm(diff)
       
        rel_error = err_norm / (ref_norm + 1e-30)
       
        history_time.append(s * sim_ref.dt)
        history_l2_error.append(rel_error)
       
        if s % params["output_interval"] == 0:
            print(f"Step {s:4d} | Rel L2 Error: {rel_error:.4e}")

    total_time = time.time() - start_time
    print(f"\nComparison Complete in {total_time:.2f}s")

    # ==========================================================================
    # 5. PLOTTING
    # ==========================================================================
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: L2 Error Over Time ---
    ax[0].semilogy(history_time, history_l2_error, 'r-', linewidth=2, label='Relative L2 Error')
    ax[0].set_title("Relative Error Growth (L2 Norm)")
    ax[0].set_xlabel("Simulation Time")
    ax[0].set_ylabel("Error |u_oz - u_ref| / |u_ref|")
    ax[0].grid(True, which="both", linestyle='--', alpha=0.7)
    ax[0].legend()

    # --- Plot 2: Spatial Error Map (Final Step) ---
    # We plot the Absolute Difference |Ozaki - Ref|
    final_diff = np.abs(u_ozaki_np[0] - u_ref_np[0])
    max_diff = np.max(final_diff)
   
    im = ax[1].imshow(final_diff, origin='lower', cmap='inferno',
                      extent=[params['Xmin'], params['Xmax'], params['Ymin'], params['Ymax']])
    ax[1].set_title(f"Spatial Error Map (Step {params['Nt']})\nMax Diff: {max_diff:.2e}")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
   
    cbar = plt.colorbar(im, ax=ax[1])
    cbar.set_label("Absolute Difference")
    plt.tight_layout()
    save_path = os.path.join("comparison_plots", "l2_error_analysis.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nGraphs saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()