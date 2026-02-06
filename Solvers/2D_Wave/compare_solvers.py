import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import time
import jax
import jax.numpy as jnp

def load_module_from_path(module_name, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find solver file: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def get_solver_class(module, possible_names):
    for name in possible_names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"Could not find class in {module}")

def main():
    # 1. SETUP
    ref_path = os.path.join("Solvers", "2D_Wave", "FP_64_Jax", "Jax_Solver.py")
    ozaki_path = os.path.join("Solvers", "2D_Wave", "Ozaki_Scheme_II", "ozaki_solver.py")
    
    RefModule = load_module_from_path("ref_solver", ref_path)
    OzakiModule = load_module_from_path("ozaki_solver", ozaki_path)
    
    RefClass = get_solver_class(RefModule, ["NonlinearSigmaSolver", "LinearWaveSolver"])
    OzakiClass = get_solver_class(OzakiModule, ["NonlinearOzakiSolver"])

    os.makedirs("comparison_plots", exist_ok=True)

    # 2. PARAMETERS
    params = {
        "Nx": 256, "Ny": 256, 
        "Nt": 2500,
        "output_interval": 10,
        "Xmin": -10.0, "Xmax": 10.0,
        "Ymin": -10.0, "Ymax": 10.0,
        "cfl": 0.1,
        "ko_sigma": 0.75,
        "id_x0": 0.0, "id_y0": 0.0,
        "id_sigma": 0.5, "id_amp": 1.0,
    }

    # 3. INITIALIZATION
    sim_ref = RefClass(params['Nx'], params['Ny'], params)
    sim_ref.initialize()
    step_ref = sim_ref.get_stepper(1)

    sim_ozaki = OzakiClass(params['Nx'], params['Ny'], params)
    sim_ozaki.initialize()
    step_ozaki = sim_ozaki.get_stepper(1)

    # --- SYNC ---
    u_start = jnp.array(sim_ref.u)
    sim_ozaki.u = u_start
    
    # --- PREPARE STATES (Both are just U now) ---
    state_ref = sim_ref.u
    state_ozaki = sim_ozaki.u  # <--- No tuple, just the array

    history_time = []
    history_l2_error = []

    print(f"Comparing Solvers | Nx={params['Nx']} | Nt={params['Nt']}")
    
    # Warmup
    print("Compiling...")
    _ = step_ref(state_ref)
    _ = step_ozaki(state_ozaki)
    print("Running Comparison...")

    start_time = time.time()

    # 4. LOOP
    for s in range(1, params['Nt'] + 1):
        # Step Both (Simple input -> Simple output)
        state_ref = step_ref(state_ref)
        state_ozaki = step_ozaki(state_ozaki)
        
        if s % params["output_interval"] == 0:
            # --- FIX: NO UNPACKING NEEDED ---
            u_ref_curr = state_ref
            u_ozaki_curr = state_ozaki  # The state IS the solution 'u'
            
            u_ozaki_curr.block_until_ready()
            
            u_oz_np = np.array(u_ozaki_curr)
            u_ref_np = np.array(u_ref_curr)
            
            # Metric: Relative Error of Field 0 (Chi)
            # Now u_oz_np is shape (2, Nx, Ny), so [0] works correctly
            diff = u_oz_np[0] - u_ref_np[0]
            ref_norm = np.linalg.norm(u_ref_np[0])
            err_norm = np.linalg.norm(diff)
            
            rel_error = err_norm / (ref_norm + 1e-30)
            
            t = s * sim_ref.dt
            history_time.append(t)
            history_l2_error.append(rel_error)
            
            print(f"Step {s:4d} | Rel Error: {rel_error:.4e}")

    # 5. PLOT
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].semilogy(history_time, history_l2_error, 'r-', label='Rel L2 Error')
    ax[0].set_title("Stochastic Error Feedback Stability")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Relative Error")
    ax[0].grid(True, linestyle='--')

    final_diff = np.abs(u_oz_np[0] - u_ref_np[0])
    im = ax[1].imshow(final_diff.T, origin='lower', cmap='inferno',
                      extent=[params['Xmin'], params['Xmax'], params['Ymin'], params['Ymax']])
    ax[1].set_title(f"Final Error Map (Max: {np.max(final_diff):.2e})")
    plt.colorbar(im, ax=ax[1])
    
    plt.savefig(os.path.join("comparison_plots", "feedback_test.png"))
    plt.show()

if __name__ == "__main__":
    main()