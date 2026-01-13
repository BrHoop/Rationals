from __future__ import annotations
import os
import sys
import tomllib
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
# Adjust imports based on your structure
try:
    import utils.ioxdmf as iox
except ImportError:
    iox = None
import bfp_ops
from bfp_core import BFPGrid

jax.config.update("jax_enable_x64", True)
PROJECT_ROOT = Path.cwd()

def load_params(parfile: Path) -> dict:
    with open(parfile, "rb") as f: return tomllib.load(f)

def parse_grid_params(params: dict):
    nx, ny = params.get("Nx", 512), params.get("Ny", 512)
    nt, out_int = params.get("Nt", 1000), params.get("output_interval", 10)
    cfl = params.get("cfl", 0.05)
    return nx, ny, nt, out_int, cfl

def parse_domain(params: dict):
    return params.get("Xmin", -5.0), params.get("Xmax", 5.0), params.get("Ymin", -5.0), params.get("Ymax", 5.0)

def initialize_grid(nx, ny, xmin, xmax, ymin, ymax, params):
    dx = (xmax - xmin) / (nx - 1)
    # Enforce square grid assumption for 4th order stencil simplicity
    dt = params.get("cfl", 0.05) * dx
    x_coords = jnp.linspace(xmin, xmax, nx)
    y_coords = jnp.linspace(ymin, ymax, ny)
    X, Y = jnp.meshgrid(x_coords, y_coords, indexing="ij")
    
    x0, y0 = params.get("id_x0", 0.0), params.get("id_y0", 0.0)
    sigma, amp = params.get("id_sigma", 0.1), params.get("id_amp", 1.0)
    phi0 = amp * jnp.exp(-((X - x0)**2 + (Y - y0)**2) / (sigma**2))
    
    # 50-bit Grid for high precision
    u_init = jnp.stack([phi0, jnp.zeros_like(phi0)], axis=0)
    grid = BFPGrid(nx, ny, num_fields=2, mantissa_bits=50).to_device(u_init)
    return grid, x_coords, y_coords, dx, dt

def make_stepper(dt, dx, out_interval, grid, params):
    # 1. Physics Scalars
    # 4th Order Lap Factor: 1 / (12 * dx^2)
    lap_factor = 1.0 / (12.0 * dx**2)
    s_lap = grid.make_scalar(lap_factor, target=50)
    
    # KO Dissipation Factor: sigma / 16
    sigma_ko = params.get("ko_sigma", 0.05)
    diss_factor = sigma_ko / 16.0
    s_diss = grid.make_scalar(diss_factor, target=50)
    
    # Time Step
    s_dt = grid.make_scalar(dt, target=50)

    @jax.jit
    def step_rk4(grid_in: BFPGrid) -> BFPGrid:
        def body(g, _):
            m, e = g.mantissas, g.exponents
            # Run Fused RK4
            # Returns (m_full, e_full) - unblocked
            m_new, e_new = bfp_ops.fused_rk4_step(m, e, s_dt, s_lap, s_diss)
            
            # Normalize (Re-block) only once per step
            # This is the cost saving: 4 sub-steps, 1 normalization
            g_temp = g.replace(mantissas=m_new, exponents=e_new)
            return bfp_ops.normalize_grid(g_temp), None
            
        final_grid, _ = jax.lax.scan(body, grid_in, None, length=out_interval)
        return final_grid
        
    return step_rk4

def main(parfile: Path, output_dir: Path):
    if not parfile.exists():
        print(f"Error: {parfile} not found.")
        return

    params = load_params(parfile)
    nx, ny, nt, out_int, _ = parse_grid_params(params)
    xmin, xmax, ymin, ymax = parse_domain(params)
    
    grid, xc, yc, dx, dt = initialize_grid(nx, ny, xmin, xmax, ymin, ymax, params)
    stepper = make_stepper(dt, dx, out_int, grid, params)
    
    func_names = ["phi", "chi"]
    x_np, y_np = np.asarray(xc), np.asarray(yc)
    
    # Initial Output
    if iox:
        u_float = grid.from_device()
        iox.write_hdf5(0, np.asarray(u_float), x_np, y_np, func_names, output_dir)
    
    print(f"Starting BFP RK4 Solver | {nx}x{ny} | 4th Order | 50-bit")
    
    for step in range(out_int, nt + 1, out_int):
        grid = stepper(grid)
        
        # Monitor
        u_float = grid.from_device()
        norm = jnp.linalg.norm(u_float[0])
        print(f"Step {step:05d} | t={step*dt:.4f} | Phi Norm: {norm:.6e}")
        
        if iox:
            iox.write_hdf5(step, np.asarray(u_float), x_np, y_np, func_names, output_dir)
            
    if iox:
        iox.write_xdmf(output_dir, nt, nx, ny, func_names, out_int, dt)

if __name__ == "__main__":
    par = Path(sys.argv[1]) if len(sys.argv)>1 else PROJECT_ROOT / "params.toml"
    out = Path(sys.argv[2]) if len(sys.argv)>2 else PROJECT_ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    main(par, out)