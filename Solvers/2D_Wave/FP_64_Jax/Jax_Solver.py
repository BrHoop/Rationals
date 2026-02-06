import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import tomllib
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

# 1. Enable Double Precision
jax.config.update("jax_enable_x64", True)

# Add utils path if necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
try:
    import utils.ioxdmf as iox
except ImportError:
    print("Warning: utils.ioxdmf not found. File output will be skipped.")
    iox = None

# --- Numerical Kernels ---

def apply_sommerfeld_rhs(dt_u, u, X, Y, dx):
    """
    Overwrites the Time Derivative (dt_u) at the boundaries to enforce
    Spherical Sommerfeld Radiation conditions.
    
    dt_u = - (x/r * dx_u + y/r * dy_u) - u/r
    """
    # 1. Geometry (2D Arrays)
    r = jnp.sqrt(X**2 + Y**2)
    inv_r = jnp.where(r < 1e-12, 0.0, 1.0/r)
    
    # Normal vectors (Direction of the 'outgoing' wave)
    norm_x = X * inv_r  # Shape: (Nx, Ny)
    norm_y = Y * inv_r  # Shape: (Nx, Ny)
    
    # 2. One-sided Gradients for the Boundaries
    #    We compute these generally, then slice specific edges.
    
    # --- LEFT BOUNDARY (x=0) ---
    # Slice u: (2, Ny)
    u_left = u[:, 0, :]
    
    # Gradients at left edge (Forward difference for x)
    dx_u_left = (u[:, 1, :] - u[:, 0, :]) / dx
    dy_u_left = (jnp.roll(u, -1, axis=2)[:, 0, :] - jnp.roll(u, 1, axis=2)[:, 0, :]) / (2*dx)
    
    # Slicing Geometry: Use [0, :] for 2D arrays
    # Broadcast: (Ny,) * (2, Ny) -> (2, Ny)
    rhs_left = - (norm_x[0, :] * dx_u_left + norm_y[0, :] * dy_u_left) - u_left * inv_r[0, :]
    dt_u = dt_u.at[:, 0, :].set(rhs_left)

    # --- RIGHT BOUNDARY (x=end) ---
    u_right = u[:, -1, :]
    
    # Backward difference for x
    dx_u_right = (u[:, -1, :] - u[:, -2, :]) / dx
    dy_u_right = (jnp.roll(u, -1, axis=2)[:, -1, :] - jnp.roll(u, 1, axis=2)[:, -1, :]) / (2*dx)
    
    rhs_right = - (norm_x[-1, :] * dx_u_right + norm_y[-1, :] * dy_u_right) - u_right * inv_r[-1, :]
    dt_u = dt_u.at[:, -1, :].set(rhs_right)

    # --- BOTTOM BOUNDARY (y=0) ---
    u_bot = u[:, :, 0]
    
    # Forward difference for y
    dy_u_bot = (u[:, :, 1] - u[:, :, 0]) / dx
    dx_u_bot = (jnp.roll(u, -1, axis=1)[:, :, 0] - jnp.roll(u, 1, axis=1)[:, :, 0]) / (2*dx)
    
    rhs_bot = - (norm_x[:, 0] * dx_u_bot + norm_y[:, 0] * dy_u_bot) - u_bot * inv_r[:, 0]
    dt_u = dt_u.at[:, :, 0].set(rhs_bot)

    # --- TOP BOUNDARY (y=end) ---
    u_top = u[:, :, -1]
    
    # Backward difference for y
    dy_u_top = (u[:, :, -1] - u[:, :, -2]) / dx
    dx_u_top = (jnp.roll(u, -1, axis=1)[:, :, -1] - jnp.roll(u, 1, axis=1)[:, :, -1]) / (2*dx)
    
    rhs_top = - (norm_x[:, -1] * dx_u_top + norm_y[:, -1] * dy_u_top) - u_top * inv_r[:, -1]
    dt_u = dt_u.at[:, :, -1].set(rhs_top)

    return dt_u

def laplacian_4th_order(u, dx2):
    """
    4th-Order Centered Finite Difference Laplacian.
    Coefficients: [-1, 16, -30, 16, -1] / 12dx^2
    """
    inv_12_dx2 = 1.0 / (12.0 * dx2)
    c0, c1, c2 = -30.0, 16.0, -1.0
    
    # X-direction
    d2x = (c2 * (jnp.roll(u, -2, axis=0) + jnp.roll(u, 2, axis=0)) +
           c1 * (jnp.roll(u, -1, axis=0) + jnp.roll(u, 1, axis=0)) +
           c0 * u)
            
    # Y-direction
    d2y = (c2 * (jnp.roll(u, -2, axis=1) + jnp.roll(u, 2, axis=1)) +
           c1 * (jnp.roll(u, -1, axis=1) + jnp.roll(u, 1, axis=1)) +
           c0 * u)
            
    return (d2x + d2y) * inv_12_dx2

def kreiss_oliger_dissipation(u, dx, sigma):
    """
    Adds 4th-order numerical dissipation (Kreiss-Oliger).
    """
    # 4th derivative in X
    d4x = (jnp.roll(u, -2, axis=0) - 4.0*jnp.roll(u, -1, axis=0) + 
           6.0*u - 4.0*jnp.roll(u, 1, axis=0) + jnp.roll(u, 2, axis=0))
            
    # 4th derivative in Y
    d4y = (jnp.roll(u, -2, axis=1) - 4.0*jnp.roll(u, -1, axis=1) + 
           6.0*u - 4.0*jnp.roll(u, 1, axis=1) + jnp.roll(u, 2, axis=1))
            
    return -sigma * (d4x + d4y) / 16.0 

def apply_boundaries(u):
    """Enforce Dirichlet boundaries (u=0) at edges."""
    u = u.at[:, 0:2, :].set(0.0)
    u = u.at[:, -2:, :].set(0.0)
    u = u.at[:, :, 0:2].set(0.0)
    u = u.at[:, :, -2:].set(0.0)
    return u

def equations_of_motion(u, dx2, r2_inv, sigma, dx, X, Y):
    chi, pi = u[0], u[1]
    
    # 1. Laplacian
    lap_chi = laplacian_4th_order(chi, dx2)
    
    # 2. NLSM Force Term
    nonlinear_term = -jnp.sin(2.0 * chi) * r2_inv
    
    # 3. Dissipation
    diss_chi = kreiss_oliger_dissipation(chi, dx, sigma)
    diss_pi = kreiss_oliger_dissipation(pi, dx, sigma)
    
    # 4. Standard Interior Evolution
    dt_chi = pi + diss_chi
    dt_pi = lap_chi + nonlinear_term + diss_pi
    
    # Pack them momentarily
    dt_u = jnp.stack([dt_chi, dt_pi])
    
    # 5. --- APPLY SOMMERFELD BOUNDARIES ---
    # This overwrites the edges of dt_u with the radiative condition
    dt_u = apply_sommerfeld_rhs(dt_u, u, X, Y, dx)
    
    return dt_u

# --- Solver Class ---

class NonlinearSigmaSolver:
    def __init__(self, nx, ny, params):
        # 1. Bounds & Grid
        xmin, xmax = params.get("Xmin", -5.0), params.get("Xmax", 5.0)
        ymin, ymax = params.get("Ymin", -5.0), params.get("Ymax", 5.0)
        
        self.dx = (xmax - xmin) / (nx - 1)
        self.dt = params["cfl"] * self.dx
        self.sigma = params.get("ko_sigma", 0.05)
        
        x = jnp.linspace(xmin, xmax, nx)
        y = jnp.linspace(ymin, ymax, ny)
        
        # 2. Meshgrid & Radial Pre-calculation
        self.X, self.Y = jnp.meshgrid(x, y, indexing="ij")
        self.params = params
        
        # r^2 calculation with singularity masking
        r2 = self.X**2 + self.Y**2
        self.r2_inv = jnp.where(r2 < 1e-12, 0.0, 1.0 / r2)
        
        # 3. State Vector u = [chi, pi]
        self.u = jnp.zeros((2, nx, ny), dtype=jnp.float64)

    def initialize(self):
        """Sets the initial Gaussian pulse for the field chi."""
        x0, y0 = self.params.get("id_x0", 0.0), self.params.get("id_y0", 0.0)
        sigma, amp = self.params.get("id_sigma", 1.0), self.params.get("id_amp", 0.5)
        
        # Gaussian pulse in Chi
        r2_loc = (self.X - x0)**2 + (self.Y - y0)**2
        chi0 = amp * jnp.exp(-r2_loc / (sigma**2))
        
        # Initial velocity (Pi) is zero (time-symmetric data)
        pi0 = jnp.zeros_like(chi0)
        
        self.u = jnp.stack([chi0, pi0])

    def get_stepper(self, steps_per_call):
        dt, dx, sigma = self.dt, self.dx, self.sigma
        dx2 = dx**2
        r2_inv = self.r2_inv
        X, Y = self.X, self.Y  # Capture grid coordinates

        def rk4_step(u, _):
            # Pass X, Y to EOM
            k1 = equations_of_motion(u, dx2, r2_inv, sigma, dx, X, Y)
            k2 = equations_of_motion(u + 0.5 * dt * k1, dx2, r2_inv, sigma, dx, X, Y)
            k3 = equations_of_motion(u + 0.5 * dt * k2, dx2, r2_inv, sigma, dx, X, Y)
            k4 = equations_of_motion(u + dt * k3, dx2, r2_inv, sigma, dx, X, Y)
            
            u_next = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # REMOVED: apply_boundaries(u_next) 
            # (Because the boundary logic is now inside EOM)
            
            return u_next, None

        @jax.jit
        def step_block(u_current):
            final_u, _ = lax.scan(rk4_step, u_current, None, length=steps_per_call)
            return final_u
            
        return step_block

# --- Runner ---

def main(parfile, output_dir):
    if not os.path.exists(parfile):
        print(f"Error: Parameter file {parfile} not found.")
        # Create a dummy params dict for testing if file missing
        params = {
            "Nx": 256, "Ny": 256, "Nt": 100, "output_interval": 10,
            "cfl": 0.25, "ko_sigma": 0.01,
            "id_amp": 0.8, "id_sigma": 1.5
        }
    else:
        with open(parfile, "rb") as f: 
            params = tomllib.load(f)
        
    nx, ny = params["Nx"], params["Ny"]
    nt, out_int = params["Nt"], params["output_interval"]
    
    sim = NonlinearSigmaSolver(nx, ny, params)
    sim.initialize()  # <--- This now works
    
    step_fn = sim.get_stepper(out_int)
    
    # Names adjusted for NLSM: Chi (field) and Pi (momentum)
    names = ["chi", "pi"]
    
    if iox:
        iox.write_hdf5(0, np.asarray(sim.u), np.array(sim.X[:,0]), np.array(sim.Y[0,:]), names, output_dir)
    
    print(f"Starting NLSM Simulation | Nx={nx} Ny={ny} Nt={nt}")
    
    for s in range(out_int, nt + 1, out_int):
        sim.u = step_fn(sim.u)
        
        # Monitor L2 norm of the field chi
        chi_norm = np.linalg.norm(sim.u[0])
        print(f"Step {s} | Time: {s*sim.dt:.3f} | |Chi|: {chi_norm:.6e}")
        
        if iox:
            iox.write_hdf5(s, np.asarray(sim.u), np.array(sim.X[:,0]), np.array(sim.Y[0,:]), names, output_dir)
            
    if iox:
        iox.write_xdmf(output_dir, nt, nx, ny, names, out_int, sim.dt)

if __name__ == "__main__":
    par = sys.argv[1] if len(sys.argv) > 1 else "params.toml"
    out = sys.argv[2] if len(sys.argv) > 2 else "data"
    os.makedirs(out, exist_ok=True)
    main(par, out)