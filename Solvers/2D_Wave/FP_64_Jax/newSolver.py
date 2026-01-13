import os
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

def laplacian_4th_order(u, dx2):
    """
    4th-Order Centered Finite Difference Laplacian.
    Coefficients: [-1, 16, -30, 16, -1]
    Divisor: 12 * dx^2
    """
    inv_12_dx2 = 1.0 / (12.0 * dx2)
    
    # Weights
    c0 = -30.0
    c1 = 16.0
    c2 = -1.0
    
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
    Adds 4th-order numerical dissipation (Kreiss-Oliger) to dampen high-freq noise.
    Q = -sigma * (dx^3) * (D4 u)  (Standard formula varies, usually scaling with dx)
    Using simple 4th derivative stencil: [1, -4, 6, -4, 1]
    """
    # 4th derivative in X
    d4x = (jnp.roll(u, -2, axis=0) - 4.0*jnp.roll(u, -1, axis=0) + 
           6.0*u - 
           4.0*jnp.roll(u, 1, axis=0) + jnp.roll(u, 2, axis=0))
           
    # 4th derivative in Y
    d4y = (jnp.roll(u, -2, axis=1) - 4.0*jnp.roll(u, -1, axis=1) + 
           6.0*u - 
           4.0*jnp.roll(u, 1, axis=1) + jnp.roll(u, 2, axis=1))
           
    # Scaling factor: sigma * dx / 2^order_dissipation approx
    # Standard KO form for 2nd order systems often scales with dx
    return -sigma * (d4x + d4y) / 16.0 

def apply_boundaries(u):
    """Enforce Dirichlet boundaries (u=0) at edges."""
    u = u.at[:, 0:2, :].set(0.0)
    u = u.at[:, -2:, :].set(0.0)
    u = u.at[:, :, 0:2].set(0.0)
    u = u.at[:, :, -2:].set(0.0)
    return u

# --- Solver Class ---

class LinearWaveSolver:
    def __init__(self, nx, ny, params):
        xmin, xmax = params.get("Xmin", -5.0), params.get("Xmax", 5.0)
        ymin, ymax = params.get("Ymin", -5.0), params.get("Ymax", 5.0)
        
        self.dx = (xmax - xmin) / (nx - 1)
        # Assuming dy approx dx for square grid logic, or enforce aspect ratio
        self.dt = params["cfl"] * self.dx
        self.sigma = params.get("ko_sigma", 0.05)
        
        x = jnp.linspace(xmin, xmax, nx)
        y = jnp.linspace(ymin, ymax, ny)
        self.X, self.Y = jnp.meshgrid(x, y, indexing="ij")
        self.params = params
        
        # State vector u = [phi, chi]
        # phi_t = chi
        # chi_t = Laplacian(phi)
        self.u = jnp.zeros((2, nx, ny), dtype=jnp.float64)

    def initialize(self):
        x0, y0 = self.params.get("id_x0", 0.0), self.params.get("id_y0", 0.0)
        sigma, amp = self.params.get("id_sigma", 0.1), self.params.get("id_amp", 1.0)
        
        # Gaussian pulse
        r2 = (self.X - x0)**2 + (self.Y - y0)**2
        phi0 = amp * jnp.exp(-r2 / (sigma**2))
        
        # Initialize chi (time derivative) to 0 or derived if needed
        chi0 = jnp.zeros_like(phi0)
        
        self.u = jnp.stack([phi0, chi0])

    def get_stepper(self, steps_per_call):
        dt, dx, sigma = self.dt, self.dx, self.sigma
        dx2 = dx**2  # <--- Pre-calculate dx^2
        
        def equations_of_motion(u):
            phi, chi = u[0], u[1]
            
            # 1. Physical derivatives
            # PASS dx2 HERE
            lap_phi = laplacian_4th_order(phi, dx2)
            
            # 2. Artificial Dissipation (Kreiss-Oliger) applied to evolved vars
            diss_phi = kreiss_oliger_dissipation(phi, dx, sigma)
            diss_chi = kreiss_oliger_dissipation(chi, dx, sigma)
            
            # 3. RHS Calculation
            # dt_phi = chi + dissipation
            # dt_chi = laplacian(phi) + dissipation
            dt_phi = chi + diss_phi
            dt_chi = lap_phi + diss_chi
            
            return jnp.stack([dt_phi, dt_chi])

        def rk4_step(u, _):
            # Standard Runge-Kutta 4 integration
            k1 = equations_of_motion(u)
            k2 = equations_of_motion(u + 0.5 * dt * k1)
            k3 = equations_of_motion(u + 0.5 * dt * k2)
            k4 = equations_of_motion(u + dt * k3)
            
            u_next = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            return apply_boundaries(u_next), None

        @jax.jit
        def step_block(u_current):
            # Run 'steps_per_call' RK4 steps using lax.scan for speed
            final_u, _ = lax.scan(rk4_step, u_current, None, length=steps_per_call)
            return final_u
            
        return step_block

# --- Runner ---

def main(parfile, output_dir):
    if not os.path.exists(parfile):
        print(f"Error: Parameter file {parfile} not found.")
        return

    with open(parfile, "rb") as f: 
        params = tomllib.load(f)
        
    nx, ny = params["Nx"], params["Ny"]
    nt, out_int = params["Nt"], params["output_interval"]
    
    sim = LinearWaveSolver(nx, ny, params)
    sim.initialize()
    
    step_fn = sim.get_stepper(out_int)
    names = ["phi", "chi"]
    
    # Save Initial State
    if iox:
        iox.write_hdf5(0, np.asarray(sim.u), np.array(sim.X[:,0]), np.array(sim.Y[0,:]), names, output_dir)
    
    print(f"Starting Simulation | Nx={nx} Ny={ny} Nt={nt} | Method=RK4 + 4th Order FD")
    
    for s in range(out_int, nt + 1, out_int):
        sim.u = step_fn(sim.u)
        
        # Compute Energy for monitoring (simplified)
        u_np = np.array(sim.u)
        phi_norm = np.linalg.norm(u_np[0])
        print(f"Step {s} | Time: {s*sim.dt:.3f} | L2 Norm: {phi_norm:.6e}")
        
        if iox:
            iox.write_hdf5(s, u_np, np.array(sim.X[:,0]), np.array(sim.Y[0,:]), names, output_dir)
            
    if iox:
        iox.write_xdmf(output_dir, nt, nx, ny, names, out_int, sim.dt)

if __name__ == "__main__":
    par = sys.argv[1] if len(sys.argv) > 1 else "params.toml"
    out = sys.argv[2] if len(sys.argv) > 2 else "data"
    os.makedirs(out, exist_ok=True)
    main(par, out)