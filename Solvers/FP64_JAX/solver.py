import jax
import jax.numpy as jnp
from jax import jit
import tomllib
import sys
import tensorflow as tf
jax.config.update("jax_enable_x64", True)

def create_grid(params):
    x_min = params["x_min"]
    x_max = params["x_max"]
    nx = params["nx"]
    if nx < 2:
        raise ValueError("nx must be >= 2 for grid spacing computation")
    dx = (x_max - x_min) / (nx -1)
    x = jnp.linspace(x_min, x_max, nx, dtype=jnp.float64)
    return x, dx

@jax.jit
def initial_data(u, x, params):
    x0 = params.get("id_x0", 0.5)
    amp = params.get("id_amp", 1.0)
    omega = params.get("id_omega", 1.0)

    phi = amp * jnp.exp(-omega * (x - x0) ** 2)
    pi = jnp.zeros_like(x)
    return [phi, pi]

@jax.jit
def grad(u, dx):
    idx_by_12 = 1.0 / (12 * dx)
    
    up1 = jnp.roll(u, -1)

    up2 = jnp.roll(u, -2)

    um1 = jnp.roll(u, 1)

    um2 = jnp.roll(u, 2)

    return (-up2 + 8 * up1 - 8 * um1 + um2) * idx_by_12

@jax.jit
def rhs(u, x):
    dx = x[1] - x[0]
    phi, pi = u
    dx_phi = grad(phi, dx)
    dx_pi = grad(pi, dx)
    return [dx_pi, dx_phi]
    
@jax.jit
def rk2(u, x, dt):
    nu = len(u)

    up = []
    k1 = []
    k2 = []
    for i in range(nu):
        up.append(jnp.empty_like(u[0], dtype=jnp.float64))
        k1.append(jnp.empty_like(u[0], dtype=jnp.float64))
        k2.append(jnp.empty_like(u[0], dtype=jnp.float64))

    k1 = rhs(u, x)
    up = [u[i] + k1[i] * dt * 0.5 for i in range(len(u))]

    k2 = rhs(up, x)
    u_new = [u[i] + k2[i] * dt for i in range(len(u))]
    return u_new

def write_curve(filename, time, x, u_names, u):
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        for m in range(len(u_names)):
            f.write(f"# {u_names[m]}\n")
            for xi, di in zip(x, u[m]):
                f.write(f"{xi:.8e} {di:.8e}\n")

@jax.jit
def l2norm(u):
    """
    Compute the L2 norm of an array.
    """
    return jnp.sqrt(jnp.mean(u**2))


def main(parfile, output_path):
    # Read parameters
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    # Create the grid and set time step size
    x, dx = create_grid(params)
    dt = params["cfl"] * dx

    # Allocating memory
    Phi = jnp.empty_like(x)
    Pi = jnp.empty_like(x)
    u = [Phi, Pi]
    u_names = ["Phi", "Pi"]

    u = initial_data(u, x, params)

    nt = params["nt"]
    time = 0.0

    iter = 0
    fname = f"{output_path}data_{iter:04d}.curve"
    write_curve(fname, time, x, u_names, u)

    freq = params.get("output_frequency", 1)

    # Inegrate in time
    for i in range(1, nt+1):
        u = rk2(u, x, dt)
        time += dt
        if i % freq == 0:
            print(f"Step {i:d}, t={time:.2e}, |Phi|={l2norm(u[0]):.2e}, |Pi|={l2norm(u[1]):.2e}")
            fname = f"{output_path}data_{i:04d}.curve"
            write_curve(fname, time, x, u_names, u)

    


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:  python solver.py <parfile> <output_path>")
        sys.exit(1)

    parfile = sys.argv[1]
    output_path = sys.argv[2]
    with jax.profiler.trace('/Users/isaacsudweeks/Library/CloudStorage/OneDrive-BrighamYoungUniversity/Personal Projects/Rationals/profiles'):
        main(parfile, output_path)
