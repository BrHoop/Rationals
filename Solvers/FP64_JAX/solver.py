#!/usr/bin/env python3
# solver_jax.py
import sys
import time
import tomllib
import jax
import jax.numpy as jnp
from jax import lax

# Enable float64 (handy for PDEs)
jax.config.update("jax_enable_x64", True)

# =========================
# Grid & utilities (host)
# =========================
def create_grid(params: dict):
    x_min, x_max, nx = params["x_min"], params["x_max"], params["nx"]
    if nx < 2:
        raise ValueError("nx must be >= 2")
    dx = (x_max - x_min) / (nx - 1)
    x = jnp.linspace(x_min, x_max, nx, dtype=jnp.float64)
    return x, dx

def write_curve(filename: str, time_val: float, x: jnp.ndarray, u_names, state):
    # state is a tuple (phi, pi)
    phi, pi = state
    # Move to host for I/O
    x_h   = jax.device_get(x)
    phi_h = jax.device_get(phi)
    pi_h  = jax.device_get(pi)

    with open(filename, "w") as f:
        f.write(f"# TIME {time_val}\n")
        for name, arr in zip(u_names, (phi_h, pi_h)):
            f.write(f"# {name}\n")
            for xi, ui in zip(x_h, arr):
                f.write(f"{xi:.8e} {ui:.8e}\n")

def l2norm(u: jnp.ndarray) -> float:
    # host-side pretty print; keep computation on device
    return float(jnp.sqrt(jnp.mean(u**2)))

# =========================
# Physics kernels (device)
# =========================
@jax.jit
def initial_data(x: jnp.ndarray, x0: float, amp: float, omega: float):
    """Return (phi, pi) as a tuple."""
    phi = amp * jnp.exp(-omega * (x - x0) ** 2)
    pi  = jnp.zeros_like(x)
    return (phi, pi)

@jax.jit
def grad(u: jnp.ndarray, dx: float):
    """4th-order centered first derivative with periodic BC via roll."""
    idx12 = 1.0 / (12.0 * dx)
    up1 = jnp.roll(u, -1)
    up2 = jnp.roll(u, -2)
    um1 = jnp.roll(u,  1)
    um2 = jnp.roll(u,  2)
    return (-up2 + 8.0*up1 - 8.0*um1 + um2) * idx12

@jax.jit
def rhs(state, dx: float):
    """Your current symmetric first-derivative system:
       phi_t = d/dx(pi),  pi_t = d/dx(phi).
       Returns (dphi_dt, dpi_dt).
    """
    phi, pi = state
    dphi = grad(pi,  dx)
    dpi  = grad(phi, dx)
    return (dphi, dpi)

@jax.jit
def rk2_step(state, dx: float, dt: float):
    """Heun / midpoint RK2 step on device."""
    k1 = rhs(state, dx)
    mid = (state[0] + 0.5*dt*k1[0], state[1] + 0.5*dt*k1[1])
    k2 = rhs(mid, dx)
    return (state[0] + dt*k2[0], state[1] + dt*k2[1])


def advance_n_steps(state, dx, dt, nsteps):
    def body(s, _):
        s = rk2_step(s, dx, dt)
        return s, None
    s_final, _ = jax.lax.scan(body, state, xs=None, length=nsteps)
    return s_final
advance_n_steps = jax.jit(advance_n_steps, static_argnames=("nsteps",))
# =========================
# Main driver (host)
# =========================
def main(parfile: str, output_path: str, trace_dir: str | None = None):
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    # Grid and timestep
    x, dx = create_grid(params)
    dt    = float(params["cfl"]) * float(dx)
    nt    = int(params["nt"])
    freq  = int(params.get("output_frequency", 1))
    if freq <= 0:
        freq = nt  # avoid division by zero, write only final

    # Initial data parameters (extract scalars for JIT-friendliness)
    x0    = float(params.get("id_x0",   0.5))
    amp   = float(params.get("id_amp",  1.0))
    omega = float(params.get("id_omega", 1.0))

    # Initial state (phi, pi)
    state = initial_data(x, x0, amp, omega)
    u_names = ["Phi", "Pi"]

    # Output t=0
    time_val = 0.0
    write_curve(f"{output_path}data_0000.curve", time_val, x, u_names, state)

    # Chunked integration: do 'freq' steps per JIT call, emit data, repeat.
    # This gives big kernels and far fewer launches, boosting GPU utilization.
    steps_done = 0
    write_index = 0

    print(f"Starting on backend={jax.default_backend()}, dx={dx:.3e}, dt={dt:.3e}, nt={nt}, freq={freq}")

    t0 = time.time()

    while steps_done < nt:
        nblock = min(freq, nt - steps_done)
        state = advance_n_steps(state, dx, dt, nblock)
        steps_done += nblock
        time_val += nblock * dt

        # Force completion before measuring/printing
        jax.block_until_ready(state[0])

        write_index = steps_done
        phi_norm = l2norm(state[0])
        pi_norm  = l2norm(state[1])
        print(f"Step {steps_done:6d}, t={time_val:.6e}, |Phi|={phi_norm:.3e}, |Pi|={pi_norm:.3e}")

        fname = f"{output_path}data_{write_index:04d}.curve"
        write_curve(fname, time_val, x, u_names, state)

    jax.block_until_ready(state[0])
    print(f"Done in {time.time() - t0:.3f} s")

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python solver_jax.py <parfile> <output_path> [trace_dir]")
        sys.exit(1)

    parfile     = sys.argv[1]
    output_path = sys.argv[2]
    trace_dir   = sys.argv[3] if len(sys.argv) == 4 else None

    if trace_dir:
        with jax.profiler.trace(trace_dir):
            main(parfile, output_path, trace_dir)
    else:
        main(parfile, output_path)