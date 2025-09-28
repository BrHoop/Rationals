import sys
import tomllib
import jax
import jax.numpy as jnp
from jax import jit
jax.config.update("jax_enable_x64", True)

FRAC_BITS = 24
SCALE = 1 << FRAC_BITS
HALF = 1 << (FRAC_BITS - 1)
FIXED_ONE = SCALE

@jax.jit
def _as_int_array(value):
    return jnp.asarray(value, dtype=jnp.int64)

@jax.jit
def _to_result(value):
    arr = _as_int_array(value)
    if arr.shape == ():
        return int(arr)
    return arr.astype(jnp.int64)

@jax.jit
def to_fixed_scalar(value):
    return (round(value.astype(float) * SCALE)).astype(jnp.int64)

@jax.jit
def to_fixed_array(values):
    return jnp.rint(values * SCALE).astype(jnp.int64)

@jax.jit
def from_fixed_scalar(value):
    return value.astype(jnp.float64) / SCALE

@jax.jit
def from_fixed_array(values):
    return values.astype(jnp.float64) / SCALE

@jax.jit
def fixed_mul(a, b):
    a_arr = _as_int_array(a)
    b_arr = _as_int_array(b)
    result = (a_arr * b_arr + HALF) >> FRAC_BITS #The addition of the Half Scale unit makes it so that it rounds to the nearest reather than truncating
    if result.shape == ():
        return result.astype(jnp.int64)
    return result.astype(jnp.int64)

@jax.jit
def fixed_div(numerator, denominator):
    num = _as_int_array(numerator)
    den = _as_int_array(denominator)
    result = ((num << FRAC_BITS) + den // 2) // den
    if result.shape == ():
        return result.astype(jnp.int64)
    return result.astype(jnp.int64)

@jax.jit
def fixed_div_int(value, divisor):
    arr = _as_int_array(value)
    result = (arr + divisor // 2) // divisor
    if result.shape == ():
        return result.astype(jnp.int64)
    return result.astype(jnp.int64)


def create_grid(params):
    nx = params["nx"]
    if nx < 2:
        raise ValueError("nx must be >= 2 for grid spacing computation")

    x_min = to_fixed_scalar(params["x_min"])
    x_max = to_fixed_scalar(params["x_max"])
    dx = fixed_div_int(x_max - x_min, nx - 1)
    x = jnp.linspace(x_min, x_max, nx, dtype=jnp.float64)
    return x, dx

@jax.jit
def initial_data(u, x, params):
    x0 = params.get("id_x0", 0.5)
    amp = params.get("id_amp", 1.0)
    omega = params.get("id_omega", 1.0)

    x_float = from_fixed_array(x)
    profile = amp * jnp.exp(-omega * (x_float - x0) ** 2)
    u[0] = u[0].at[:].set(to_fixed_array(profile))
    u[1] = u[1].at[:].set(0)
    return u

@jax.jit
def grad(u, idx_by_12):
    up1 = jnp.empty_like(u, dtype=jnp.int64)
    up1 = jnp.roll(u, -1)

    up2 = jnp.empty_like(u, dtype=jnp.int64)
    up2 = jnp.roll(u, -2)

    um1 = jnp.empty_like(u, dtype=jnp.int64)
    um1 = jnp.roll(u, 1)

    um2 = jnp.empty_like(u, dtype=jnp.int64)
    um2 = jnp.roll(u, 2)

    stencil = -up2 + 8 * up1 - 8 * um1 + um2
    return fixed_mul(stencil, idx_by_12)

@jax.jit
def rhs(u, x):
    dx = x[1] - x[0]
    inv_12_dx = fixed_div(FIXED_ONE, dx * 12)
    phi, pi = u
    dx_phi = grad(phi, inv_12_dx)
    dx_pi = grad(pi, inv_12_dx)
    return [dx_pi, dx_phi]

@jax.jit
def rk2(u, x, dt):
    nu = len(u)
    half_dt = fixed_mul(dt, to_fixed_scalar(0.5))

    up = []
    k1 = []
    k2 = []
    for _ in range(nu):
        up.append(jnp.empty_like(u[0], dtype=jnp.int64))
        k1.append(jnp.empty_like(u[0], dtype=jnp.int64))
        k2.append(jnp.empty_like(u[0], dtype=jnp.int64))

    k1 = rhs(u, x)
    up = [u[i] + fixed_mul(k1[i], half_dt) for i in range(len(u))]
    
    k2 = rhs(up, x)
    u_new = [u[i] + fixed_mul(k2[i], dt) for i in range(len(u))]
    return u_new


def write_curve(filename, time, x, u_names, u):
    x_float = from_fixed_array(x)
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        for name, values in zip(u_names, u):
            f.write(f"# {name}\n")
            values_float = from_fixed_array(values)
            for xi, vi in zip(x_float, values_float):
                f.write(f"{xi:.8e} {vi:.8e}\n")

@jax.jit
def l2norm(u):
    return jnp.sqrt(jnp.mean(from_fixed_array(u) ** 2)).astype(jnp.float64)


def main(parfile, output_path):
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    x, dx = create_grid(params)
    dt = fixed_mul(dx, to_fixed_scalar(params["cfl"]))

    phi = jnp.empty_like(x, dtype=jnp.int64)
    pi = jnp.empty_like(x, dtype=jnp.int64)
    u = [phi, pi]
    u_names = ["Phi", "Pi"]

    u = initial_data(u, x, params)

    nt = params["nt"]
    time_fixed = jnp.int64(0)

    fname = f"{output_path}data_0000.curve"
    write_curve(fname, from_fixed_scalar(time_fixed), x, u_names, u)

    freq = params.get("output_frequency", 1)

    for i in range(1, nt + 1):
        u = rk2(u, x, dt)
        time_fixed += dt
        if i % freq == 0:
            print(
                f"Step {i:d}, t={from_fixed_scalar(time_fixed):.2e}, "
                f"|Phi|={l2norm(u[0]):.2e}, |Pi|={l2norm(u[1]):.2e}"
            )
            fname = f"{output_path}data_{i:04d}.curve"
            write_curve(fname, from_fixed_scalar(time_fixed), x, u_names, u)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:  python solver.py <parfile> <output_path>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
