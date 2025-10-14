import sys

import numpy as np
import tomllib

FRAC_BITS = 24
SCALE = 1 << FRAC_BITS
HALF = 1 << (FRAC_BITS - 1)
FIXED_ONE = SCALE


def _as_int_array(value):
    return np.asarray(value, dtype=np.int64)


def _to_result(value):
    arr = _as_int_array(value)
    if arr.shape == ():
        return int(arr)
    return arr.astype(np.int64)


def to_fixed_scalar(value):
    return int(round(float(value) * SCALE))


def to_fixed_array(values):
    return np.rint(values * SCALE).astype(np.int64)


def from_fixed_scalar(value):
    return float(value) / SCALE


def from_fixed_array(values):
    return values.astype(np.float64) / SCALE


def fixed_mul(a, b):
    a_arr = _as_int_array(a)
    b_arr = _as_int_array(b)
    result = (a_arr * b_arr + HALF) >> FRAC_BITS #The addition of the Half Scale unit makes it so that it rounds to the nearest reather than truncating
    if result.shape == ():
        return int(result)
    return result.astype(np.int64)


def fixed_div(numerator, denominator):
    num = _as_int_array(numerator)
    den = _as_int_array(denominator)
    result = ((num << FRAC_BITS) + den // 2) // den
    if result.shape == ():
        return int(result)
    return result.astype(np.int64)


def fixed_div_int(value, divisor):
    arr = _as_int_array(value)
    result = (arr + divisor // 2) // divisor
    if result.shape == ():
        return int(result)
    return result.astype(np.int64)


def create_grid(params):
    nx = params["nx"]
    if nx < 2:
        raise ValueError("nx must be >= 2 for grid spacing computation")

    x_min = to_fixed_scalar(params["x_min"])
    x_max = to_fixed_scalar(params["x_max"])
    dx = fixed_div_int(x_max - x_min, nx - 1)
    x = np.linspace(x_min, x_max, nx, dtype=np.int64)
    #x = np.zeros(nx, dtype=np.int64)
    # Cant we just change this to linspace
    # for i in range(nx):
    #     x[i] = x_min + i * dx
    return x, dx


def initial_data(u, x, params):
    x0 = params.get("id_x0", 0.5)
    amp = params.get("id_amp", 1.0)
    omega = params.get("id_omega", 1.0)

    x_float = from_fixed_array(x)
    profile = amp * np.exp(-omega * (x_float - x0) ** 2)

    u[0][:] = to_fixed_array(profile)
    u[1][:] = 0


def grad(u, idx_by_12):
    up1 = np.empty_like(u, dtype=np.int64)
    up1[:-1] = u[1:]
    up1[-1] = u[0]

    up2 = np.empty_like(u, dtype=np.int64)
    up2[:-2] = u[2:]
    up2[-2:] = u[:2]

    um1 = np.empty_like(u, dtype=np.int64)
    um1[1:] = u[:-1]
    um1[0] = u[-1]

    um2 = np.empty_like(u, dtype=np.int64)
    um2[2:] = u[:-2]
    um2[:2] = u[-2:]

    stencil = -up2 + 8 * up1 - 8 * um1 + um2
    return fixed_mul(stencil, idx_by_12)


def rhs(dtu, u, x):
    dx = x[1] - x[0]
    inv_12_dx = fixed_div(FIXED_ONE, dx * 12)

    phi = u[0]
    pi = u[1]

    dx_phi = grad(phi, inv_12_dx)
    dx_pi = grad(pi, inv_12_dx)

    dtu[0][:] = dx_pi
    dtu[1][:] = dx_phi


def rk2(u, x, dt):
    nu = len(u)
    half_dt = fixed_mul(dt, to_fixed_scalar(0.5))

    up = []
    k1 = []
    for _ in range(nu):
        up.append(np.empty_like(u[0], dtype=np.int64))
        k1.append(np.empty_like(u[0], dtype=np.int64))

    rhs(k1, u, x)
    for i in range(nu):
        up[i][:] = u[i][:] + fixed_mul(k1[i], half_dt)

    rhs(k1, up, x)
    for i in range(nu):
        u[i][:] = u[i][:] + fixed_mul(k1[i], dt)


def write_curve(filename, time, x, u_names, u):
    x_float = from_fixed_array(x)
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        for name, values in zip(u_names, u):
            f.write(f"# {name}\n")
            values_float = from_fixed_array(values)
            for xi, vi in zip(x_float, values_float):
                f.write(f"{xi:.8e} {vi:.8e}\n")


def l2norm(u):
    return float(np.sqrt(np.mean(from_fixed_array(u) ** 2)))


def main(parfile, output_path):
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    x, dx = create_grid(params)
    dt = fixed_mul(dx, to_fixed_scalar(params["cfl"]))

    phi = np.empty_like(x, dtype=np.int64)
    pi = np.empty_like(x, dtype=np.int64)
    u = [phi, pi]
    u_names = ["Phi", "Pi"]

    initial_data(u, x, params)

    nt = params["nt"]
    time_fixed = 0

    fname = f"{output_path}data_0000.curve"
    write_curve(fname, from_fixed_scalar(time_fixed), x, u_names, u)

    freq = params.get("output_frequency", 1)

    for i in range(1, nt + 1):
        rk2(u, x, dt)
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
