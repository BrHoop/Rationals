import numpy as np
import tomllib
import sys
from fractions import Fraction as fc
import math

N = 10_000_000  # max denominator for float->Fraction conversion

def create_grid(params):
    x_min = fc(params["x_min"])
    x_max = fc(params["x_max"])
    nx = params["nx"]
    dx = fc(x_max - x_min, nx)   # exact fraction
    x = np.empty(nx, dtype=object)
    for i in range(nx):
        x[i] = x_min + i * dx
    return x, dx

def initial_data(u, x, params):
    x0 = fc(params.get("id_x0", 0.5))
    amp = fc(params.get("id_amp", 1.0))
    omega = fc(params.get("id_omega", 1.0))
    for i in range(len(u[0])):
        # exp returns float, approximate as Fraction
        val = amp * math.exp(float(-omega * (x[i] - x0) ** 2))
        u[0][i] = fc(val).limit_denominator(N)
    for i in range(len(u[1])):
        u[1][i] = fc(0)

def grad(u, dx):
    nx = len(u)
    du = np.empty(nx, dtype=object)
    idx_by_12 = fc(1, 12 * dx)   # exact reciprocal
    for i in range(nx):
        du[i] = ((-u[(i+2) % nx]
                  + 8 * u[(i+1) % nx]
                  - 8 * u[(i-1) % nx]
                  + u[(i-2) % nx]) * idx_by_12).limit_denominator(N)
    return du

def rhs(dtu, u, x):
    dx = x[1] - x[0]
    Phi = u[0]
    Pi = u[1]
    dxPhi = grad(Phi, dx)
    dxPi = grad(Pi, dx)
    dtu[0][:] = dxPi
    dtu[1][:] = dxPhi

def rk2(u, x, dt):
    nu = len(u)
    up = []
    k1 = []
    for i in range(nu):
        ux = np.empty_like(u[0], dtype=object)
        kx = np.empty_like(u[0], dtype=object)
        up.append(ux)
        k1.append(kx)

    rhs(k1, u, x)
    for i in range(nu):
        for j in range(len(up[i])):
            up[i][j] = u[i][j] + fc(1, 2) * dt * k1[i][j]

    rhs(k1, up, x)
    for i in range(nu):
        for j in range(len(u[i])):
            u[i][j] = u[i][j] + dt * k1[i][j]

def write_curve(filename, time, x, u_names, u):
    with open(filename, "w") as f:
        f.write(f"# TIME {float(time):.8e}\n")
        for m in range(len(u_names)):
            f.write(f"# {u_names[m]}\n")
            for xi, di in zip(x, u[m]):
                f.write(f"{float(xi):.8e} {float(di):.8e}\n")

def l2norm(u):
    """
    Compute the L2 norm of an array (float output).
    """
    arr = np.array([float(val) for val in u])
    return math.sqrt(np.mean(arr ** 2))

def main(parfile):
    # Read parameters
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    # Create the grid and set time step size
    x, dx = create_grid(params)
    dt = fc(params["cfl"]) * dx  # cfl wrapped in Fraction

    # Allocating memory
    Phi = np.empty_like(x, dtype=object)
    Pi = np.empty_like(x, dtype=object)
    u = np.array([Phi, Pi], dtype=object)
    u_names = ["Phi", "Pi"]

    initial_data(u, x, params)

    nt = params["nt"]
    time = fc(0)

    iter = 0
    fname = f"data_{iter:04d}.curve"
    write_curve(fname, time, x, u_names, u)

    freq = params.get("output_frequency", 1)

    # Integrate in time
    for i in range(1, nt + 1):
        rk2(u, x, dt)
        time += dt
        if i % freq == 0:
            print(f"Step {i:d}, t={float(time):.2e}, "
                  f"|Phi|={l2norm(u[0]):.2e}, |Pi|={l2norm(u[1]):.2e}")
            fname = f"PB_data_frac{i:04d}.curve"
            write_curve(fname, time, x, u_names, u)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python solver.py <parfile>")
        sys.exit(1)

    parfile = sys.argv[1]
    main(parfile)
