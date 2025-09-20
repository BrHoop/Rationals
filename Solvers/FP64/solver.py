import numpy as np
import tomllib
import sys

def create_grid(params):
    x_min = params["x_min"]
    x_max = params["x_max"]
    nx = params["nx"]
    dx = (x_max - x_min) / (nx -1)
    x = np.zeros(nx)
    for i in range(nx):
        x[i] = x_min + i*dx
    return x, dx

def initial_data(u, x, params):
    x0 = params.get("id_x0", 0.5)
    amp = params.get("id_amp", 1.0)
    omega = params.get("id_omega", 1.0)
    u[0][:] = amp * np.exp(-omega * (x[:] - x0) ** 2)
    u[1][:] = 0.0

def grad(u, dx):
    idx_by_12 = 1.0 / (12 * dx)

    up1 = np.empty_like(u)
    up1[:-1] = u[1:]
    up1[-1] = u[0]

    up2 = np.empty_like(u)
    up2[:-2] = u[2:]
    up2[-2:] = u[:2]

    um1 = np.empty_like(u)
    um1[1:] = u[:-1]
    um1[0] = u[-1]

    um2 = np.empty_like(u)
    um2[2:] = u[:-2]
    um2[:2] = u[-2:]

    return (-up2 + 8 * up1 - 8 * um1 + um2) * idx_by_12


def rhs(dtu, u, x):
    # RHS for the wave  equation
    dx = x[1] - x[0]
    Phi = u[0]
    Pi = u[1]
    dxPhi = grad(Phi, dx)
    dxPi = grad(Pi, dx)

    dtu[0][:] = dxPi[:]
    dtu[1][:] = dxPhi[:]
    

def rk2(u, x, dt):
    nu = len(u)

    up = []
    k1 = []
    for i in range(nu):
        ux = np.empty_like(u[0])
        kx = np.empty_like(u[0])
        up.append(ux)
        k1.append(kx)

    rhs(k1, u, x)
    for i in range(nu):
        up[i][:] = u[i][:] + 0.5 * dt * k1[i][:]

    rhs(k1, up, x)
    for i in range(nu):
        u[i][:] = u[i][:] + dt * k1[i][:]

def write_curve(filename, time, x, u_names, u):
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        for m in range(len(u_names)):
            f.write(f"# {u_names[m]}\n")
            for xi, di in zip(x, u[m]):
                f.write(f"{xi:.8e} {di:.8e}\n")

def l2norm(u):
    """
    Compute the L2 norm of an array.
    """
    return np.sqrt(np.mean(u**2))


def main(parfile, output_path):
    # Read parameters
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    # Create the grid and set time step size
    x, dx = create_grid(params)
    dt = params["cfl"] * dx

    # Allocating memory
    Phi = np.empty_like(x)
    Pi = np.empty_like(x)
    u = [Phi, Pi]
    u_names = ["Phi", "Pi"]

    initial_data(u, x, params)

    nt = params["nt"]
    time = 0.0

    iter = 0
    fname = f"{output_path}data_{iter:04d}.curve"
    write_curve(fname, time, x, u_names, u)

    freq = params.get("output_frequency", 1)

    # Inegrate in time
    for i in range(1, nt+1):
        rk2(u, x, dt)
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
    main(parfile, output_path)
