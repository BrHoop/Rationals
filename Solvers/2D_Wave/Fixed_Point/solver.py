import math
import os
import sys
import tomllib

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),'/Users/bryso/Documents/Github/Rationals'
            #"/Users/isaacsudweeks/Library/CloudStorage/OneDrive-BrighamYoungUniversity/Personal Projects/Rationals",
        )
    ),
)

import numpy as np
import utils.ioxdmf as iox
from utils.eqs import Equations
from utils.fixed_point import fixed_point
from utils.grid import Grid
from utils.types import BCType


class Grid2D(Grid):
    """
    Fixed-point 2D grid.
    """

    def __init__(self, params):
        if "Nx" not in params:
            raise ValueError("Nx is required")
        if "Ny" not in params:
            raise ValueError("Ny is required")
        if "frac_bits" not in params:
            raise ValueError("frac_bits is required for fixed-point arithmetic")

        self.f = fixed_point(params["frac_bits"])

        nx_phys = params["Nx"]
        ny_phys = params["Ny"]
        xmin = self.f.to_fixed_scalar(params.get("Xmin", 0.0))
        xmax = self.f.to_fixed_scalar(params.get("Xmax", 1.0))
        ymin = self.f.to_fixed_scalar(params.get("Ymin", 0.0))
        ymax = self.f.to_fixed_scalar(params.get("Ymax", 1.0))

        dx = self.f.fixed_div_int(xmax - xmin, nx_phys - 1)
        dy = self.f.fixed_div_int(ymax - ymin, ny_phys - 1)

        ng = params.get("NGhost", 0)
        nx = nx_phys + 2 * ng
        ny = ny_phys + 2 * ng

        xmin -= dx * ng
        ymin -= dy * ng

        xi_x = np.empty(nx, dtype=np.int64)
        xi_y = np.empty(ny, dtype=np.int64)
        for i in range(nx):
            xi_x[i] = xmin + i * dx
        for j in range(ny):
            xi_y[j] = ymin + j * dy

        shp = [nx, ny]
        dxn = np.array([dx, dy], dtype=np.int64)

        super().__init__(shp, [xi_x, xi_y], dxn, ng)


def _fixed_inverse_length(fp: fixed_point, x_val: int, y_val: int) -> int:
    """
    Return 1/sqrt(x^2 + y^2) in fixed-point form.
    """
    r_sq = fp.fixed_mul(x_val, x_val) + fp.fixed_mul(y_val, y_val)
    if r_sq <= 0:
        return 0
    inv = 1.0 / math.sqrt(fp.from_fixed_scalar(r_sq))
    return fp.to_fixed_scalar(inv)


def grad_x(u: np.ndarray, g: Grid2D) -> np.ndarray:
    dudx = np.zeros_like(u, dtype=np.int64)
    idx_by_2 = g.f.fixed_div(g.f.to_fixed_scalar(1.0), 2 * g.dx[0])
    idx_by_12 = g.f.fixed_div(g.f.to_fixed_scalar(1.0), 12 * g.dx[0])

    dudx[2:-2, :] = g.f.fixed_mul((-u[4:, :] + 8 * u[3:-1, :] - 8 * u[1:-3, :] + u[:-4, :]), idx_by_12)
    dudx[0, :] = g.f.fixed_mul((-3 * u[0, :] + 4 * u[1, :] - u[2, :]), idx_by_2)
    dudx[1, :] = g.f.fixed_mul((-u[0, :] + u[2, :]), idx_by_2)
    dudx[-2, :] = g.f.fixed_mul((-u[-3, :] + u[-1, :]), idx_by_2)
    dudx[-1, :] = g.f.fixed_mul((u[-3, :] - 4 * u[-2, :] + 3 * u[-1, :]), idx_by_2)
    return dudx


def grad_y(u: np.ndarray, g: Grid2D) -> np.ndarray:
    dudy = np.zeros_like(u, dtype=np.int64)
    idy_by_2 = g.f.fixed_div(g.f.to_fixed_scalar(1.0), 2 * g.dx[1])
    idy_by_12 = g.f.fixed_div(g.f.to_fixed_scalar(1.0), 12 * g.dx[1])

    dudy[:, 2:-2] = g.f.fixed_mul(
        (-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, :-4]), idy_by_12
    )

    dudy[:, 0] = g.f.fixed_mul((-3 * u[:, 0] + 4 * u[:, 1] - u[:, 2]), idy_by_2)
    dudy[:, 1] = g.f.fixed_mul((-u[:, 0] + u[:, 2]), idy_by_2)
    dudy[:, -2] = g.f.fixed_mul((-u[:, -3] + u[:, -1]), idy_by_2)
    dudy[:, -1] = g.f.fixed_mul((u[:, -3] - 4 * u[:, -2] + 3 * u[:, -1]), idy_by_2)

    return dudy


def grad_xx(u: np.ndarray, g: Grid2D) -> np.ndarray:
    idx_sq = g.f.fixed_div(g.f.to_fixed_scalar(1.0), g.f.fixed_mul(g.dx[0], g.dx[0]))
    idx_sq_by_12 = g.f.fixed_div(idx_sq, g.f.to_fixed_scalar(12.0))
    dxxu = np.zeros_like(u, dtype=np.int64)

    dxxu[2:-2, :] = g.f.fixed_mul(
        -u[4:, :]
        + 16 * u[3:-1, :]
        - 30 * u[2:-2, :]
        + 16 * u[1:-3, :]
        - u[:-4, :],
        idx_sq_by_12,
    )

    dxxu[0, :] = g.f.fixed_mul(
        2 * u[0, :] - 5 * u[1, :] + 4 * u[2, :] - u[3, :], idx_sq
    )
    dxxu[1, :] = g.f.fixed_mul(u[0, :] - 2 * u[1, :] + u[2, :], idx_sq)
    dxxu[-2, :] = g.f.fixed_mul(u[-3, :] - 2 * u[-2, :] + u[-1, :], idx_sq)
    dxxu[-1, :] = g.f.fixed_mul(
        -u[-4, :] + 4 * u[-3, :] - 5 * u[-2, :] + 2 * u[-1, :], idx_sq
    )
    return dxxu


def grad_yy(u: np.ndarray, g: Grid2D) -> np.ndarray:
    idy_sq = g.f.fixed_div(g.f.to_fixed_scalar(1.0), g.f.fixed_mul(g.dx[1], g.dx[1]))
    idy_sq_by_12 = g.f.fixed_div(idy_sq, g.f.to_fixed_scalar(12.0))
    dyyu = np.zeros_like(u, dtype=np.int64)

    dyyu[:, 2:-2] = g.f.fixed_mul(
        -u[:, 4:]
        + 16 * u[:, 3:-1]
        - 30 * u[:, 2:-2]
        + 16 * u[:, 1:-3]
        - u[:, :-4],
        idy_sq_by_12,
    )

    dyyu[:, 0] = g.f.fixed_mul(2 * u[:, 0] - 5 * u[:, 1] + 4 * u[:, 2] - u[:, 3], idy_sq)
    dyyu[:, 1] = g.f.fixed_mul(u[:, 0] - 2 * u[:, 1] + u[:, 2], idy_sq)
    dyyu[:, -2] = g.f.fixed_mul(u[:, -3] - 2 * u[:, -2] + u[:, -1], idy_sq)
    dyyu[:, -1] = g.f.fixed_mul(
        -u[:, -4] + 4 * u[:, -3] - 5 * u[:, -2] + 2 * u[:, -1], idy_sq
    )
    return dyyu


class ScalarField(Equations):
    def __init__(self, NU, g: Grid2D, bctype: str):
        if bctype == "SOMMERFELD":
            apply_bc = BCType.RHS
        elif bctype == "REFLECT":
            apply_bc = BCType.FUNCTION
        else:
            raise ValueError("Invalid boundary condition type. Use 'SOMMERFELD' or 'REFLECT'.")

        self.bound_cond = bctype
        super().__init__(NU, g, apply_bc)
        self.u = np.zeros((NU, *self.shp), dtype=np.int64)
        self.U_PHI = 0
        self.U_CHI = 1

    def rhs(self, dtu, u, x, y, g: Grid2D):
        dtphi = dtu[0]
        dtchi = dtu[1]
        phi = u[0]
        chi = u[1]

        dtphi[:] = chi[:]
        dxxphi = grad_xx(phi, g)
        dyyphi = grad_yy(phi, g)
        r = np.sqrt(x**2+y**2)
        dtchi[:] = dxxphi[:] + dyyphi[:] - g.f.to_fixed_array(np.sin(2*g.f.from_fixed_array(phi))/(r**2+1e-2))

        if self.apply_bc == BCType.RHS and self.bound_cond == "SOMMERFELD":
            x = g.xi[0]
            y = g.xi[1]
            Nx = g.shp[0]
            Ny = g.shp[1]
            dxphi = grad_x(phi, g)
            dyphi = grad_y(phi, g)
            falloff = g.f.to_fixed_scalar(1.0)
            bc_sommerfeld(g.f, dtphi, phi, dxphi, dyphi, falloff, g.nghost, x, y, Nx, Ny)
            dxchi = grad_x(chi, g)
            dychi = grad_y(chi, g)
            bc_sommerfeld(g.f, dtchi, chi, dxchi, dychi, falloff, g.nghost, x, y, Nx, Ny)

    def initialize(self, g: Grid2D, params):
        x = g.f.from_fixed_array(g.xi[0])
        y = g.f.from_fixed_array(g.xi[1])

        x0, y0 = params["id_x0"], params["id_y0"]
        amp, sigma = params["id_amp"], params["id_sigma"]
        X, Y = np.meshgrid(x, y, indexing="ij")
        profile = amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))
        self.u[self.U_PHI][:, :] = g.f.to_fixed_array(profile)
        self.u[self.U_CHI][:, :] = 0

    def apply_bcs(self, u, g: Grid2D):
        if self.bound_cond == "REFLECT":
            bc_reflect(u[self.U_PHI], u[self.U_CHI], g)


def bc_reflect(phi: np.ndarray, chi: np.ndarray, g: Grid2D):
    phi[0, :] = 0
    phi[-1, :] = 0
    phi[:, 0] = 0
    phi[:, -1] = 0

    three = g.f.to_fixed_scalar(3.0)
    chi[0, :] = g.f.fixed_div(4 * chi[1, :] - chi[2, :], three)
    chi[-1, :] = g.f.fixed_div(4 * chi[-2, :] - chi[-3, :], three)
    chi[:, 0] = g.f.fixed_div(4 * chi[:, 1] - chi[:, 2], three)
    chi[:, -1] = g.f.fixed_div(4 * chi[:, -2] - chi[:, -3], three)


def bc_sommerfeld(
    fp: fixed_point,
    dtf: np.ndarray,
    f: np.ndarray,
    dxf: np.ndarray,
    dyf: np.ndarray,
    falloff: int,
    ngz: int,
    x: np.ndarray,
    y: np.ndarray,
    Nx: int,
    Ny: int,
):
    if ngz <= 0:
        return

    for j in range(Ny):
        y_val = y[j]
        for i in range(ngz):
            x_val = x[i]
            inv_r = _fixed_inverse_length(fp, x_val, y_val)
            if inv_r == 0:
                dtf[i, j] = 0
            else:
                damping = (
                    fp.fixed_mul(x_val, dxf[i, j])
                    + fp.fixed_mul(y_val, dyf[i, j])
                    + fp.fixed_mul(falloff, f[i, j])
                )
                dtf[i, j] = -fp.fixed_mul(damping, inv_r)

        for i in range(Nx - ngz, Nx):
            x_val = x[i]
            inv_r = _fixed_inverse_length(fp, x_val, y_val)
            if inv_r == 0:
                dtf[i, j] = 0
            else:
                damping = (
                    fp.fixed_mul(x_val, dxf[i, j])
                    + fp.fixed_mul(y_val, dyf[i, j])
                    + fp.fixed_mul(falloff, f[i, j])
                )
                dtf[i, j] = -fp.fixed_mul(damping, inv_r)

    for i in range(Nx):
        x_val = x[i]
        for j in range(ngz):
            y_val = y[j]
            inv_r = _fixed_inverse_length(fp, x_val, y_val)
            if inv_r == 0:
                dtf[i, j] = 0
            else:
                damping = (
                    fp.fixed_mul(x_val, dxf[i, j])
                    + fp.fixed_mul(y_val, dyf[i, j])
                    + fp.fixed_mul(falloff, f[i, j])
                )
                dtf[i, j] = -fp.fixed_mul(damping, inv_r)

        for j in range(Ny - ngz, Ny):
            y_val = y[j]
            inv_r = _fixed_inverse_length(fp, x_val, y_val)
            if inv_r == 0:
                dtf[i, j] = 0
            else:
                damping = (
                    fp.fixed_mul(x_val, dxf[i, j])
                    + fp.fixed_mul(y_val, dyf[i, j])
                    + fp.fixed_mul(falloff, f[i, j])
                )
                dtf[i, j] = -fp.fixed_mul(damping, inv_r)


def rk2(eqs: ScalarField, x, y, g: Grid2D, dt: int):
    half_dt = g.f.fixed_mul(dt, g.f.to_fixed_scalar(0.5))
    nu = eqs.u.shape[0]
    up = np.empty_like(eqs.u, dtype=np.int64)
    k1 = np.empty_like(eqs.u, dtype=np.int64)

    eqs.rhs(k1, eqs.u, x, y, g)
    up[:] = eqs.u + g.f.fixed_mul(k1, half_dt)

    if eqs.apply_bc == BCType.FUNCTION:
        eqs.apply_bcs(up, g)

    eqs.rhs(k1, up, x, y, g)
    eqs.u[:] += g.f.fixed_mul(k1, dt)

    if eqs.apply_bc == BCType.FUNCTION:
        eqs.apply_bcs(eqs.u, g)


def main(parfile: str):
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    g = Grid2D(params)
    x = g.xi[0]
    y = g.xi[1]

    eqs = ScalarField(2, g, params["bound_cond"])
    eqs.initialize(g, params)

    output_dir = params["output_dir"]
    output_interval = params["output_interval"]
    os.makedirs(output_dir, exist_ok=True)
    dt = g.f.fixed_mul(g.f.to_fixed_scalar(params["cfl"]), g.dx[0])

    time_fixed = 0
    func_names = ["phi", "chi"]
    u_float = [g.f.from_fixed_array(arr) for arr in eqs.u]
    x_float = g.f.from_fixed_array(x)
    y_float = g.f.from_fixed_array(y)
    iox.write_hdf5(0, u_float, x_float, y_float, func_names, output_dir)

    Nt = params["Nt"]

    for step in range(1, Nt + 1):
        rk2(eqs, x_float, y_float, g, dt)
        time_fixed += dt

        print(f"Step {step:d} t={g.f.from_fixed_scalar(time_fixed):.2e}")
        if step % output_interval == 0:
            u_float = [g.f.from_fixed_array(arr) for arr in eqs.u]
            iox.write_hdf5(step, u_float, x_float, y_float, func_names, output_dir)

    iox.write_xdmf(
        output_dir,
        Nt,
        g.shp[0],
        g.shp[1],
        func_names,
        output_interval,
        g.f.from_fixed_scalar(dt),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python solver.py <parfile>")
        sys.exit(1)
    main(sys.argv[1])
