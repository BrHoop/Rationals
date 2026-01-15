import math
import os
import sys
import tomllib
from pathlib import Path

from numba import njit

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),'../../..'
        )
    ),
)

import numpy as np
import utils.ioxdmf as iox
from utils.eqs import Equations
from utils.fixed_point import fixed_point
from utils.grid import Grid
from utils.types import BCType, FilterApply, FilterType

project_root = Path.cwd()

class KreissOligerFilterO6_2D():
    """
    Kreiss-Oliger filter in 2D
    """

    def __init__(self, dx, dy, sigma, f, filter_boundary=True,):
        self.f = f
        self.sigma = self.f.to_fixed_scalar(sigma) #Lets store sigma in its fixed point form.
        self.filter_boundary = filter_boundary
        self.dx = dx
        self.dy = dy
        self.frequency = 1

    def get_sigma(self):
        return self.sigma

    def filter_x(self, u):
        dux = np.zeros_like(u)

        # Kreiss-Oliger filter in x direction
        dx = self.dx
        sigma = self.sigma
        fbound = self.filter_boundary
        self._apply_ko6_filter_x(dux, u, dx, sigma, fbound)
        return dux

    def filter_y(self, u):
        duy = np.zeros_like(u)

        # Kreiss-Oliger filter in the y direction
        dy = self.dy
        sigma = self.sigma
        fbound = self.filter_boundary
        self._apply_ko6_filter_y(duy, u ,dy, sigma, fbound)
        return duy

    def apply(self, u):
        """
        Apply the KO filter in both x and y directions and return the sum.
        """
        return self.filter_x(u) + self.filter_y(u)


    def _apply_ko6_filter_x(self, du, u, dx, sigma, filter_boundary):
        factor = self.f.fixed_div(sigma, 64 * dx)

        # centered stencil
        du[3:-3,:] = self.f.fixed_mul(factor,(u[:-6,:] - 6 * u[1:-5,:] + 15 * u[2:-4,:] - 20 * u[3:-3,:] + 15 * u[4:-2,:] - 6 * u[5:-1,:] + u[6:,:]))

        if filter_boundary:
            smr3 = self.f.fixed_div(self.f.to_fixed_scalar(9), 48*64*dx)
            smr2 = self.f.fixed_div(self.f.to_fixed_scalar(43), 48*64*dx)
            smr1 = self.f.fixed_div(self.f.to_fixed_scalar(49), 48*64*dx)

            spr3 = smr3
            spr2 = smr2
            spr1 = smr1


            du[0,:] = self.f.fixed_div(self.f.fixed_mul(sigma, (-u[0,:] + 3 * u[1,:] - 3 * u[2,:] + u[3,:])),smr3)

            du[1,:] = self.f.fixed_div(self.f.fixed_mul(sigma, (3 * u[0,:] - 10 * u[1,:] + 12 * u[2,:] - 6 * u[3,:] + u[4,:])),smr2)

            du[2,:] = self.f.fixed_div(self.f.fixed_mul(sigma, (-3 * u[0,:] + 12 * u[1,:] - 19 * u[2,:] + 15 * u[3,:] - 6.0 * u[4,:] + u[5,:])),smr1)

            du[-3,:] = self.f.fixed_div(self.f.fixed_mul(sigma, (u[-6,:] - 6 * u[-5,:] + 15 * u[-4,:] - 19 * u[-3,:] + 12 * u[-2,:] - 3 * u[-1,:])),spr1)

            du[-2,:] = self.f.fixed_div(self.f.fixed_mul(sigma, (u[-5,:] - 6 * u[-4,:] + 12 * u[-3,:] - 10 * u[-2,:] + 3 * u[-1,:])),spr2)

            du[-1,:] = self.f.fixed_div(self.f.fixed_mul(sigma, (u[-4,:] - 3 * u[-3,:] + 3 * u[-2,:] - u[-1,:])),spr3)


    def _apply_ko6_filter_y(self, du: np.ndarray, u: np.ndarray, dy: float, sigma: float, filter_boundary: bool):
        factor = self.f.fixed_div(sigma, 64 * dy)

        # centered stencil
        du[:,3:-3] = self.f.fixed_mul(factor,(u[:,:-6] - 6 * u[:,1:-5] + 15 * u[:,2:-4] - 20 * u[:,3:-3] + 15 * u[:,4:-2] - 6 * u[:,5:-1] + u[:,6:]))


        if filter_boundary:
            smr3 = self.f.fixed_div(self.f.to_fixed_scalar(9), 48*64*dy)
            smr2 = self.f.fixed_div(self.f.to_fixed_scalar(43), 48*64*dy)
            smr1 = self.f.fixed_div(self.f.to_fixed_scalar(49), 48*64*dy)

            spr3 = smr3
            spr2 = smr2
            spr1 = smr1

            du[:,0] = self.f.fixed_div(self.f.fixed_mul(sigma, (-u[:,0] + 3 * u[:,1] - 3 * u[:,2] + u[:,3])),smr3)

            du[:,1] = self.f.fixed_div(self.f.fixed_mul(sigma, (3 * u[:,0] - 10 * u[:,1] + 12 * u[:,2] - 6 * u[:,3] + u[:,4])),smr2)

            du[:,2] = self.f.fixed_div(self.f.fixed_mul(sigma, (-3 * u[:,0] + 12 * u[:,1] - 19 * u[:,2] + 15 * u[:,3] - 6.0 * u[:,4] + u[:,5])),smr1)

            du[:,-3] = self.f.fixed_div(self.f.fixed_mul(sigma, (u[:,-6] - 6 * u[:,-5] + 15 * u[:,-4] - 19 * u[:,-3] + 12 * u[:,-2] - 3 * u[:,-1])),spr1)

            du[:,-2] = self.f.fixed_div(self.f.fixed_mul(sigma, (u[:,-5] - 6 * u[:,-4] + 12 * u[:,-3] - 10 * u[:,-2] + 3 * u[:,-1])),spr2)

            du[:,-1] = self.f.fixed_div(self.f.fixed_mul(sigma, (u[:,-4] - 3 * u[:,-3] + 3 * u[:,-2] - u[:,-1])),spr3)



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


def _fixed_radius(fp: fixed_point, x_val: int, y_val: int, eps_fixed: int) -> int:
    """
    Return max(hypot(x, y), eps) in fixed-point form; eps is already fixed point.
    """
    r_sq = fp.fixed_mul(x_val, x_val) + fp.fixed_mul(y_val, y_val)
    r_float = math.sqrt(max(fp.from_fixed_scalar(r_sq), 0.0))
    eps_float = fp.from_fixed_scalar(eps_fixed)
    r_fixed = fp.to_fixed_scalar(max(r_float, eps_float))
    return r_fixed if r_fixed != 0 else eps_fixed


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
        X, Y = np.meshgrid(x, y, indexing="ij")
        dtphi[:] = chi[:]
        dxxphi = grad_xx(phi, g)
        dyyphi = grad_yy(phi, g)
        dtchi[:] = dxxphi[:] + dyyphi[:] - g.f.to_fixed_array(np.sin(2*g.f.from_fixed_array(phi))/(X**2+Y**2+1e-5))

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

    eps_fixed = fp.to_fixed_scalar(1e-12)
    if eps_fixed <= 0:
        eps_fixed = 1
    # Use face normals rather than radial vectors for Sommerfeld projection.
    for j in range(Ny):
        y_val = y[j]
        for i in range(ngz):
            x_val = x[i]
            r = _fixed_radius(fp, x_val, y_val, eps_fixed)
            falloff_term = fp.fixed_div(fp.fixed_mul(falloff, f[i, j]), r)
            dtf[i, j] = dxf[i, j] - falloff_term

        for i in range(Nx - ngz, Nx):
            x_val = x[i]
            r = _fixed_radius(fp, x_val, y_val, eps_fixed)
            falloff_term = fp.fixed_div(fp.fixed_mul(falloff, f[i, j]), r)
            dtf[i, j] = -dxf[i, j] - falloff_term

    for i in range(Nx):
        x_val = x[i]
        for j in range(ngz):
            y_val = y[j]
            r = _fixed_radius(fp, x_val, y_val, eps_fixed)
            falloff_term = fp.fixed_div(fp.fixed_mul(falloff, f[i, j]), r)
            dtf[i, j] = dyf[i, j] - falloff_term

        for j in range(Ny - ngz, Ny):
            y_val = y[j]
            r = _fixed_radius(fp, x_val, y_val, eps_fixed)
            falloff_term = fp.fixed_div(fp.fixed_mul(falloff, f[i, j]), r)
            dtf[i, j] = -dyf[i, j] - falloff_term


def rk2(eqs: ScalarField, x, y, g: Grid2D, dt: int,flt:KreissOligerFilterO6_2D):
    half_dt = g.f.fixed_mul(dt, g.f.to_fixed_scalar(0.5))
    nu = eqs.u.shape[0]
    up = np.empty_like(eqs.u, dtype=np.int64)
    k1 = np.empty_like(eqs.u, dtype=np.int64)
    if eqs.apply_bc == BCType.FUNCTION:
        eqs.apply_bcs(eqs.u, g)
    eqs.rhs(k1, eqs.u, x, y, g)

    #Apply KO filter for first stage
    for i in range(nu):
        k1[i] += flt.apply(eqs.u[i])
    up[:] = eqs.u + g.f.fixed_mul(k1, half_dt)

    if eqs.apply_bc == BCType.FUNCTION:
        eqs.apply_bcs(up, g)

    eqs.rhs(k1, up, x, y, g)
    #Apply KO filter for second stage
    for i in range(nu):
        k1[i] += flt.apply(up[i])

    eqs.u[:] += g.f.fixed_mul(k1, dt)

    if eqs.apply_bc == BCType.FUNCTION:
        eqs.apply_bcs(eqs.u, g)


def main(parfile: str, output_dir:str):
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    g = Grid2D(params)
    x = g.xi[0]
    y = g.xi[1]

    eqs = ScalarField(2, g, params["bound_cond"])
    eqs.initialize(g, params)
    if eqs.apply_bc == BCType.FUNCTION:
        eqs.apply_bcs(eqs.u, g)

    output_interval = params["output_interval"]

    dt = g.f.fixed_mul(g.f.to_fixed_scalar(params["cfl"]), g.dx[0])

    #Setup KO filter
    flt = KreissOligerFilterO6_2D(g.dx[0], g.dx[1], sigma=params.get("ko_sigma", 0.01), f=g.f,filter_boundary=False)

    time_fixed = 0
    func_names = ["phi", "chi"]
    u_float = [g.f.from_fixed_array(arr) for arr in eqs.u]
    x_float = g.f.from_fixed_array(x)
    y_float = g.f.from_fixed_array(y)
    iox.write_hdf5(0, u_float, x_float, y_float, func_names, output_dir)

    Nt = params["Nt"]

    for step in range(1, Nt + 1):
        rk2(eqs, x_float, y_float, g, dt, flt)
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
    if len(sys.argv) > 1:
        parfile = sys.argv[1]
    else:
        parfile = (project_root / "Solvers/2D_Wave/Fixed_Point/params.toml").as_posix()

    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = (project_root / "Solvers/2D_Wave/Fixed_Point/data").as_posix()

    os.makedirs(output_dir, exist_ok=True)

    main(parfile,output_dir)
