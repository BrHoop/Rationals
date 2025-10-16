

import os
import sys
import tomllib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/isaacsudweeks/Library/CloudStorage/OneDrive-BrighamYoungUniversity/Personal Projects/Rationals')))
import utils.ioxdmf as iox
from numba import njit
import utils
from utils.eqs import Equations
from utils.grid import Grid
from utils.sowave import ScalarField
import numpy as np
from utils.types import BCType





class Grid2D(Grid):
    """
    Class to define a 2D grid for a PDE system.
    Parameters:
    ----------
    Nx : int
        Number of grid points in the x-direction.
    Ny : int
        Number of grid points in the y-direction.
    """

    def __init__(self, params):
        if "Nx" not in params:
            raise ValueError("Nx is required")
        if "Ny" not in params:
            raise ValueError("Ny is required")
        self.f = utils.fixed_point.fixed_point(params['frac_bits'])
        nx = params["Nx"]
        ny = params["Ny"]
        xmin = self.f.to_fixed_scalar(params.get("Xmin", 0.0))
        xmax = self.f.to_fixed_scalar(params.get("Xmax", 1.0))
        ymin = self.f.to_fixed_scalar(params.get("Ymin", 0.0))
        ymax = self.f.to_fixed_scalar(params.get("Ymax", 1.0))

        dx = self.f.fixed_div_int(xmax - xmin, nx-1)
        dy = self.f.fixed_div_int(ymax - ymin, ny-1)
        ng = params.get("NGhost", 0)
        nx = nx + 2 * ng #These should be fine
        ny = ny + 2 * ng #These should be fine

        ng = self.f.to_fixed_scalar(ng)
        nx = self.f.to_fixed_scalar(nx)
        ny = self.f.to_fixed_scalar(ny)

        xmin -= self.f.fixed_mul(ng , dx)
        xmax += self.f.fixed_mul(ng , dx)
        ymin -= self.f.fixed_mul(ng , dy)
        ymax += self.f.fixed_mul(ng , dy)

        shp = [nx, ny]

        xi = [np.linspace(xmin, xmax, nx,dtype=np.int64), np.linspace(ymin, ymax, ny,dtype=np.int64)]

        dxn = self.f.to_fixed_array(np.array([dx, dy]))
        #print(f"Grid2D: {shp}, {xi}, {dxn}")
        super().__init__(shp, xi, dxn, ng)
        #TODO check this function to see if it is doing the Fixed Point conversion correctly



def grad_x(u,g:Grid2D):
    dudx = np.zeros_like(u)
    idx_by_2 = g.f.fixed_div(g.f.to_fixed_scalar(1.0), 2*g.dx[0])
    idx_by_12 = g.f.fixed_div(g.f.to_fixed_scalar(12.0), 12*g.dx[0]) #This should be fine because dx should already be in fixed point form


    #center stencil
    dudx[2:-2, :] = g.f.fixed_mul((-u[4:, :] + 8 * u[3:-1, :] - 8 * u[1:-3, :] + u[:-4, :]),idx_by_12)

    # 4th order boundary stencils
    dudx[0, :] = g.f.fixed_mul((-3 * u[0, :] + 4 * u[1, :] - u[2, :]), idx_by_2)
    dudx[1, :] = g.f.fixed_mul((-u[0, :] + u[2, :]),idx_by_2)
    dudx[-2, :] = g.f.fixed_mul((-u[-3, :] + u[-1, :]),idx_by_2)
    dudx[-1, :] = g.f.fixed_mul((u[-3, :] - 4 * u[-2, :] + 3 * u[-1, :]), idx_by_2)

    return dudx

def grad_y(u,g:Grid2D):
    dudy = np.zeros_like(u)
    idy_by_2 = g.f.fixed_div(g.f.to_fixed_scalar(1.0), 2*g.dx[1])
    idy_by_12 = g.f.fixed_div(g.f.to_fixed_scalar(12.0), 12*g.dx[1])

    # center stencil
    dudy[:, 2:-2] = g.f.fixed_mul((-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, 0:-4]), idy_by_12)

    # 4th order boundary stencils
    dudy[:, 0] = g.f.fixed_mul((-3 * u[:, 0] + 4 * u[:, 1] - u[:, 2]), idy_by_2)
    dudy[:, 1] = g.f.fixed_mul((-u[:, 0] + u[:, 2]), idy_by_2)
    dudy[:, -2] = g.f.fixed_mul((-u[:, -3] + u[:, -1]), idy_by_2)
    dudy[:, -1] = g.f.fixed_mul((u[:, -3] - 4 * u[:, -2] + 3 * u[:, -1]), idy_by_2)

    return dudy


def grad_xx(u,g):
    idx_sqrd = g.f.fixed_div(g.f.to_fixed_scalar(1.0), g.f.fixed_mul(g.dx[0], g.dx[0]))
    idx_sqrd_by_12 = int(idx_sqrd / 12.0)
    dxxu = np.zeros_like(u)
    dxxu[2:-2, :] = g.f.fixed_mul((-u[4:, :] + 16 * u[3:-1, :] - 30 * u[2:-2, :] + 16 * u[1:-3, :] - u[0:-4, :]), idx_sqrd_by_12)
    # boundary stencils
    dxxu[0, :] = g.f.fixed_mul((2 * u[0, :] - 5 * u[1, :] + 4 * u[2, :] - u[3, :]), idx_sqrd)
    dxxu[1, :] = g.f.fixed_mul((u[0, :] - 2 * u[1, :] + u[2, :]), idx_sqrd)
    dxxu[-2, :] = g.f.fixed_mul((u[-3, :] - 2 * u[-2, :] + u[-1, :]), idx_sqrd)
    dxxu[-1, :] = g.f.fixed_mul((-u[-4, :] + 4 * u[-3, :] - 5 * u[-2, :] + 2 * u[-1, :]), idx_sqrd)
    return dxxu


def grad_yy(u,g):
    idy_sqrd = g.f.fixed_div(g.f.to_fixed_scalar(1.0), g.f.fixed_mul(g.dx[1], g.dx[1]))
    idy_sqrd_by_12 = int(idy_sqrd / 12.0)
    dyyu = np.zeros_like(u)

    # centered stencils
    dyyu[:, 2:-2] = g.f.fixed_mul((-u[:, 4:] + 16 * u[:, 3:-1] - 30 * u[:, 2:-2] + 16 * u[:, 1:-3] - u[:, 0:-4]), idy_sqrd_by_12)

    # boundary stencils
    dyyu[:, 0] = g.f.fixed_mul((2 * u[:, 0] - 5 * u[:, 1] + 4 * u[:, 2] - u[:, 3]), idy_sqrd)
    dyyu[:, 1] = g.f.fixed_mul((u[:, 0] - 2 * u[:, 1] + u[:, 2]), idy_sqrd)
    dyyu[:, -2] = g.f.fixed_mul((u[:, -3] - 2 * u[:, -2] + u[:, -1]), idy_sqrd)
    dyyu[:, -1] = g.f.fixed_mul((-u[:, -4] + 4 * u[:, -3] - 5 * u[:, -2] + 2 * u[:, -1]), idy_sqrd)
    return dyyu

class ScalarField(Equations):
    def __init__(self, NU, g: Grid2D, bctype):
        if bctype == "SOMMERFELD":
            apply_bc = BCType.RHS
        elif bctype == "REFLECT":
            apply_bc = BCType.FUNCTION
        else:
            raise ValueError(
                    "Invalid boundary condition type. Use 'SOMMERFELD' or 'REFLECT'."
                )

        self.bound_cond = bctype
        super().__init__(NU, g, apply_bc)
        self.U_PHI = 0
        self.U_CHI = 1

    def rhs(self, dtu, u, g: Grid2D):
        #TODO this seems like it is good, but check it.
        dtphi = dtu[0]
        dtchi = dtu[1]
        phi = u[0]
        chi = u[1]

        dtphi[:] = chi[:]
        dxxphi = grad_xx(phi,g)
        dyyphi = grad_yy(phi,g)
        dtchi[:] = dxxphi[:] + dyyphi[:]

        if self.apply_bc == BCType.RHS and self.bound_cond == "SOMMERFELD":
                # Sommerfeld boundary conditions
            x = g.xi[0]
            y = g.xi[1]
            Nx = g.shp[0]
            Ny = g.shp[1]
            dxphi = grad_x(phi,g)
            dyphi = grad_y(phi,g)
            bc_sommerfeld(dtphi, phi, dxphi, dyphi, 1.0, 1, x, y, Nx, Ny)
            dxchi = grad_x(chi,g)
            dychi = grad_y(chi,g)
            bc_sommerfeld(dtchi, chi, dxchi, dychi, 1.0, 1, x, y, Nx, Ny)

    def initialize(self, g: Grid2D, params):
        x = g.xi[0]
        y = g.xi[1]

        #All this stuff is in FP 64 and then is converted
        x0, y0 = params["id_x0"], params["id_y0"]
        amp, sigma = params["id_amp"], params["id_sigma"]
        X, Y = np.meshgrid(x, y, indexing="ij")
        profile = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))
        self.u[0][:, :] = g.f.to_fixed_array(profile)
        self.u[1][:, :] = 0

    def apply_bcs(self, u, g: Grid2D):
        if self.bound_cond == "REFLECT":
            bc_reflect(u[0], u[1],g)


def bc_reflect(phi, chi,g):
    # Reflective boundary conditions
    phi[0, :] = 0
    phi[-1, :] = 0
    phi[:, 0] = 0
    phi[:, -1] = 0

    three = g.f.to_fixed_scalar(3)

    chi[0, :] = g.f.fixed_div((4 * chi[1, :] - chi[2, :]), three)
    chi[-1, :] = g.f.fixed_div((4 * chi[-2, :] - chi[-3, :]), three)
    chi[:, 0] = g.f.fixed_div((4 * chi[:, 1] - chi[:, 2]), three)
    chi[:, -1] = g.f.fixed_div((4 * chi[:, -2] - chi[:, -3]),three)

#TODO: Switch this one later it is STILL FP64
def bc_sommerfeld(dtf, f, dxf, dyf, falloff, ngz, x, y, Nx, Ny):
    for j in range(Ny):
        for i in range(ngz):
            # xmin boundary
            inv_r = 1.0 / np.sqrt(x[i] ** 2 + y[j] ** 2)
            dtf[i, j] = (-(x[i] * dxf[i, j] + y[j] * dyf[i, j] + falloff * f[i, j]) * inv_r)
        for i in range(Nx - ngz, Nx):
            # xmax boundary
            inv_r = 1.0 / np.sqrt(x[i] ** 2 + y[j] ** 2)
            dtf[i, j] = (-(x[i] * dxf[i, j] + y[j] * dyf[i, j] + falloff * f[i, j]) * inv_r)

    for i in range(Nx):
        for j in range(ngz):
            # ymin boundary
            inv_r = 1.0 / np.sqrt(x[i] ** 2 + y[j] ** 2)
            dtf[i, j] = (-(x[i] * dxf[i, j] + y[j] * dyf[i, j] + falloff * f[i, j]) * inv_r)
        for j in range(Ny - ngz, Ny):
            # ymax boundary
            inv_r = 1.0 / np.sqrt(x[i] ** 2 + y[j] ** 2)
            dtf[i, j] = (-(x[i] * dxf[i, j] + y[j] * dyf[i, j] + falloff * f[i, j]) * inv_r)


def rk2(eqs, g, dt):
    nu = len(eqs.u)
    half_dt = g.f.fixed_div(dt, g.f.to_fixed_scalar(0.5))

    up = []
    k1 = []
    for _ in range(nu):
        up.append(np.empty_like(eqs.u[0], dtype=object))
        k1.append(np.empty_like(eqs.u[0], dtype=object))
    eqs.rhs(k1, eqs.u ,g)
    for i in range(nu):
        up[i][:] = eqs.u[i][:] + g.f.fixed_mul(half_dt,k1[i][:])
        up[i][:] = eqs.u[i][:] + g.f.fixed_mul(half_dt,k1[i][:])
    eqs.rhs(k1, up, g)
    for i in range(nu):
        eqs.u[i][:] = eqs.u[i][:] + g.f.fixed_mul(k1[i][:], dt)


def main(parfile):

    #Read the parfile
    with open(parfile,"rb") as f:
        params = tomllib.load(f)
    g = Grid2D(params)
    x = g.xi[0]
    y = g.xi[1]
    dx = g.dx[0]
    dy = g.dx[1]

    eqs = ScalarField(2, g, params["bound_cond"])
    eqs.initialize(g, params)

    output_dir = params["output_dir"]
    output_interval = params["output_interval"]
    os.makedirs(output_dir, exist_ok=True)
    #TODO Start here
    dt = params["cfl"] * dx

    time = 0.0
    func_names = ["phi","chi"]
    iox.write_hdf5(0,g.f.from_fixed_array(eqs.u),g.f.from_fixed_array(x),g.f.from_fixed_array(y),func_names,output_dir)

    Nt = params["Nt"]


    for i in range(1, Nt +1):
        rk2(eqs, g, dt)
        time += dt
        print(f"Step {i:d} t={time:.2e}")
        if i % output_interval == 0:
            iox.write_hdf5(i,eqs.u,x,y,func_names,output_dir)
    iox.write_xdmf(output_dir, Nt, g.shp[0], g.shp[1], func_names, output_interval, dt)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python solver.py <parfile>")
        sys.exit(1)
    parfile = sys.argv[1]
    main(parfile)
