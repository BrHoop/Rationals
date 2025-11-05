import os
import sys
import tomllib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import utils.ioxdmf as iox
from numba import njit

from utils.eqs import Equations
from utils.grid import Grid
from utils.sowave import ScalarField
import numpy as np
from utils.types import BCType, FilterApply, FilterType


class KreissOligerFilterO6_2D():
    """
    Kreiss-Oliger filter in 2D
    """

    def __init__(self, dx, dy, sigma, filter_boundary=True):
        self.sigma = sigma
        self.filter_boundary = filter_boundary
        self.dx = dx
        self.dy = dy

        filter_apply = FilterApply.RHS
        filter_type = FilterType.KREISS_OLIGER_O6
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

    @staticmethod
    @njit
    def _apply_ko6_filter_x(du : np.ndarray, u : np.ndarray, dx : float, sigma : float, filter_boundary : bool):
        factor = sigma / (64.0 * dx)

        # centered stencil
        du[3:-3,:] = factor * (
            u[:-6,:]
            - 6.0 * u[1:-5,:]
            + 15.0 * u[2:-4,:]
            - 20.0 * u[3:-3,:]
            + 15.0 * u[4:-2,:]
            - 6.0 * u[5:-1,:]
            + u[6:,:]
        )

        if filter_boundary:
            smr3 = 9.0 / 48.0 * 64 * dx
            smr2 = 43.0 / 48.0 * 64 * dx
            smr1 = 49.0 / 48.0 * 64 * dx
            spr3 = smr3
            spr2 = smr2
            spr1 = smr1
            du[0,:] = sigma * (-u[0,:] + 3.0 * u[1,:] - 3.0 * u[2,:] + u[3,:]) / smr3
            du[1,:] = (
                sigma
                * (3.0 * u[0,:] - 10.0 * u[1,:] + 12.0 * u[2,:] - 6.0 * u[3,:] + u[4,:])
                / smr2
            )
            du[2,:] = (
                sigma
                * (
                    -3.0 * u[0,:]
                    + 12.0 * u[1,:]
                    - 19.0 * u[2,:]
                    + 15.0 * u[3,:]
                    - 6.0 * u[4,:]
                    + u[5,:]
                )
                / smr1
            )
            du[-3,:] = (
                sigma
                * (
                    u[-6,:]
                    - 6.0 * u[-5,:]
                    + 15.0 * u[-4,:]
                    - 19.0 * u[-3,:]
                    + 12.0 * u[-2,:]
                    - 3.0 * u[-1,:]
                )
                / spr1
            )
            du[-2,:] = (
                sigma
                * (u[-5,:] - 6.0 * u[-4,:] + 12.0 * u[-3,:] - 10.0 * u[-2,:] + 3.0 * u[-1,:])
                / spr2
            )
            du[-1,:] = sigma * (u[-4,:] - 3.0 * u[-3,:] + 3.0 * u[-2,:] - u[-1,:]) / spr3


    @staticmethod
    @njit
    def _apply_ko6_filter_y(du: np.ndarray, u: np.ndarray, dy: float, sigma: float, filter_boundary: bool):
        factor = sigma / (64.0 * dy)

        # centered stencil
        du[:,3:-3] = factor * (
            u[:,:-6]
            - 6.0 * u[:,1:-5]
            + 15.0 * u[:,2:-4]
            - 20.0 * u[:,3:-3]
            + 15.0 * u[:,4:-2]
            - 6.0 * u[:,5:-1]
            + u[:,6:]
        )

        if filter_boundary:
            smr3 = 9.0 / 48.0 * 64 * dy
            smr2 = 43.0 / 48.0 * 64 * dy
            smr1 = 49.0 / 48.0 * 64 * dy
            spr3 = smr3
            spr2 = smr2
            spr1 = smr1
            du[:,0] = sigma * (-u[:,0] + 3.0 * u[:,1] - 3.0 * u[:,2] + u[:,3]) / smr3
            du[:,1] = (
                sigma
                * (3.0 * u[:,0] - 10.0 * u[:,1] + 12.0 * u[:,2] - 6.0 * u[:,3] + u[:,4])
                / smr2
            )
            du[:,2] = (
                sigma
                * (
                    -3.0 * u[:,0]
                    + 12.0 * u[:,1]
                    - 19.0 * u[:,2]
                    + 15.0 * u[:,3]
                    - 6.0 * u[:,4]
                    + u[:,5]
                )
                / smr1
            )
            du[:,-3] = (
                sigma
                * (
                    u[:,-6]
                    - 6.0 * u[:,-5]
                    + 15.0 * u[:,-4]
                    - 19.0 * u[:,-3]
                    + 12.0 * u[:,-2]
                    - 3.0 * u[:,-1]
                )
                / spr1
            )
            du[:,-2] = (
                sigma
                * (u[:,-5] - 6.0 * u[:,-4] + 12.0 * u[:,-3] - 10.0 * u[:,-2] + 3.0 * u[:,-1])
                / spr2
            )
            du[:,-1] = sigma * (u[:,-4] - 3.0 * u[:,-3] + 3.0 * u[:,-2] - u[:,-1]) / spr3

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

        nx = params["Nx"]
        ny = params["Ny"]
        xmin = params.get("Xmin", 0.0)
        xmax = params.get("Xmax", 1.0)
        ymin = params.get("Ymin", 0.0)
        ymax = params.get("Ymax", 1.0)

        dx = (xmax - xmin) / (nx - 1)
        dy = (ymax - ymin) / (ny - 1)
        ng = params.get("NGhost", 0)
        nx = nx + 2 * ng
        ny = ny + 2 * ng
        xmin -= ng * dx
        xmax += ng * dx
        ymin -= ng * dy
        ymax += ng * dy

        shp = [nx, ny]

        xi = [np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)]

        dxn = np.array([dx, dy])
        #print(f"Grid2D: {shp}, {xi}, {dxn}")
        super().__init__(shp, xi, dxn, ng)



def grad_x(u,g:Grid2D):
    dudx = np.zeros_like(u)
    idx_by_2 = 1.0 / (2*g.dx[0])
    idx_by_12 = 1.0 / (12 * g.dx[0])

    #center stencil
    dudx[2:-2, :] = (-u[4:, :] + 8 * u[3:-1, :] - 8 * u[1:-3, :] + u[:-4, :]) * idx_by_12

    # 4th order boundary stencils
    dudx[0, :] = (-3 * u[0, :] + 4 * u[1, :] - u[2, :]) * idx_by_2
    dudx[1, :] = (-u[0, :] + u[2, :]) * idx_by_2
    dudx[-2, :] = (-u[-3, :] + u[-1, :]) * idx_by_2
    dudx[-1, :] = (u[-3, :] - 4 * u[-2, :] + 3 * u[-1, :]) * idx_by_2

    return dudx

def grad_y(u,g:Grid2D):
    dudy = np.zeros_like(u)
    idy_by_2 = 1.0 / (2 * g.dx[1])
    idy_by_12 = 1.0 / (12 * g.dx[1])

    # center stencil
    dudy[:, 2:-2] = (-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, 0:-4]) * idy_by_12

    # 4th order boundary stencils
    dudy[:, 0] = (-3 * u[:, 0] + 4 * u[:, 1] - u[:, 2]) * idy_by_2
    dudy[:, 1] = (-u[:, 0] + u[:, 2]) * idy_by_2
    dudy[:, -2] = (-u[:, -3] + u[:, -1]) * idy_by_2
    dudy[:, -1] = (u[:, -3] - 4 * u[:, -2] + 3 * u[:, -1]) * idy_by_2

    return dudy


def grad_xx(u,g):
    idx_sqrd = 1.0 / g.dx[0]**2
    idx_sqrd_by_12 = idx_sqrd / 12.0
    dxxu = np.zeros_like(u)
    dxxu[2:-2, :] = (-u[4:, :] + 16 * u[3:-1, :] - 30 * u[2:-2, :] + 16 * u[1:-3, :] - u[0:-4, :]) * idx_sqrd_by_12
    # boundary stencils
    dxxu[0, :] = (2 * u[0, :] - 5 * u[1, :] + 4 * u[2, :] - u[3, :]) * idx_sqrd
    dxxu[1, :] = (u[0, :] - 2 * u[1, :] + u[2, :]) * idx_sqrd
    dxxu[-2, :] = (u[-3, :] - 2 * u[-2, :] + u[-1, :]) * idx_sqrd
    dxxu[-1, :] = (-u[-4, :] + 4 * u[-3, :] - 5 * u[-2, :] + 2 * u[-1, :]) * idx_sqrd
    return dxxu


def grad_yy(u,g):
    idy_sqrd = 1.0 / g.dx[1]**2
    idy_sqrd_by_12 = idy_sqrd / 12.0
    dyyu = np.zeros_like(u)

    # centered stencils
    dyyu[:, 2:-2] = (
                            -u[:, 4:] + 16 * u[:, 3:-1] - 30 * u[:, 2:-2] + 16 * u[:, 1:-3] - u[:, 0:-4]
                    ) * idy_sqrd_by_12

    # boundary stencils
    dyyu[:, 0] = (2 * u[:, 0] - 5 * u[:, 1] + 4 * u[:, 2] - u[:, 3]) * idy_sqrd
    dyyu[:, 1] = (u[:, 0] - 2 * u[:, 1] + u[:, 2]) * idy_sqrd
    dyyu[:, -2] = (u[:, -3] - 2 * u[:, -2] + u[:, -1]) * idy_sqrd
    dyyu[:, -1] = (
                          -u[:, -4] + 4 * u[:, -3] - 5 * u[:, -2] + 2 * u[:, -1]
                  ) * idy_sqrd
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

    def rhs(self, dtu, u, x, y, g: Grid2D):
        dtphi = dtu[0]
        dtchi = dtu[1]
        phi = u[0]
        chi = u[1]

        dtphi[:] = chi[:]
        dxxphi = grad_xx(phi,g)
        dyyphi = grad_yy(phi,g)
        X, Y = np.meshgrid(x, y, indexing="ij")
        dtchi[:] = dxxphi[:] + dyyphi[:] - np.sin(2*phi)/(X**2+Y**2+1e-5)

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
        x0, y0 = params["id_x0"], params["id_y0"]
        amp, sigma = params["id_amp"], params["id_sigma"]
        X, Y = np.meshgrid(x, y, indexing="ij")
        self.u[0][:, :] = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))
        self.u[1][:, :] = 0.0

    def apply_bcs(self, u, g: Grid2D):
        if self.bound_cond == "REFLECT":
            bc_reflect(u[0], u[1])


def bc_reflect(phi, chi):
    # Reflective boundary conditions
    phi[0, :] = 0.0
    phi[-1, :] = 0.0
    phi[:, 0] = 0.0
    phi[:, -1] = 0.0

    chi[0, :] = (4.0 * chi[1, :] - chi[2, :]) / 3.0
    chi[-1, :] = (4.0 * chi[-2, :] - chi[-3, :]) / 3.0
    chi[:, 0] = (4.0 * chi[:, 1] - chi[:, 2]) / 3.0
    chi[:, -1] = (4.0 * chi[:, -2] - chi[:, -3]) / 3.0

def bc_sommerfeld(dtf, f, dxf, dyf, falloff, ngz, x, y, Nx, Ny):
    eps = 1e-12
    # Use face normals rather than radial vectors for Sommerfeld projection.
    for j in range(Ny):
        for i in range(ngz):
            # xmin boundary
            r = max(np.hypot(x[i], y[j]), eps)
            dtf[i, j] = dxf[i, j] - falloff * f[i, j] / r
        for i in range(Nx - ngz, Nx):
            # xmax boundary
            r = max(np.hypot(x[i], y[j]), eps)
            dtf[i, j] = -dxf[i, j] - falloff * f[i, j] / r

    for i in range(Nx):
        for j in range(ngz):
            # ymin boundary
            r = max(np.hypot(x[i], y[j]), eps)
            dtf[i, j] = dyf[i, j] - falloff * f[i, j] / r
        for j in range(Ny - ngz, Ny):
            # ymax boundary
            r = max(np.hypot(x[i], y[j]), eps)
            dtf[i, j] = -dyf[i, j] - falloff * f[i, j] / r


def rk2(eqs, g, dt, x, y, fltr):
    nu = eqs.u.shape[0]
    up = np.empty_like(eqs.u)
    k1 = np.empty_like(eqs.u)
    if eqs.apply_bc == BCType.FUNCTION:
        eqs.apply_bcs(eqs.u, g)
    eqs.rhs(k1, eqs.u ,x, y, g)
    # Apply KO filter as dissipative RHS term (stage 1)
    for i in range(nu):
        k1[i] += fltr.apply(eqs.u[i])

    up[:] = eqs.u + 0.5 * dt * k1
    if eqs.apply_bc == BCType.FUNCTION:
        eqs.apply_bcs(up, g)
    eqs.rhs(k1, up, x, y, g)
    # Apply KO filter as dissipative RHS term (stage 2)
    for i in range(nu):
        k1[i] += fltr.apply(up[i])

    eqs.u[:] = eqs.u + dt * k1
    if eqs.apply_bc == BCType.FUNCTION:
        eqs.apply_bcs(eqs.u, g)


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
    if eqs.apply_bc == BCType.FUNCTION:
        eqs.apply_bcs(eqs.u, g)

    output_dir = params["output_dir"]
    output_interval = params["output_interval"]
    os.makedirs(output_dir, exist_ok=True)

    dt = params["cfl"] * dx

    # Setup KO filter
    fltr = KreissOligerFilterO6_2D(dx, dy, sigma=params.get("ko_sigma", 0.1), filter_boundary=True)

    time = 0.0
    func_names = ["phi","chi"]
    iox.write_hdf5(0,eqs.u,x,y,func_names,output_dir)

    Nt = params["Nt"]

    for i in range(1, Nt +1):
        rk2(eqs, g, dt, x, y, fltr)
        time += dt
        print(f"Step {i:d} t={time:.2e}")
        if i % output_interval == 0:
            iox.write_hdf5(i,eqs.u,x,y,func_names,output_dir)
    iox.write_xdmf(output_dir, Nt, g.shp[0], g.shp[1], func_names, output_interval, dt)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python JaxSolver.py <parfile>")
        sys.exit(1)
    parfile = sys.argv[1]
    main(parfile)
