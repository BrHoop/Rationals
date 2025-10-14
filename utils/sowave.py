import numpy as np
from numba import njit
import sys
import os
from eqs import *
from grid import *


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
        dtphi = dtu[0]
        dtchi = dtu[1]
        phi = u[0]
        chi = u[1]

        dtphi[:] = chi[:]
        dxxphi = g.D2.grad_xx(phi)
        dyyphi = g.D2.grad_yy(phi)
        dtchi[:] = dxxphi[:] + dyyphi[:]

        if self.apply_bc == BCType.RHS and self.bound_cond == "SOMMERFELD":
            # Sommerfeld boundary conditions
            x = g.xi[0]
            y = g.xi[1]
            Nx = g.shp[0]
            Ny = g.shp[1]
            dxphi = g.D1.grad_x(phi)
            dyphi = g.D1.grad_y(phi)
            bc_sommerfeld(dtphi, phi, dxphi, dyphi, 1.0, 1, x, y, Nx, Ny)
            dxchi = g.D1.grad_x(chi)
            dychi = g.D1.grad_y(chi)
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


@njit
def bc_sommerfeld(dtf, f, dxf, dyf, falloff, ngz, x, y, Nx, Ny):
    for j in range(Ny):
        for i in range(ngz):
            # xmin boundary
            inv_r = 1.0 / np.sqrt(x[i] ** 2 + y[j] ** 2)
            dtf[i, j] = (
                -(x[i] * dxf[i, j] + y[j] * dyf[i, j] + falloff * f[i, j]) * inv_r
            )
        for i in range(Nx - ngz, Nx):
            # xmax boundary
            inv_r = 1.0 / np.sqrt(x[i] ** 2 + y[j] ** 2)
            dtf[i, j] = (
                -(x[i] * dxf[i, j] + y[j] * dyf[i, j] + falloff * f[i, j]) * inv_r
            )

    for i in range(Nx):
        for j in range(ngz):
            # ymin boundary
            inv_r = 1.0 / np.sqrt(x[i] ** 2 + y[j] ** 2)
            dtf[i, j] = (
                -(x[i] * dxf[i, j] + y[j] * dyf[i, j] + falloff * f[i, j]) * inv_r
            )
        for j in range(Ny - ngz, Ny):
            # ymax boundary
            inv_r = 1.0 / np.sqrt(x[i] ** 2 + y[j] ** 2)
            dtf[i, j] = (
                -(x[i] * dxf[i, j] + y[j] * dyf[i, j] + falloff * f[i, j]) * inv_r
            )
