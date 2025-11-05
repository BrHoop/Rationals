import os
import sys
import tomllib

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import utils.ioxdmf as iox
from utils.eqs import Equations
from utils.grid import Grid
from utils.types import BCType


def grad_x(u: jnp.ndarray, dx: float) -> jnp.ndarray:
    idx_by_2 = 0.5 / dx
    idx_by_12 = 1.0 / (12.0 * dx)
    dudx = jnp.zeros_like(u)
    dudx = dudx.at[2:-2, :].set(
        (-u[4:, :] + 8.0 * u[3:-1, :] - 8.0 * u[1:-3, :] + u[:-4, :]) * idx_by_12
    )
    dudx = dudx.at[0, :].set((-3.0 * u[0, :] + 4.0 * u[1, :] - u[2, :]) * idx_by_2)
    dudx = dudx.at[1, :].set((-u[0, :] + u[2, :]) * idx_by_2)
    dudx = dudx.at[-2, :].set((-u[-3, :] + u[-1, :]) * idx_by_2)
    dudx = dudx.at[-1, :].set((u[-3, :] - 4.0 * u[-2, :] + 3.0 * u[-1, :]) * idx_by_2)
    return dudx


def grad_y(u: jnp.ndarray, dy: float) -> jnp.ndarray:
    idy_by_2 = 0.5 / dy
    idy_by_12 = 1.0 / (12.0 * dy)
    dudy = jnp.zeros_like(u)
    dudy = dudy.at[:, 2:-2].set(
        (-u[:, 4:] + 8.0 * u[:, 3:-1] - 8.0 * u[:, 1:-3] + u[:, :-4]) * idy_by_12
    )
    dudy = dudy.at[:, 0].set((-3.0 * u[:, 0] + 4.0 * u[:, 1] - u[:, 2]) * idy_by_2)
    dudy = dudy.at[:, 1].set((-u[:, 0] + u[:, 2]) * idy_by_2)
    dudy = dudy.at[:, -2].set((-u[:, -3] + u[:, -1]) * idy_by_2)
    dudy = dudy.at[:, -1].set((u[:, -3] - 4.0 * u[:, -2] + 3.0 * u[:, -1]) * idy_by_2)
    return dudy


def grad_xx(u: jnp.ndarray, dx: float) -> jnp.ndarray:
    idx_sq = 1.0 / (dx * dx)
    idx_sq_by_12 = idx_sq / 12.0
    dxxu = jnp.zeros_like(u)
    dxxu = dxxu.at[2:-2, :].set(
        (-u[4:, :] + 16.0 * u[3:-1, :] - 30.0 * u[2:-2, :] + 16.0 * u[1:-3, :] - u[:-4, :])
        * idx_sq_by_12
    )
    dxxu = dxxu.at[0, :].set((2.0 * u[0, :] - 5.0 * u[1, :] + 4.0 * u[2, :] - u[3, :]) * idx_sq)
    dxxu = dxxu.at[1, :].set((u[0, :] - 2.0 * u[1, :] + u[2, :]) * idx_sq)
    dxxu = dxxu.at[-2, :].set((u[-3, :] - 2.0 * u[-2, :] + u[-1, :]) * idx_sq)
    dxxu = dxxu.at[-1, :].set((-u[-4, :] + 4.0 * u[-3, :] - 5.0 * u[-2, :] + 2.0 * u[-1, :]) * idx_sq)
    return dxxu


def grad_yy(u: jnp.ndarray, dy: float) -> jnp.ndarray:
    idy_sq = 1.0 / (dy * dy)
    idy_sq_by_12 = idy_sq / 12.0
    dyyu = jnp.zeros_like(u)
    dyyu = dyyu.at[:, 2:-2].set(
        (-u[:, 4:] + 16.0 * u[:, 3:-1] - 30.0 * u[:, 2:-2] + 16.0 * u[:, 1:-3] - u[:, :-4])
        * idy_sq_by_12
    )
    dyyu = dyyu.at[:, 0].set((2.0 * u[:, 0] - 5.0 * u[:, 1] + 4.0 * u[:, 2] - u[:, 3]) * idy_sq)
    dyyu = dyyu.at[:, 1].set((u[:, 0] - 2.0 * u[:, 1] + u[:, 2]) * idy_sq)
    dyyu = dyyu.at[:, -2].set((u[:, -3] - 2.0 * u[:, -2] + u[:, -1]) * idy_sq)
    dyyu = dyyu.at[:, -1].set((-u[:, -4] + 4.0 * u[:, -3] - 5.0 * u[:, -2] + 2.0 * u[:, -1]) * idy_sq)
    return dyyu


def ko6_filter_x(u: jnp.ndarray, dx: float, sigma: float, filter_boundary: bool) -> jnp.ndarray:
    factor = sigma / (64.0 * dx)
    du = jnp.zeros_like(u)
    du = du.at[3:-3, :].set(
        factor
        * (
            u[:-6, :]
            - 6.0 * u[1:-5, :]
            + 15.0 * u[2:-4, :]
            - 20.0 * u[3:-3, :]
            + 15.0 * u[4:-2, :]
            - 6.0 * u[5:-1, :]
            + u[6:, :]
        )
    )

    if filter_boundary:
        smr3 = (9.0 / 48.0) * 64.0 * dx
        smr2 = (43.0 / 48.0) * 64.0 * dx
        smr1 = (49.0 / 48.0) * 64.0 * dx
        spr3 = smr3
        spr2 = smr2
        spr1 = smr1

        du = du.at[0, :].set(sigma * (-u[0, :] + 3.0 * u[1, :] - 3.0 * u[2, :] + u[3, :]) / smr3)
        du = du.at[1, :].set(
            sigma * (3.0 * u[0, :] - 10.0 * u[1, :] + 12.0 * u[2, :] - 6.0 * u[3, :] + u[4, :]) / smr2
        )
        du = du.at[2, :].set(
            sigma
            * (
                -3.0 * u[0, :]
                + 12.0 * u[1, :]
                - 19.0 * u[2, :]
                + 15.0 * u[3, :]
                - 6.0 * u[4, :]
                + u[5, :]
            )
            / smr1
        )
        du = du.at[-3, :].set(
            sigma
            * (
                u[-6, :]
                - 6.0 * u[-5, :]
                + 15.0 * u[-4, :]
                - 19.0 * u[-3, :]
                + 12.0 * u[-2, :]
                - 3.0 * u[-1, :]
            )
            / spr1
        )
        du = du.at[-2, :].set(
            sigma * (u[-5, :] - 6.0 * u[-4, :] + 12.0 * u[-3, :] - 10.0 * u[-2, :] + 3.0 * u[-1, :]) / spr2
        )
        du = du.at[-1, :].set(sigma * (u[-4, :] - 3.0 * u[-3, :] + 3.0 * u[-2, :] - u[-1, :]) / spr3)

    return du


def ko6_filter_y(u: jnp.ndarray, dy: float, sigma: float, filter_boundary: bool) -> jnp.ndarray:
    factor = sigma / (64.0 * dy)
    du = jnp.zeros_like(u)
    du = du.at[:, 3:-3].set(
        factor
        * (
            u[:, :-6]
            - 6.0 * u[:, 1:-5]
            + 15.0 * u[:, 2:-4]
            - 20.0 * u[:, 3:-3]
            + 15.0 * u[:, 4:-2]
            - 6.0 * u[:, 5:-1]
            + u[:, 6:]
        )
    )

    if filter_boundary:
        smr3 = (9.0 / 48.0) * 64.0 * dy
        smr2 = (43.0 / 48.0) * 64.0 * dy
        smr1 = (49.0 / 48.0) * 64.0 * dy
        spr3 = smr3
        spr2 = smr2
        spr1 = smr1

        du = du.at[:, 0].set(sigma * (-u[:, 0] + 3.0 * u[:, 1] - 3.0 * u[:, 2] + u[:, 3]) / smr3)
        du = du.at[:, 1].set(
            sigma * (3.0 * u[:, 0] - 10.0 * u[:, 1] + 12.0 * u[:, 2] - 6.0 * u[:, 3] + u[:, 4]) / smr2
        )
        du = du.at[:, 2].set(
            sigma
            * (
                -3.0 * u[:, 0]
                + 12.0 * u[:, 1]
                - 19.0 * u[:, 2]
                + 15.0 * u[:, 3]
                - 6.0 * u[:, 4]
                + u[:, 5]
            )
            / smr1
        )
        du = du.at[:, -3].set(
            sigma
            * (
                u[:, -6]
                - 6.0 * u[:, -5]
                + 15.0 * u[:, -4]
                - 19.0 * u[:, -3]
                + 12.0 * u[:, -2]
                - 3.0 * u[:, -1]
            )
            / spr1
        )
        du = du.at[:, -2].set(
            sigma * (u[:, -5] - 6.0 * u[:, -4] + 12.0 * u[:, -3] - 10.0 * u[:, -2] + 3.0 * u[:, -1]) / spr2
        )
        du = du.at[:, -1].set(sigma * (u[:, -4] - 3.0 * u[:, -3] + 3.0 * u[:, -2] - u[:, -1]) / spr3)

    return du


def ko_filter_2d(u: jnp.ndarray, dx: float, dy: float, sigma: float, filter_boundary: bool) -> jnp.ndarray:
    return ko6_filter_x(u, dx, sigma, filter_boundary) + ko6_filter_y(u, dy, sigma, filter_boundary)


def sommerfeld_rhs(
    dtf: jnp.ndarray,
    f: jnp.ndarray,
    dxf: jnp.ndarray,
    dyf: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    inv_r: jnp.ndarray,
    ng: int,
    falloff: float = 1.0,
) -> jnp.ndarray:
    if ng == 0:
        return dtf

    update = -(x * dxf + y * dyf + falloff * f) * inv_r
    dtf = dtf.at[:ng, :].set(update[:ng, :])
    dtf = dtf.at[-ng:, :].set(update[-ng:, :])
    dtf = dtf.at[:, :ng].set(update[:, :ng])
    dtf = dtf.at[:, -ng:].set(update[:, -ng:])
    return dtf


def apply_reflect(u: jnp.ndarray) -> jnp.ndarray:
    phi = u[0]
    chi = u[1]

    phi = phi.at[0, :].set(0.0)
    phi = phi.at[-1, :].set(0.0)
    phi = phi.at[:, 0].set(0.0)
    phi = phi.at[:, -1].set(0.0)

    chi = chi.at[0, :].set((4.0 * chi[1, :] - chi[2, :]) / 3.0)
    chi = chi.at[-1, :].set((4.0 * chi[-2, :] - chi[-3, :]) / 3.0)
    chi = chi.at[:, 0].set((4.0 * chi[:, 1] - chi[:, 2]) / 3.0)
    chi = chi.at[:, -1].set((4.0 * chi[:, -2] - chi[:, -3]) / 3.0)

    return jnp.stack((phi, chi), axis=0)


def make_rhs_fn(
    dx: float,
    dy: float,
    inv_rsq_eps: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    inv_r: jnp.ndarray,
    ng: int,
    apply_sommerfeld: bool,
):
    def rhs(u: jnp.ndarray) -> jnp.ndarray:
        phi = u[0]
        chi = u[1]

        dtphi = chi
        dxxphi = grad_xx(phi, dx)
        dyyphi = grad_yy(phi, dy)
        dtchi = dxxphi + dyyphi - jnp.sin(2.0 * phi) * inv_rsq_eps

        if apply_sommerfeld:
            dxphi = grad_x(phi, dx)
            dyphi = grad_y(phi, dy)
            dtphi = sommerfeld_rhs(dtphi, phi, dxphi, dyphi, x, y, inv_r, ng)

            dxchi = grad_x(chi, dx)
            dychi = grad_y(chi, dy)
            dtchi = sommerfeld_rhs(dtchi, chi, dxchi, dychi, x, y, inv_r, ng)

        return jnp.stack((dtphi, dtchi), axis=0)

    return jax.jit(rhs)


def make_bc_fn(bound_cond: str):
    if bound_cond != "REFLECT":
        return None

    @jax.jit
    def apply_bc(u: jnp.ndarray) -> jnp.ndarray:
        return apply_reflect(u)

    return apply_bc


def make_filter_fn(dx: float, dy: float, sigma: float, filter_boundary: bool):
    def apply_filter(u: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda field: ko_filter_2d(field, dx, dy, sigma, filter_boundary))(u)

    return apply_filter


def make_rk2_step(rhs_fn, filter_fn, dt: float, apply_bc_fn=None):
    if apply_bc_fn is None:

        @jax.jit
        def step(u: jnp.ndarray) -> jnp.ndarray:
            k1 = rhs_fn(u) + filter_fn(u)
            up = u + 0.5 * dt * k1
            k2 = rhs_fn(up) + filter_fn(up)
            return u + dt * k2

        return step

    @jax.jit
    def step_with_bc(u: jnp.ndarray) -> jnp.ndarray:
        u_bc = apply_bc_fn(u)
        k1 = rhs_fn(u_bc) + filter_fn(u_bc)
        up = apply_bc_fn(u_bc + 0.5 * dt * k1)
        k2 = rhs_fn(up) + filter_fn(up)
        return apply_bc_fn(u_bc + dt * k2)

    return step_with_bc


class Grid2D(Grid):
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

        nx_eff = nx + 2 * ng
        ny_eff = ny + 2 * ng
        xmin -= ng * dx
        xmax += ng * dx
        ymin -= ng * dy
        ymax += ng * dy

        shp = [nx_eff, ny_eff]
        xi = [np.linspace(xmin, xmax, nx_eff), np.linspace(ymin, ymax, ny_eff)]
        dxn = np.array([dx, dy])

        super().__init__(shp, xi, dxn, ng)


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

        self.dx = float(g.dx[0])
        self.dy = float(g.dx[1])
        self.ng = g.nghost

        self.x = jnp.asarray(g.xi[0])
        self.y = jnp.asarray(g.xi[1])
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing="ij")

        r = jnp.sqrt(self.X * self.X + self.Y * self.Y)
        self.inv_r = jnp.where(r > 0.0, 1.0 / r, 0.0)
        self.inv_rsq_eps = 1.0 / (self.X * self.X + self.Y * self.Y + 1.0e-2)

        self.u = jnp.zeros((NU, *g.shp), dtype=jnp.float64)

        apply_sommerfeld = self.apply_bc == BCType.RHS and self.bound_cond == "SOMMERFELD"
        self._rhs_fn = make_rhs_fn(self.dx, self.dy, self.inv_rsq_eps, self.X, self.Y, self.inv_r, self.ng, apply_sommerfeld)
        self._apply_bc_fn = make_bc_fn(self.bound_cond)

    def rhs(self, u: jnp.ndarray, *_, **__) -> jnp.ndarray:
        return self._rhs_fn(u)

    def initialize(self, g: Grid, params):
        x0 = params["id_x0"]
        y0 = params["id_y0"]
        sigma = params["id_sigma"]

        phi0 = jnp.exp(-((self.X - x0) ** 2 + (self.Y - y0) ** 2) / (2.0 * sigma * sigma))
        chi0 = jnp.zeros_like(phi0)

        self.u = self.u.at[0].set(phi0)
        self.u = self.u.at[1].set(chi0)

        if self._apply_bc_fn is not None:
            self.u = self._apply_bc_fn(self.u)

    def apply_bcs(self, u: jnp.ndarray, *_, **__) -> jnp.ndarray:
        if self._apply_bc_fn is None:
            return u
        return self._apply_bc_fn(u)


def main(parfile: str):
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    g = Grid2D(params)
    eqs = ScalarField(2, g, params["bound_cond"])
    eqs.initialize(g, params)

    dt = params["cfl"] * g.dx[0]
    sigma = params.get("ko_sigma", 0.1) #At most 0.4
    filter_fn = make_filter_fn(eqs.dx, eqs.dy, sigma, False)
    step_fn = make_rk2_step(eqs.rhs, filter_fn, dt, eqs._apply_bc_fn)

    output_dir = params["output_dir"]
    output_interval = params["output_interval"]
    os.makedirs(output_dir, exist_ok=True)

    func_names = ["phi", "chi"]

    x_np = np.asarray(g.xi[0])
    y_np = np.asarray(g.xi[1])
    iox.write_hdf5(0, np.asarray(eqs.u), x_np, y_np, func_names, output_dir)

    Nt = params["Nt"]
    time = 0.0

    for step in range(1, Nt + 1):
        eqs.u = step_fn(eqs.u)
        time += dt
        print(f"Step {step:d} t={time:.2e} ||phi||={np.linalg.norm(eqs.u[0],"fro"):.2e}  ||chi||={np.linalg.norm(eqs.u[1],"fro"):.2e}")

        if step % output_interval == 0:
            iox.write_hdf5(step, np.asarray(eqs.u), x_np, y_np, func_names, output_dir)

    iox.write_xdmf(output_dir, Nt, g.shp[0], g.shp[1], func_names, output_interval, dt)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python JaxSolver.py <parfile>")
        sys.exit(1)
    main(sys.argv[1])
