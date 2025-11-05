import os
import sys
import tomllib

from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import utils.ioxdmf as iox  # noqa: E402
from utils.eqs import Equations  # noqa: E402
from utils.grid import Grid  # noqa: E402
from utils.types import BCType  # noqa: E402
from utils.fixed_point import fixed_point  # noqa: E402


def make_fixed_point_ops(frac_bits: int) -> SimpleNamespace:
    """
    Create a collection of fixed-point helper routines backed by JAX arrays.
    """
    base = fixed_point(frac_bits)
    scale = base.SCALE
    half = base.HALF

    def to_fixed_scalar(value):
        return base.to_fixed_scalar(value)

    def from_fixed_scalar(value):
        return base.from_fixed_scalar(value)

    def to_fixed_array(values):
        return jnp.asarray(base.to_fixed_array(np.asarray(values)), dtype=jnp.int64)

    def from_fixed_array(values):
        values = jnp.asarray(values, dtype=jnp.int64)
        return values.astype(jnp.float64) / scale

    def fixed_mul(a, b):
        a_arr = jnp.asarray(a, dtype=jnp.int64)
        b_arr = jnp.asarray(b, dtype=jnp.int64)
        return jnp.right_shift(a_arr * b_arr + half, frac_bits)

    def fixed_div(num, den):
        num_arr = jnp.asarray(num, dtype=jnp.int64)
        den_arr = jnp.asarray(den, dtype=jnp.int64)
        numerator = jnp.left_shift(num_arr, frac_bits) + jnp.floor_divide(den_arr, 2)
        return jnp.floor_divide(numerator, den_arr)

    def fixed_div_int(value, divisor: int):
        value_arr = jnp.asarray(value, dtype=jnp.int64)
        return jnp.floor_divide(value_arr + divisor // 2, divisor)

    def float_to_fixed(values):
        values = jnp.asarray(values, dtype=jnp.float64)
        return jnp.rint(values * scale).astype(jnp.int64)

    return SimpleNamespace(
        frac_bits=frac_bits,
        scale=scale,
        half=half,
        np_impl=base,
        to_fixed_scalar=to_fixed_scalar,
        from_fixed_scalar=from_fixed_scalar,
        to_fixed_array=to_fixed_array,
        from_fixed_array=from_fixed_array,
        fixed_mul=fixed_mul,
        fixed_div=fixed_div,
        fixed_div_int=fixed_div_int,
        float_to_fixed=float_to_fixed,
    )


def make_grad_x(fp, idx_by_2, idx_by_12):
    idx_by_2 = jnp.asarray(idx_by_2, dtype=jnp.int64)
    idx_by_12 = jnp.asarray(idx_by_12, dtype=jnp.int64)

    @jax.jit
    def grad_x(u):
        dudx = jnp.zeros_like(u)
        centered = (
            -u[4:, :]
            + 8 * u[3:-1, :]
            - 8 * u[1:-3, :]
            + u[:-4, :]
        )
        dudx = dudx.at[2:-2, :].set(fp.fixed_mul(centered, idx_by_12))

        dudx = dudx.at[0, :].set(fp.fixed_mul(-3 * u[0, :] + 4 * u[1, :] - u[2, :], idx_by_2))
        dudx = dudx.at[1, :].set(fp.fixed_mul(-u[0, :] + u[2, :], idx_by_2))
        dudx = dudx.at[-2, :].set(fp.fixed_mul(-u[-3, :] + u[-1, :], idx_by_2))
        dudx = dudx.at[-1, :].set(fp.fixed_mul(u[-3, :] - 4 * u[-2, :] + 3 * u[-1, :], idx_by_2))
        return dudx

    return grad_x


def make_grad_y(fp, idy_by_2, idy_by_12):
    idy_by_2 = jnp.asarray(idy_by_2, dtype=jnp.int64)
    idy_by_12 = jnp.asarray(idy_by_12, dtype=jnp.int64)

    @jax.jit
    def grad_y(u):
        dudy = jnp.zeros_like(u)
        centered = (
            -u[:, 4:]
            + 8 * u[:, 3:-1]
            - 8 * u[:, 1:-3]
            + u[:, :-4]
        )
        dudy = dudy.at[:, 2:-2].set(fp.fixed_mul(centered, idy_by_12))

        dudy = dudy.at[:, 0].set(fp.fixed_mul(-3 * u[:, 0] + 4 * u[:, 1] - u[:, 2], idy_by_2))
        dudy = dudy.at[:, 1].set(fp.fixed_mul(-u[:, 0] + u[:, 2], idy_by_2))
        dudy = dudy.at[:, -2].set(fp.fixed_mul(-u[:, -3] + u[:, -1], idy_by_2))
        dudy = dudy.at[:, -1].set(fp.fixed_mul(u[:, -3] - 4 * u[:, -2] + 3 * u[:, -1], idy_by_2))
        return dudy

    return grad_y


def make_grad_xx(fp, idx_sq, idx_sq_by_12):
    idx_sq = jnp.asarray(idx_sq, dtype=jnp.int64)
    idx_sq_by_12 = jnp.asarray(idx_sq_by_12, dtype=jnp.int64)

    @jax.jit
    def grad_xx(u):
        dxxu = jnp.zeros_like(u)
        centered = (
            -u[4:, :]
            + 16 * u[3:-1, :]
            - 30 * u[2:-2, :]
            + 16 * u[1:-3, :]
            - u[:-4, :]
        )
        dxxu = dxxu.at[2:-2, :].set(fp.fixed_mul(centered, idx_sq_by_12))

        dxxu = dxxu.at[0, :].set(fp.fixed_mul(2 * u[0, :] - 5 * u[1, :] + 4 * u[2, :] - u[3, :], idx_sq))
        dxxu = dxxu.at[1, :].set(fp.fixed_mul(u[0, :] - 2 * u[1, :] + u[2, :], idx_sq))
        dxxu = dxxu.at[-2, :].set(fp.fixed_mul(u[-3, :] - 2 * u[-2, :] + u[-1, :], idx_sq))
        dxxu = dxxu.at[-1, :].set(fp.fixed_mul(-u[-4, :] + 4 * u[-3, :] - 5 * u[-2, :] + 2 * u[-1, :], idx_sq))
        return dxxu

    return grad_xx


def make_grad_yy(fp, idy_sq, idy_sq_by_12):
    idy_sq = jnp.asarray(idy_sq, dtype=jnp.int64)
    idy_sq_by_12 = jnp.asarray(idy_sq_by_12, dtype=jnp.int64)

    @jax.jit
    def grad_yy(u):
        dyyu = jnp.zeros_like(u)
        centered = (
            -u[:, 4:]
            + 16 * u[:, 3:-1]
            - 30 * u[:, 2:-2]
            + 16 * u[:, 1:-3]
            - u[:, :-4]
        )
        dyyu = dyyu.at[:, 2:-2].set(fp.fixed_mul(centered, idy_sq_by_12))

        dyyu = dyyu.at[:, 0].set(fp.fixed_mul(2 * u[:, 0] - 5 * u[:, 1] + 4 * u[:, 2] - u[:, 3], idy_sq))
        dyyu = dyyu.at[:, 1].set(fp.fixed_mul(u[:, 0] - 2 * u[:, 1] + u[:, 2], idy_sq))
        dyyu = dyyu.at[:, -2].set(fp.fixed_mul(u[:, -3] - 2 * u[:, -2] + u[:, -1], idy_sq))
        dyyu = dyyu.at[:, -1].set(fp.fixed_mul(-u[:, -4] + 4 * u[:, -3] - 5 * u[:, -2] + 2 * u[:, -1], idy_sq))
        return dyyu

    return grad_yy


def make_sommerfeld_fn(fp, radius_fixed, ng, falloff=1.0):
    if ng <= 0:
        @jax.jit
        def passthrough(dtf, *_args):
            return dtf

        return passthrough

    radius_fixed = jnp.asarray(radius_fixed, dtype=jnp.int64)
    falloff_fixed = jnp.asarray(fp.to_fixed_scalar(float(falloff)), dtype=jnp.int64)
    eps_fixed = max(fp.to_fixed_scalar(1.0e-12), 1)
    radius_fixed = jnp.maximum(radius_fixed, jnp.asarray(eps_fixed, dtype=jnp.int64))

    @jax.jit
    def sommerfeld(dtf, f, dxf, dyf):
        falloff_term = fp.fixed_div(fp.fixed_mul(falloff_fixed, f), radius_fixed)
        # Use face normals rather than radial vectors for Sommerfeld projection.
        dtf = dtf.at[:ng, :].set(dxf[:ng, :] - falloff_term[:ng, :])
        dtf = dtf.at[-ng:, :].set(-dxf[-ng:, :] - falloff_term[-ng:, :])
        dtf = dtf.at[:, :ng].set(dyf[:, :ng] - falloff_term[:, :ng])
        dtf = dtf.at[:, -ng:].set(-dyf[:, -ng:] - falloff_term[:, -ng:])
        return dtf

    return sommerfeld


def make_reflect_fn(fp):
    three = jnp.asarray(fp.to_fixed_scalar(3.0), dtype=jnp.int64)

    @jax.jit
    def apply_reflect(u):
        phi = u[0]
        chi = u[1]

        phi = phi.at[0, :].set(0)
        phi = phi.at[-1, :].set(0)
        phi = phi.at[:, 0].set(0)
        phi = phi.at[:, -1].set(0)

        chi = chi.at[0, :].set(fp.fixed_div(4 * chi[1, :] - chi[2, :], three))
        chi = chi.at[-1, :].set(fp.fixed_div(4 * chi[-2, :] - chi[-3, :], three))
        chi = chi.at[:, 0].set(fp.fixed_div(4 * chi[:, 1] - chi[:, 2], three))
        chi = chi.at[:, -1].set(fp.fixed_div(4 * chi[:, -2] - chi[:, -3], three))

        return jnp.stack((phi, chi), axis=0)

    return apply_reflect


def make_filter_fn(fp, dx, dy, sigma_fixed, filter_boundary=True):
    np_fp = fp.np_impl
    dx = int(dx)
    dy = int(dy)
    sigma = int(sigma_fixed)

    sigma_arr = jnp.asarray(sigma, dtype=jnp.int64)
    factor_x = jnp.asarray(np_fp.fixed_div(sigma, 64 * dx), dtype=jnp.int64)
    factor_y = jnp.asarray(np_fp.fixed_div(sigma, 64 * dy), dtype=jnp.int64)

    if filter_boundary:
        smr3 = jnp.asarray(np_fp.fixed_div(np_fp.to_fixed_scalar(9.0), 48 * 64 * dx), dtype=jnp.int64)
        smr2 = jnp.asarray(np_fp.fixed_div(np_fp.to_fixed_scalar(43.0), 48 * 64 * dx), dtype=jnp.int64)
        smr1 = jnp.asarray(np_fp.fixed_div(np_fp.to_fixed_scalar(49.0), 48 * 64 * dx), dtype=jnp.int64)
        spr3 = smr3
        spr2 = smr2
        spr1 = smr1

        tmr3 = jnp.asarray(np_fp.fixed_div(np_fp.to_fixed_scalar(9.0), 48 * 64 * dy), dtype=jnp.int64)
        tmr2 = jnp.asarray(np_fp.fixed_div(np_fp.to_fixed_scalar(43.0), 48 * 64 * dy), dtype=jnp.int64)
        tmr1 = jnp.asarray(np_fp.fixed_div(np_fp.to_fixed_scalar(49.0), 48 * 64 * dy), dtype=jnp.int64)
        tpr3 = tmr3
        tpr2 = tmr2
        tpr1 = tmr1
    else:
        smr3 = smr2 = smr1 = spr3 = spr2 = spr1 = None
        tmr3 = tmr2 = tmr1 = tpr3 = tpr2 = tpr1 = None

    @jax.jit
    def ko6_filter_x(u):
        du = jnp.zeros_like(u)
        stencil = (
            u[:-6, :]
            - 6 * u[1:-5, :]
            + 15 * u[2:-4, :]
            - 20 * u[3:-3, :]
            + 15 * u[4:-2, :]
            - 6 * u[5:-1, :]
            + u[6:, :]
        )
        du = du.at[3:-3, :].set(fp.fixed_mul(stencil, factor_x))

        if filter_boundary:
            du = du.at[0, :].set(
                fp.fixed_div(fp.fixed_mul(sigma_arr, -u[0, :] + 3 * u[1, :] - 3 * u[2, :] + u[3, :]), smr3)
            )
            du = du.at[1, :].set(
                fp.fixed_div(
                    fp.fixed_mul(sigma_arr, 3 * u[0, :] - 10 * u[1, :] + 12 * u[2, :] - 6 * u[3, :] + u[4, :]),
                    smr2,
                )
            )
            du = du.at[2, :].set(
                fp.fixed_div(
                    fp.fixed_mul(
                        sigma_arr,
                        -3 * u[0, :] + 12 * u[1, :] - 19 * u[2, :] + 15 * u[3, :] - 6 * u[4, :] + u[5, :],
                    ),
                    smr1,
                )
            )
            du = du.at[-3, :].set(
                fp.fixed_div(
                    fp.fixed_mul(
                        sigma_arr,
                        u[-6, :] - 6 * u[-5, :] + 15 * u[-4, :] - 19 * u[-3, :] + 12 * u[-2, :] - 3 * u[-1, :],
                    ),
                    spr1,
                )
            )
            du = du.at[-2, :].set(
                fp.fixed_div(
                    fp.fixed_mul(sigma_arr, u[-5, :] - 6 * u[-4, :] + 12 * u[-3, :] - 10 * u[-2, :] + 3 * u[-1, :]),
                    spr2,
                )
            )
            du = du.at[-1, :].set(
                fp.fixed_div(fp.fixed_mul(sigma_arr, u[-4, :] - 3 * u[-3, :] + 3 * u[-2, :] - u[-1, :]), spr3)
            )

        return du

    @jax.jit
    def ko6_filter_y(u):
        du = jnp.zeros_like(u)
        stencil = (
            u[:, :-6]
            - 6 * u[:, 1:-5]
            + 15 * u[:, 2:-4]
            - 20 * u[:, 3:-3]
            + 15 * u[:, 4:-2]
            - 6 * u[:, 5:-1]
            + u[:, 6:]
        )
        du = du.at[:, 3:-3].set(fp.fixed_mul(stencil, factor_y))

        if filter_boundary:
            du = du.at[:, 0].set(
                fp.fixed_div(fp.fixed_mul(sigma_arr, -u[:, 0] + 3 * u[:, 1] - 3 * u[:, 2] + u[:, 3]), tmr3)
            )
            du = du.at[:, 1].set(
                fp.fixed_div(
                    fp.fixed_mul(sigma_arr, 3 * u[:, 0] - 10 * u[:, 1] + 12 * u[:, 2] - 6 * u[:, 3] + u[:, 4]),
                    tmr2,
                )
            )
            du = du.at[:, 2].set(
                fp.fixed_div(
                    fp.fixed_mul(
                        sigma_arr,
                        -3 * u[:, 0] + 12 * u[:, 1] - 19 * u[:, 2] + 15 * u[:, 3] - 6 * u[:, 4] + u[:, 5],
                    ),
                    tmr1,
                )
            )
            du = du.at[:, -3].set(
                fp.fixed_div(
                    fp.fixed_mul(
                        sigma_arr,
                        u[:, -6] - 6 * u[:, -5] + 15 * u[:, -4] - 19 * u[:, -3] + 12 * u[:, -2] - 3 * u[:, -1],
                    ),
                    tpr1,
                )
            )
            du = du.at[:, -2].set(
                fp.fixed_div(
                    fp.fixed_mul(sigma_arr, u[:, -5] - 6 * u[:, -4] + 12 * u[:, -3] - 10 * u[:, -2] + 3 * u[:, -1]),
                    tpr2,
                )
            )
            du = du.at[:, -1].set(
                fp.fixed_div(fp.fixed_mul(sigma_arr, u[:, -4] - 3 * u[:, -3] + 3 * u[:, -2] - u[:, -1]), tpr3)
            )
        return du

    @jax.jit
    def apply_filter(u):
        return jax.vmap(lambda field: ko6_filter_x(field) + ko6_filter_y(field))(u)

    return apply_filter


def make_rhs_fn(fp, grad_x, grad_y, grad_xx, grad_yy, inv_rsq_eps, sommerfeld_fn, apply_sommerfeld):
    inv_rsq_eps = jnp.asarray(inv_rsq_eps, dtype=jnp.float64)

    @jax.jit
    def rhs(u):
        phi = u[0]
        chi = u[1]

        dtphi = chi
        dxxphi = grad_xx(phi)
        dyyphi = grad_yy(phi)

        phi_float = fp.from_fixed_array(phi)
        nonlinear = jnp.sin(2.0 * phi_float) * inv_rsq_eps
        # nonlinear = 0
        dtchi = dxxphi + dyyphi - fp.float_to_fixed(nonlinear)

        if apply_sommerfeld:
            dxphi = grad_x(phi)
            dyphi = grad_y(phi)
            dxchi = grad_x(chi)
            dychi = grad_y(chi)
            dtphi = sommerfeld_fn(dtphi, phi, dxphi, dyphi)
            dtchi = sommerfeld_fn(dtchi, chi, dxchi, dychi)

        return jnp.stack((dtphi, dtchi), axis=0)

    return rhs


def make_rk2_step(fp, rhs_fn, filter_fn, dt_fixed, apply_bc_fn=None):
    dt_int = int(dt_fixed)
    half_dt_int = fp.np_impl.fixed_mul(dt_int, fp.np_impl.to_fixed_scalar(0.5))

    dt_arr = jnp.asarray(dt_int, dtype=jnp.int64)
    half_dt_arr = jnp.asarray(half_dt_int, dtype=jnp.int64)

    @jax.jit
    def step(u):
        k1 = rhs_fn(u) + filter_fn(u)
        up = u + fp.fixed_mul(k1, half_dt_arr)
        if apply_bc_fn is not None:
            up = apply_bc_fn(up)

        k2 = rhs_fn(up) + filter_fn(up)
        u_new = u + fp.fixed_mul(k2, dt_arr)
        if apply_bc_fn is not None:
            u_new = apply_bc_fn(u_new)
        return u_new

    return step


class Grid2D(Grid):
    """
    Fixed-point 2D grid using integer coordinates.
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
        self.frac_bits = params["frac_bits"]


class ScalarField(Equations):
    U_PHI = 0
    U_CHI = 1

    def __init__(self, NU, g: Grid2D, bctype: str):
        if bctype == "SOMMERFELD":
            apply_bc = BCType.RHS
        elif bctype == "REFLECT":
            apply_bc = BCType.FUNCTION
        else:
            raise ValueError("Invalid boundary condition type. Use 'SOMMERFELD' or 'REFLECT'.")

        super().__init__(NU, g, apply_bc)

        self.bound_cond = bctype
        self.fp = make_fixed_point_ops(g.frac_bits)
        self.u = jnp.zeros((NU, *g.shp), dtype=jnp.int64)

        self.dx = int(g.dx[0])
        self.dy = int(g.dx[1])
        self.ng = g.nghost

        self.x_fixed = jnp.asarray(g.xi[0], dtype=jnp.int64)
        self.y_fixed = jnp.asarray(g.xi[1], dtype=jnp.int64)
        self.x_float = self.fp.from_fixed_array(self.x_fixed)
        self.y_float = self.fp.from_fixed_array(self.y_fixed)
        self.X_fixed, self.Y_fixed = jnp.meshgrid(self.x_fixed, self.y_fixed, indexing="ij") #This is the difference
        self.X_float, self.Y_float = jnp.meshgrid(self.x_float, self.y_float, indexing="ij")

        r_sq = self.X_float**2 + self.Y_float**2
        radius_float = jnp.maximum(jnp.sqrt(r_sq), 1.0e-12)
        eps_fixed = max(self.fp.to_fixed_scalar(1.0e-12), 1)
        self.radius_fixed = jnp.maximum(self.fp.float_to_fixed(radius_float), jnp.asarray(eps_fixed, dtype=jnp.int64))
        self.inv_rsq_eps = 1.0 / (r_sq + 1.0e-5)

        idx_by_2 = self.fp.np_impl.fixed_div(self.fp.to_fixed_scalar(1.0), 2 * self.dx)
        idx_by_12 = self.fp.np_impl.fixed_div(self.fp.to_fixed_scalar(1.0), 12 * self.dx)
        idy_by_2 = self.fp.np_impl.fixed_div(self.fp.to_fixed_scalar(1.0), 2 * self.dy)
        idy_by_12 = self.fp.np_impl.fixed_div(self.fp.to_fixed_scalar(1.0), 12 * self.dy)

        dx_sq = self.fp.np_impl.fixed_mul(self.dx, self.dx)
        dy_sq = self.fp.np_impl.fixed_mul(self.dy, self.dy)
        idx_sq = self.fp.np_impl.fixed_div(self.fp.to_fixed_scalar(1.0), dx_sq)
        idy_sq = self.fp.np_impl.fixed_div(self.fp.to_fixed_scalar(1.0), dy_sq)
        idx_sq_by_12 = self.fp.np_impl.fixed_div(idx_sq, self.fp.to_fixed_scalar(12.0))
        idy_sq_by_12 = self.fp.np_impl.fixed_div(idy_sq, self.fp.to_fixed_scalar(12.0))

        self.grad_x = make_grad_x(self.fp, idx_by_2, idx_by_12)
        self.grad_y = make_grad_y(self.fp, idy_by_2, idy_by_12)
        self.grad_xx = make_grad_xx(self.fp, idx_sq, idx_sq_by_12)
        self.grad_yy = make_grad_yy(self.fp, idy_sq, idy_sq_by_12)

        apply_sommerfeld = self.apply_bc == BCType.RHS and self.bound_cond == "SOMMERFELD"
        self._sommerfeld_fn = make_sommerfeld_fn(
            self.fp,
            self.radius_fixed,
            self.ng,
            falloff=1.0,
        )
        self._rhs_fn = make_rhs_fn(
            self.fp,
            self.grad_x,
            self.grad_y,
            self.grad_xx,
            self.grad_yy,
            self.inv_rsq_eps,
            self._sommerfeld_fn,
            apply_sommerfeld,
        )

        self._apply_bc_fn = None
        if self.bound_cond == "REFLECT":
            self._apply_bc_fn = make_reflect_fn(self.fp)

    def rhs(self, u: jnp.ndarray, *_, **__) -> jnp.ndarray:
        return self._rhs_fn(u)

    def initialize(self, g: Grid, params):
        x0 = params["id_x0"]
        y0 = params["id_y0"]
        sigma = params["id_sigma"]
        amp = params["id_amp"]

        profile = amp * jnp.exp(-((self.X_float - x0) ** 2 + (self.Y_float - y0) ** 2) / (2.0 * sigma * sigma))
        phi0 = self.fp.float_to_fixed(profile)
        chi0 = jnp.zeros_like(phi0)

        self.u = self.u.at[self.U_PHI].set(phi0)
        self.u = self.u.at[self.U_CHI].set(chi0)

        if self._apply_bc_fn is not None:
            self.u = self._apply_bc_fn(self.u)

    def apply_bcs(self, u: jnp.ndarray, *_args, **_kwargs) -> jnp.ndarray:
        if self._apply_bc_fn is None:
            return u
        return self._apply_bc_fn(u)


def main(parfile: str):
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    g = Grid2D(params)
    eqs = ScalarField(2, g, params["bound_cond"])
    eqs.initialize(g, params)

    cfl_fixed = g.f.to_fixed_scalar(params["cfl"])
    dt_fixed = g.f.fixed_mul(cfl_fixed, g.dx[0])

    sigma_fixed = g.f.to_fixed_scalar(params.get("ko_sigma", 0.01))
    filter_fn = make_filter_fn(eqs.fp, g.dx[0], g.dx[1], sigma_fixed, filter_boundary=True)
    step_fn = make_rk2_step(eqs.fp, eqs.rhs, filter_fn, dt_fixed, eqs._apply_bc_fn)

    output_dir = params["output_dir"]
    output_interval = params["output_interval"]
    os.makedirs(output_dir, exist_ok=True)

    func_names = ["phi", "chi"]
    x_float = np.asarray(eqs.x_float)
    y_float = np.asarray(eqs.y_float)

    u_float = np.asarray(eqs.fp.from_fixed_array(eqs.u))
    iox.write_hdf5(0, u_float, x_float, y_float, func_names, output_dir)

    Nt = params["Nt"]
    time_fixed = 0

    for step in range(1, Nt + 1):
        eqs.u = step_fn(eqs.u)
        time_fixed += dt_fixed
        print(f"Step {step:d} t={g.f.from_fixed_scalar(time_fixed):.2e}")

        if step % output_interval == 0:
            u_float = np.asarray(eqs.fp.from_fixed_array(eqs.u))
            iox.write_hdf5(step, u_float, x_float, y_float, func_names, output_dir)

    iox.write_xdmf(
        output_dir,
        Nt,
        g.shp[0],
        g.shp[1],
        func_names,
        output_interval,
        g.f.from_fixed_scalar(dt_fixed),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python JaxSolver.py <parfile>")
        sys.exit(1)
    main(sys.argv[1])
