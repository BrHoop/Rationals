import os
import sys
import tomllib
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.signal as jsignal
from jax.tree_util import register_pytree_node_class

# 1. PRECISION
jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import utils.ioxdmf as iox
from utils.eqs import Equations
from utils.grid import Grid
from utils.types import BCType

project_root = Path.cwd()

# --- PYTREE STATE (FP64 Version) ---
# We use the same class structure as BFP, but holding raw arrays instead of BFP fields.

@register_pytree_node_class
class WaveState:
    """The full state of the simulation (FP64)."""
    def __init__(self, phi: jnp.ndarray, chi: jnp.ndarray):
        self.phi = phi
        self.chi = chi

    def tree_flatten(self):
        return ((self.phi, self.chi), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

# --- KERNELS & DERIVATIVES ---
def get_stencils(dx, dy):
    k_d1 = jnp.array([[-1.0, 8.0, 0.0, -8.0, 1.0]]) / 12.0
    k_d2 = jnp.array([[-1.0, 16.0, -30.0, 16.0, -1.0]]) / 12.0
    k_ko = jnp.array([[1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0]]) / 64.0
    return {'dx': k_d1.T/dx, 'dy': k_d1/dy, 'dxx': k_d2.T/dx**2, 'dyy': k_d2.T/dy**2, 'ko_x': k_ko.T, 'ko_y': k_ko}

def grad_x(u, dx, kernel):
    dudx = jsignal.convolve2d(u, kernel, mode='same', boundary='fill')
    idx_by_2 = 0.5 / dx
    dudx = dudx.at[0, :].set((-3.0 * u[0, :] + 4.0 * u[1, :] - u[2, :]) * idx_by_2)
    dudx = dudx.at[1, :].set((-u[0, :] + u[2, :]) * idx_by_2)
    dudx = dudx.at[-2, :].set((-u[-3, :] + u[-1, :]) * idx_by_2)
    dudx = dudx.at[-1, :].set((u[-3, :] - 4.0 * u[-2, :] + 3.0 * u[-1, :]) * idx_by_2)
    return dudx

def grad_y(u, dy, kernel):
    dudy = jsignal.convolve2d(u, kernel, mode='same', boundary='fill')
    idy_by_2 = 0.5 / dy
    dudy = dudy.at[:, 0].set((-3.0 * u[:, 0] + 4.0 * u[:, 1] - u[:, 2]) * idy_by_2)
    dudy = dudy.at[:, 1].set((-u[:, 0] + u[:, 2]) * idy_by_2)
    dudy = dudy.at[:, -2].set((-u[:, -3] + u[:, -1]) * idy_by_2)
    dudy = dudy.at[:, -1].set((u[:, -3] - 4.0 * u[:, -2] + 3.0 * u[:, -1]) * idy_by_2)
    return dudy

def grad_xx(u, dx, kernel):
    dxxu = jsignal.convolve2d(u, kernel, mode='same', boundary='fill')
    idx_sq = 1.0 / (dx * dx)
    dxxu = dxxu.at[0, :].set((2.0 * u[0, :] - 5.0 * u[1, :] + 4.0 * u[2, :] - u[3, :]) * idx_sq)
    dxxu = dxxu.at[1, :].set((u[0, :] - 2.0 * u[1, :] + u[2, :]) * idx_sq)
    dxxu = dxxu.at[-2, :].set((u[-3, :] - 2.0 * u[-2, :] + u[-1, :]) * idx_sq)
    dxxu = dxxu.at[-1, :].set((-u[-4, :] + 4.0 * u[-3, :] - 5.0 * u[-2, :] + 2.0 * u[-1, :]) * idx_sq)
    return dxxu

def grad_yy(u, dy, kernel):
    dyyu = jsignal.convolve2d(u, kernel, mode='same', boundary='fill')
    idy_sq = 1.0 / (dy * dy)
    dyyu = dyyu.at[:, 0].set((2.0 * u[:, 0] - 5.0 * u[:, 1] + 4.0 * u[:, 2] - u[:, 3]) * idy_sq)
    dyyu = dyyu.at[:, 1].set((u[:, 0] - 2.0 * u[:, 1] + u[:, 2]) * idy_sq)
    dyyu = dyyu.at[:, -2].set((u[:, -3] - 2.0 * u[:, -2] + u[:, -1]) * idy_sq)
    dyyu = dyyu.at[:, -1].set((-u[:, -4] + 4.0 * u[:, -3] - 5.0 * u[:, -2] + 2.0 * u[:, -1]) * idy_sq)
    return dyyu

# --- KO FILTER (VMAP) ---
def apply_ko_boundaries_x(du, u, sigma, dx):
    smr3, smr2, smr1 = (9./48.)*64.*dx, (43./48.)*64.*dx, (49./48.)*64.*dx
    du = du.at[0, :].set(sigma * (-u[0, :] + 3.0*u[1, :] - 3.0*u[2, :] + u[3, :]) / smr3)
    du = du.at[1, :].set(sigma * (3.0*u[0, :] - 10.0*u[1, :] + 12.0*u[2, :] - 6.0*u[3, :] + u[4, :]) / smr2)
    du = du.at[2, :].set(sigma * (-3.0*u[0, :] + 12.0*u[1, :] - 19.0*u[2, :] + 15.0*u[3, :] - 6.0*u[4, :] + u[5, :]) / smr1)
    du = du.at[-3, :].set(sigma * (u[-6, :] - 6.0*u[-5, :] + 15.0*u[-4, :] - 19.0*u[-3, :] + 12.0*u[-2, :] - 3.0*u[-1, :]) / smr1)
    du = du.at[-2, :].set(sigma * (u[-5, :] - 6.0*u[-4, :] + 12.0*u[-3, :] - 10.0*u[-2, :] + 3.0*u[-1, :]) / smr2)
    du = du.at[-1, :].set(sigma * (u[-4, :] - 3.0*u[-3, :] + 3.0*u[-2, :] - u[-1, :]) / smr3)
    return du

def apply_ko_boundaries_y(du, u, sigma, dy):
    smr3, smr2, smr1 = (9./48.)*64.*dy, (43./48.)*64.*dy, (49./48.)*64.*dy
    du = du.at[:, 0].set(sigma * (-u[:, 0] + 3.0*u[:, 1] - 3.0*u[:, 2] + u[:, 3]) / smr3)
    du = du.at[:, 1].set(sigma * (3.0*u[:, 0] - 10.0*u[:, 1] + 12.0*u[:, 2] - 6.0*u[:, 3] + u[:, 4]) / smr2)
    du = du.at[:, 2].set(sigma * (-3.0*u[:, 0] + 12.0*u[:, 1] - 19.0*u[:, 2] + 15.0*u[:, 3] - 6.0*u[:, 4] + u[:, 5]) / smr1)
    du = du.at[:, -3].set(sigma * (u[:, -6] - 6.0*u[:, -5] + 15.0*u[:, -4] - 19.0*u[:, -3] + 12.0*u[:, -2] - 3.0*u[:, -1]) / smr1)
    du = du.at[:, -2].set(sigma * (u[:, -5] - 6.0*u[:, -4] + 12.0*u[:, -3] - 10.0*u[:, -2] + 3.0*u[:, -1]) / smr2)
    du = du.at[:, -1].set(sigma * (u[:, -4] - 3.0*u[:, -3] + 3.0*u[:, -2] - u[:, -1]) / smr3)
    return du

def ko6_filter_2d(u, dx, dy, sigma, kx, ky):
    def _filter_field(field):
        diss_x = jsignal.convolve2d(field, kx, mode='same', boundary='fill') * (sigma / dx)
        diss_y = jsignal.convolve2d(field, ky, mode='same', boundary='fill') * (sigma / dy)
        diss_x = apply_ko_boundaries_x(diss_x, field, sigma, dx)
        diss_y = apply_ko_boundaries_y(diss_y, field, sigma, dy)
        return diss_x + diss_y
    return jax.vmap(_filter_field)(u)

# --- BOUNDARIES ---
def sommerfeld_rhs(dtf, f, dxf, dyf, radius, ng, falloff=1.0):
    if ng == 0: return dtf
    falloff_term = falloff * f / radius
    dtf = dtf.at[:ng, :].set(dxf[:ng, :] - falloff_term[:ng, :])
    dtf = dtf.at[-ng:, :].set(-dxf[-ng:, :] - falloff_term[-ng:, :])
    dtf = dtf.at[:, :ng].set(dyf[:, :ng] - falloff_term[:, :ng])
    dtf = dtf.at[:, -ng:].set(-dyf[:, -ng:] - falloff_term[:, -ng:])
    return dtf

def apply_reflect(u):
    phi, chi = u[0], u[1]
    phi = phi.at[0, :].set(0.0).at[-1, :].set(0.0).at[:, 0].set(0.0).at[:, -1].set(0.0)
    chi = chi.at[0, :].set((4.0*chi[1, :] - chi[2, :])/3.0).at[-1, :].set((4.0*chi[-2, :] - chi[-3, :])/3.0)
    chi = chi.at[:, 0].set((4.0*chi[:, 1] - chi[:, 2])/3.0).at[:, -1].set((4.0*chi[:, -2] - chi[:, -3])/3.0)
    return jnp.stack((phi, chi), axis=0)

def make_bc_fn(bound_cond):
    if bound_cond != "REFLECT": return None
    @jax.jit
    def apply_bc(u): return apply_reflect(u)
    return apply_bc

# --- RK4 INTEGRATION (Optimized with Checkpointing) ---

@partial(jax.jit, static_argnames=['rhs_fn', 'filter_fn', 'apply_bc_fn'])
def rk4_step_fp64(rhs_fn, filter_fn, dt, state: WaveState, apply_bc_fn=None):
    # Reconstruct the stacked array u from the PyTree state
    u = jnp.stack((state.phi, state.chi), axis=0)

    # Use checkpointing for the math-heavy RHS
    # This ensures memory access patterns match the BFP code's behavior
    @jax.checkpoint
    def get_rhs(field):
        r = rhs_fn(field)
        if filter_fn: r = r + filter_fn(field)
        return r

    k1 = get_rhs(u)
    u2 = u + 0.5 * dt * k1
    if apply_bc_fn: u2 = apply_bc_fn(u2)
    
    k2 = get_rhs(u2)
    u3 = u + 0.5 * dt * k2
    if apply_bc_fn: u3 = apply_bc_fn(u3)
    
    k3 = get_rhs(u3)
    u4 = u + dt * k3
    if apply_bc_fn: u4 = apply_bc_fn(u4)
    
    k4 = get_rhs(u4)
    u_next = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    if apply_bc_fn: u_next = apply_bc_fn(u_next)
    
    # Return as WaveState
    return WaveState(u_next[0], u_next[1])

# --- SETUP CLASSES ---
class ScalarField(Equations):
    def __init__(self, NU, g: Grid, bctype: str):
        apply_bc_type = BCType.FUNCTION if bctype == "REFLECT" else BCType.RHS
        self.bound_cond = bctype
        super().__init__(NU, g, apply_bc_type)
        self.dx, self.dy, self.ng = float(g.dx[0]), float(g.dx[1]), g.nghost
        self.kernels = get_stencils(self.dx, self.dy)
        self.x, self.y = jnp.asarray(g.xi[0]), jnp.asarray(g.xi[1])
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing="ij")
        r_sq = self.X**2 + self.Y**2
        self.radius = jnp.maximum(jnp.sqrt(r_sq), 1.0e-12)
        self.inv_rsq_eps = 1.0 / (r_sq + 1.0e-5)
        self.u = jnp.zeros((NU, *g.shp), dtype=jnp.float64)
        apply_sommerfeld = (self.apply_bc == BCType.RHS and self.bound_cond == "SOMMERFELD")
        self._rhs_fn = self.make_rhs_fn_closure(apply_sommerfeld)
        self._apply_bc_fn = make_bc_fn(self.bound_cond)

    def make_rhs_fn_closure(self, apply_sommerfeld):
        dx, dy, kernels, inv_rsq, rad, ng = self.dx, self.dy, self.kernels, self.inv_rsq_eps, self.radius, self.ng
        @jax.jit
        def rhs(u):
            phi, chi = u[0], u[1]
            dtphi = chi
            dxxphi = grad_xx(phi, dx, kernels['dxx'])
            dyyphi = grad_yy(phi, dy, kernels['dyy'])
            dtchi = dxxphi + dyyphi - jnp.sin(2.0 * phi) * inv_rsq
            if apply_sommerfeld:
                dxphi, dyphi = grad_x(phi, dx, kernels['dx']), grad_y(phi, dy, kernels['dy'])
                dtphi = sommerfeld_rhs(dtphi, phi, dxphi, dyphi, rad, ng)
                dxchi, dychi = grad_x(chi, dx, kernels['dx']), grad_y(chi, dy, kernels['dy'])
                dtchi = sommerfeld_rhs(dtchi, chi, dxchi, dychi, rad, ng)
            return jnp.stack((dtphi, dtchi), axis=0)
        return rhs

    def rhs(self, u, *_, **__): return self._rhs_fn(u)
    def apply_bcs(self, u, *_, **__): return self._apply_bc_fn(u) if self._apply_bc_fn else u
    def initialize(self, g, params):
        x0, y0, sigma = params["id_x0"], params["id_y0"], params["id_sigma"]
        phi0 = jnp.exp(-((self.X - x0)**2 + (self.Y - y0)**2) / (2.0 * sigma**2))
        self.u = self.u.at[0].set(phi0).at[1].set(jnp.zeros_like(phi0))
        if self._apply_bc_fn: self.u = self._apply_bc_fn(self.u)

# --- MAIN LOOP ---dy
def main(parfile: str, output_dir: str):
    with open(parfile, "rb") as f: params = tomllib.load(f)

    g = Grid2D(params)
    eqs = ScalarField(2, g, params["bound_cond"])
    eqs.initialize(g, params)

    dt = params["cfl"] * g.dx[0]
    sigma_ko = params.get("ko_sigma", 0.1)
    
    # Pack initial state into PyTree
    state = WaveState(phi=eqs.u[0], chi=eqs.u[1])

    def filter_closure(u):
        return ko6_filter_2d(u, eqs.dx, eqs.dy, sigma_ko, eqs.kernels['ko_x'], eqs.kernels['ko_y'])

    @jax.jit
    def step_fn(state: WaveState):
        return rk4_step_fp64(eqs._rhs_fn, filter_closure, dt, state, eqs._apply_bc_fn)

    output_interval, Nt = params["output_interval"], params["Nt"]
    func_names = ["phi", "chi"]
    x_np, y_np = np.asarray(g.xi[0]), np.asarray(g.xi[1])
    
    iox.write_hdf5(0, np.array([state.phi, state.chi]), x_np, y_np, func_names, output_dir)

    num_chunks, remainder = Nt // output_interval, Nt % output_interval
    
    @jax.jit
    def evolve_chunk(start_state, _):
        def scan_body(carry, _): return step_fn(carry), None
        final_state, _ = jax.lax.scan(scan_body, start_state, None, length=output_interval)
        return final_state, None

    print(f"Starting FP64 PyTree Evolution: {Nt} steps.")

    current_state = state
    for chunk in range(num_chunks):
        current_state, _ = evolve_chunk(current_state, None)
        step_count = (chunk + 1) * output_interval
        
        u_out = np.array([current_state.phi, current_state.chi])
        print(f"Step {step_count} t={step_count*dt:.2e} ||phi||={np.linalg.norm(current_state.phi):.2e}")
        iox.write_hdf5(step_count, u_out, x_np, y_np, func_names, output_dir)

    if remainder > 0:
        def final_body(s, _): return step_fn(s), None
        current_state, _ = jax.lax.scan(final_body, current_state, None, length=remainder)
    
    iox.write_xdmf(output_dir, Nt, g.shp[0], g.shp[1], func_names, output_interval, dt)

class Grid2D(Grid):
    def __init__(self, params):
        nx, ny = params["Nx"], params["Ny"]
        xmin, xmax, ymin, ymax = params.get("Xmin",0.0), params.get("Xmax",1.0), params.get("Ymin",0.0), params.get("Ymax",1.0)
        dx, dy = (xmax-xmin)/(nx-1), (ymax-ymin)/(ny-1)
        ng = params.get("NGhost", 0)
        nx_eff, ny_eff = nx + 2*ng, ny + 2*ng
        xmin -= ng*dx; xmax += ng*dx; ymin -= ng*dy; ymax += ng*dy
        super().__init__([nx_eff, ny_eff], [np.linspace(xmin, xmax, nx_eff), np.linspace(ymin, ymax, ny_eff)], np.array([dx, dy]), ng)

if __name__ == "__main__":
    if len(sys.argv) > 1: parfile = sys.argv[1]
    else: parfile = (project_root / "Solvers/2D_Wave/FP_64_Jax/params.toml").as_posix()
    output_dir = (project_root / "Solvers/2D_Wave/FP_64_Jax/data").as_posix() if len(sys.argv) <= 2 else sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    main(parfile, output_dir)