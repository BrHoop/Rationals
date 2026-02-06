import os
import sys
import tomllib
import numpy as np
import time
import shutil
from functools import reduce

import jax
import jax.numpy as jnp
from jax import lax, jit, vmap

# Add utils path for I/O
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
try:
    import utils.ioxdmf as iox
except ImportError:
    print("Warning: utils.ioxdmf not found. File output will be skipped.")
    iox = None

jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 0. BFP48 COMPRESSION UTILS
# ==============================================================================

@jit
def float64_to_bfp48(block_32):
    """
    Compresses using Standard Rounding (Fast & Bias-Neutral). The goal is to produce a
    a block floating point number made up of 3 int16s that you can send from VRAM to the GPU.
    """
    max_val = jnp.max(jnp.abs(block_32))
    
    # Target 47 bits effective magnitude
    scale_exp = jnp.where(
        max_val == 0,
        0.0,
        jnp.floor(jnp.log2((2.0**47 - 1) / max_val))
    )
    exponent = scale_exp.astype(jnp.int32)
    scaled = block_32 * (2.0 ** scale_exp)
    scaled_int = jnp.floor(scaled + 0.5).astype(jnp.int64)
    
    # Split into 3x Int16 parts
    p0 = (scaled_int & 0xFFFF).astype(jnp.int16)
    p1 = ((scaled_int >> 16) & 0xFFFF).astype(jnp.int16)
    p2 = ((scaled_int >> 32) & 0xFFFF).astype(jnp.int16)
    
    mantissas = jnp.stack([p0, p1, p2], axis=-1)

    return mantissas, exponent

@jit
def bfp48_to_float64(mantissas, exponent):
    """
    Reconstructs float64 block from BFP48 format.
    """
    # 1. Reassemble 64-bit integer
    p0 = mantissas[..., 0].astype(jnp.int64) & 0xFFFF
    p1 = mantissas[..., 1].astype(jnp.int64) & 0xFFFF
    p2 = mantissas[..., 2].astype(jnp.int64) & 0xFFFF
    
    combined_int = p0 | (p1 << 16) | (p2 << 32)
    
    # Checks to see if number is negative, and performs sign correction in base 48
    is_neg = combined_int >= (1 << 47)
    combined_int = jnp.where(is_neg, combined_int - (1 << 48), combined_int)
    
    return combined_int * (2.0 ** (-exponent.astype(jnp.float64)))

# ==============================================================================
# 1. JAX KERNELS
# ==============================================================================

def extended_gcd(a, b):
    if a == 0: return b, 0, 1
    d, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return d, x, y

def modinv(a, m):
    d, x, y = extended_gcd(a, m)
    if d != 1: raise ValueError("Modular inverse does not exist")
    return x % m

@jit
def base_extension(regs_in, mods_in, mods_out, garner_consts, basis_mod_out, basis_in, m_prod_in, m_prod_in_mod_out):
    """
    Generic Base Extension (N moduli -> M moduli) with Sign Correction.
    """
    num_in = mods_in.shape[0]
    v = jnp.zeros_like(regs_in)
    v = v.at[0].set(regs_in[0])
    
    for i in range(1, num_in):
        partial_sum = v[0] % mods_in[i]
        m_acc = 1
        for j in range(1, i):
            m_acc = (m_acc * mods_in[j-1]) % mods_in[i]
            term = (v[j] * m_acc) % mods_in[i]
            partial_sum = (partial_sum + term) % mods_in[i]
            
        diff = (regs_in[i] - partial_sum) % mods_in[i]
        v_i = (diff * garner_consts[i, i]) % mods_in[i]
        v = v.at[i].set(v_i)
        
    regs_ext = jnp.einsum('kij,km->mij', v, basis_mod_out)
    
    # Sign Correction (Integer Arithmetic)
    val_reconstructed = jnp.sum(v * basis_in.reshape(-1, 1, 1), axis=0)
    is_negative = val_reconstructed > (m_prod_in // 2)
    correction = is_negative[None, :, :] * m_prod_in_mod_out[:, None, None]
    regs_ext = (regs_ext - correction) % mods_out[:, None, None]
    
    return jnp.concatenate([regs_in, regs_ext], axis=0)

@jit
def garner_reconstruction(residues, mods, garner_consts, basis, m_prod):
    """
    Reconstructs integer value from residues to Float64.
    """
    num_mods = mods.shape[0]
    x = jnp.zeros_like(residues)
    x = x.at[0].set(residues[0])
    
    for i in range(1, num_mods):
        partial_sum = x[0] % mods[i]
        m_acc = 1
        for j in range(1, i):
            m_acc = (m_acc * mods[j-1]) % mods[i]
            term = (x[j] * m_acc) % mods[i]
            partial_sum = (partial_sum + term) % mods[i]
            
        diff = (residues[i] - partial_sum) % mods[i]
        v_i = (diff * garner_consts[i, i]) % mods[i]
        x = x.at[i].set(v_i)
        
    result = jnp.sum(x * basis.reshape(-1, 1, 1), axis=0)
    m_half = m_prod / 2.0
    result = jnp.where(result > m_half, result - m_prod, result)
    return result

@jit
def ozaki_wgmma_kernel(block_in, mods_full, mods_base, mods_ext,
                        garner_consts, basis_mod_ext,
                        basis_full, basis_base,
                        m_prod_full, m_prod_base, m_prod_base_mod_ext,
                        D_lap, D_ko):
    """
    Ozaki Kernel.
    """
    # 1. ALIGNMENT
    max_val = jnp.max(jnp.abs(block_in))
    
    m_half = m_prod_base / 2.0
    HEADROOM_BITS = 0.0
    
    scale_exp = jnp.where(max_val == 0,
                          0.0,
                          jnp.floor(jnp.log2(m_half / max_val)) - HEADROOM_BITS)
    exponent = -scale_exp.astype(jnp.int32)
    
    # "SHIFT UP" happens here.
    # Since M_prod_base is ~62.7 bits, we can scale 'quiet' blocks much higher
    # than if M_prod_base was 55 bits. This preserves 15 bits of extra LSBs.
    scaled_block = block_in * (2.0 ** scale_exp)
    scaled_ints = jnp.floor(scaled_block + 0.5).astype(jnp.int64) 
    
    regs_base = scaled_ints[None, :, :] % mods_base[:, None, None]

    # 2. EXTEND
    regs_full = base_extension(regs_base, mods_base, mods_ext, garner_consts,
                               basis_mod_ext, basis_base, m_prod_base, m_prod_base_mod_ext)

    # 3. COMPUTE
    acc_lap_x = jnp.einsum('krc,kcy->kry', D_lap, regs_full)[:, :, 2:-2]
    acc_ko_x  = jnp.einsum('krc,kcy->kry', D_ko,  regs_full)[:, :, 2:-2]

    acc_lap_y = jnp.einsum('krc,koc->kro', regs_full, D_lap)[:, 2:-2, :]
    acc_ko_y  = jnp.einsum('krc,koc->kro', regs_full, D_ko)[:, 2:-2, :]

    res_lap = (acc_lap_x + acc_lap_y) % mods_full[:, None, None]
    res_ko  = (acc_ko_x  + acc_ko_y)  % mods_full[:, None, None]

    # 4. RECONSTRUCT
    m_prod_full_f = m_prod_full.astype(jnp.float64)
    out_lap_int = garner_reconstruction(res_lap, mods_full, garner_consts, basis_full, m_prod_full_f)
    out_ko_int  = garner_reconstruction(res_ko,  mods_full, garner_consts, basis_full, m_prod_full_f)
    
    out_lap = out_lap_int * (2.0 ** exponent)
    out_ko  = out_ko_int  * (2.0 ** exponent)
    
    return out_lap, out_ko

# ==============================================================================
# 2. CRT CONVERTER
# ==============================================================================

# ==============================================================================
# 2. CRT CONVERTER (Fixed for Overflow)
# ==============================================================================

class CrtFloatConverter:
    def __init__(self):
        # 1. Moduli Definitions
        # Base Moduli: ~62.7 bits (Fits in signed int64).
        self.mods_base_list = [251, 127, 125, 121, 119, 113, 109, 107, 103]
        
        # Extension Moduli: Adds ~13 bits headroom.
        self.mods_ext_list = [101, 97]
        self.mods_full_list = self.mods_base_list + self.mods_ext_list
        
        self.k_full = len(self.mods_full_list)
        self.k_base = len(self.mods_base_list)

        # Numpy Arrays (for JAX usage later)
        mods_base_np = np.array(self.mods_base_list, dtype=np.int64)
        mods_ext_np  = np.array(self.mods_ext_list, dtype=np.int64)
        mods_full_np = np.array(self.mods_full_list, dtype=np.int64)
        
        # Safety Check: Base product MUST fit in JAX int64 (signed)
        # We use Python 'int' for the calculation to check true value
        m_prod_base_py = reduce(lambda x, y: x * y, self.mods_base_list)
        assert m_prod_base_py < (2**63 - 1), "Base Moduli product exceeds int64 max!"
        assert m_prod_base_py > (2**60), "Base Moduli under-utilized (<60 bits)"
        
        # Full product (used for reconstruction scaling)
        # We store as float because it exceeds int64
        m_prod_full_py = reduce(lambda x, y: x * y, self.mods_full_list)
        
        # 2. Garner Constants (Pre-calculated using Python Big-Ints)
        # FIX: Use python int arithmetic, NOT numpy int64 accumulation
        garner_consts = np.zeros((self.k_full, self.k_full), dtype=np.int64)
        
        for i in range(self.k_full):
            # Compute product of previous moduli using Python arbitrary precision
            if i == 0:
                p_val = 1
            else:
                p_val = 1
                for k in range(i):
                    p_val *= int(self.mods_full_list[k])
            
            for j in range(i, self.k_full):
                # Now we can safely compute inverse
                inv_p = modinv(p_val, int(self.mods_full_list[j]))
                garner_consts[i, j] = inv_p

        # 3. Basis for Reconstruction (Full) -> Float64
        basis_full = np.zeros(self.k_full, dtype=np.float64)
        for i in range(self.k_full):
            prod_val = 1
            for k in range(i):
                prod_val *= int(self.mods_full_list[k])
            basis_full[i] = float(prod_val)
            
        # 4. Basis for Sign Check (Base) -> Int64
        basis_base = np.zeros(self.k_base, dtype=np.int64)
        for i in range(self.k_base):
            prod_val = 1
            for k in range(i):
                prod_val *= int(self.mods_base_list[k])
            basis_base[i] = int(prod_val)

        # 5. Basis for Base Extension (Base -> Ext)
        basis_mod_ext = np.zeros((self.k_base, len(mods_ext_np)), dtype=np.int64)
        for i in range(self.k_base):
            # Calculate weight P_i (product of previous base mods)
            weight = 1
            for k in range(i):
                weight *= int(self.mods_base_list[k])
            
            for j in range(len(mods_ext_np)):
                basis_mod_ext[i, j] = weight % int(mods_ext_np[j])
            
        # 6. M_base modulo M_ext
        m_prod_base_mod_ext = np.zeros(len(mods_ext_np), dtype=np.int64)
        for j in range(len(mods_ext_np)):
            m_prod_base_mod_ext[j] = m_prod_base_py % int(mods_ext_np[j])

        # Move to JAX
        self.mods_base = jnp.array(mods_base_np)
        self.mods_ext  = jnp.array(mods_ext_np)
        self.mods_full = jnp.array(mods_full_np)
        self.garner_consts = jnp.array(garner_consts)
        self.basis_mod_ext = jnp.array(basis_mod_ext)
        self.basis_full = jnp.array(basis_full)
        self.basis_base = jnp.array(basis_base)
        self.M_prod_full = jnp.array(float(m_prod_full_py), dtype=jnp.float64)
        self.M_prod_base = jnp.array(m_prod_base_py, dtype=jnp.int64)
        self.M_prod_base_mod_ext = jnp.array(m_prod_base_mod_ext, dtype=jnp.int64)

# ==============================================================================
# 3. SOLVER
# ==============================================================================

class LinearWaveSolver:
    def __init__(self, nx, ny, params):
        xmin, xmax = params.get("Xmin", -5.0), params.get("Xmax", 5.0)
        ymin, ymax = params.get("Ymin", -5.0), params.get("Ymax", 5.0)
        
        self.dx = (xmax - xmin) / (nx - 1)
        self.dt = params["cfl"] * self.dx
        self.sigma = params.get("ko_sigma", 0.05)
        
        x = jnp.linspace(xmin, xmax, nx)
        y = jnp.linspace(ymin, ymax, ny)
        self.X, self.Y = jnp.meshgrid(x, y, indexing="ij")
        
        self.params = params
        self.nx, self.ny = nx, ny
        self.u = jnp.zeros((2, nx, ny), dtype=np.float64)

        # --- Efficient Config ---
        # BFP compresses 32x32 blocks.
        # Compute merges them into 64x64 blocks.
        # Due to 62.7-bit Base Moduli, we have 15 bits slack to absorb differences.
        self.BFP_BLOCK = 32
        self.COMPUTE_BLOCK = 64 
        self.HALO = 2
        
        self.converter = CrtFloatConverter()
        self._build_matrices()
        self._compiled_step = self._build_compiled_step()

    def _build_matrices(self):
        bs = self.COMPUTE_BLOCK
        h = self.HALO
        D_lap = np.zeros((bs, bs + 2*h), dtype=np.int64)
        coeffs_lap = [-1, 16, -30, 16, -1]
        for r in range(bs):
            c = r + h
            D_lap[r, c-2:c+3] = coeffs_lap
            
        D_ko = np.zeros((bs, bs + 2*h), dtype=np.int64)
        coeffs_ko = [1, -4, 6, -4, 1]
        for r in range(bs):
            c = r + h
            D_ko[r, c-2:c+3] = coeffs_ko
            
        mods_np = np.array(self.converter.mods_full_list)
        self.D_lap_mods = jnp.stack([jnp.array(D_lap % m) for m in mods_np])
        self.D_ko_mods  = jnp.stack([jnp.array(D_ko % m)  for m in mods_np])

    def initialize(self):
        x0, y0 = self.params.get("id_x0", 0.0), self.params.get("id_y0", 0.0)
        sigma, amp = self.params.get("id_sigma", 0.1), self.params.get("id_amp", 1.0)
        r2 = (self.X - x0)**2 + (self.Y - y0)**2
        phi0 = amp * jnp.exp(-r2 / (sigma**2))
        chi0 = jnp.zeros_like(phi0)
        self.u = jnp.stack([phi0, chi0])

    def _build_compiled_step(self):
        c = self.converter
        d_lap = self.D_lap_mods
        d_ko = self.D_ko_mods
        dx = self.dx
        sigma = self.sigma
        nx, ny = self.nx, self.ny
        
        bfp_bs = self.BFP_BLOCK
        cmp_bs = self.COMPUTE_BLOCK
        
        nbx_bfp, nby_bfp = nx // bfp_bs, ny // bfp_bs
        nbx_cmp, nby_cmp = nx // cmp_bs, ny // cmp_bs
        
        patch_size = cmp_bs + 4
        
        @jit
        def _step_internal(u_current):
            # 1. Compress
            u_reshaped = u_current.reshape(2, nbx_bfp, bfp_bs, nby_bfp, bfp_bs)
            u_reshaped = u_reshaped.transpose(0, 1, 3, 2, 4)
            mantissas, exponents = vmap(vmap(vmap(float64_to_bfp48)))(u_reshaped)
            
            # 2. Decompress
            u_recon_blocks = vmap(vmap(vmap(bfp48_to_float64)))(mantissas, exponents)
            u_recon = u_recon_blocks.transpose(0, 1, 3, 2, 4).reshape(2, nx, ny)
            phi, chi = u_recon[0], u_recon[1]

            # 3. Compute (64x64 blocks)
            pad_w = ((2, 2), (2, 2))
            phi_pad = jnp.pad(phi, pad_w, mode='wrap')
            chi_pad = jnp.pad(chi, pad_w, mode='wrap')
            
            phi_patches = lax.conv_general_dilated_patches(
                phi_pad[None, :, :, None], (patch_size, patch_size), (cmp_bs, cmp_bs), padding="VALID",
                dimension_numbers=('NHWC', 'OIHW', 'NHWC')
            )
            chi_patches = lax.conv_general_dilated_patches(
                chi_pad[None, :, :, None], (patch_size, patch_size), (cmp_bs, cmp_bs), padding="VALID",
                dimension_numbers=('NHWC', 'OIHW', 'NHWC')
            )
            
            phi_blks = phi_patches.reshape(-1, patch_size, patch_size)
            chi_blks = chi_patches.reshape(-1, patch_size, patch_size)
            
            wgmma_fn = lambda b: ozaki_wgmma_kernel(
                b, c.mods_full, c.mods_base, c.mods_ext,
                c.garner_consts, c.basis_mod_ext, c.basis_full, c.basis_base,
                c.M_prod_full, c.M_prod_base, c.M_prod_base_mod_ext,
                d_lap, d_ko
            )
            
            lap_phi, diss_phi = vmap(wgmma_fn)(phi_blks)
            _, diss_chi = vmap(wgmma_fn)(chi_blks)
            
            lap_phi = lap_phi / (12.0 * dx**2)
            diss_phi = diss_phi * (-sigma / 16.0)
            diss_chi = diss_chi * (-sigma / 16.0)
            
            chi_centers = chi_blks[:, 2:-2, 2:-2]
            dt_phi_blks = chi_centers + diss_phi
            dt_chi_blks = lap_phi + diss_chi
            
            # 4. Col2Im
            dt_phi = dt_phi_blks.reshape(nbx_cmp, nby_cmp, cmp_bs, cmp_bs)
            dt_phi = dt_phi.transpose(0, 2, 1, 3).reshape(nbx_cmp*cmp_bs, nby_cmp*cmp_bs)
            dt_chi = dt_chi_blks.reshape(nbx_cmp, nby_cmp, cmp_bs, cmp_bs)
            dt_chi = dt_chi.transpose(0, 2, 1, 3).reshape(nbx_cmp*cmp_bs, nby_cmp*cmp_bs)
            
            return jnp.stack([dt_phi, dt_chi])
        return _step_internal
    
    # ... (rk4_step, get_stepper same as before)
    def rk4_step(self, u):
        dt = self.dt
        k1 = self._compiled_step(u)
        k2 = self._compiled_step(u + 0.5 * dt * k1)
        k3 = self._compiled_step(u + 0.5 * dt * k2)
        k4 = self._compiled_step(u + dt * k3)
        u_next = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        u_next = u_next.at[:, 0:2, :].set(0.0)
        u_next = u_next.at[:, -2:, :].set(0.0)
        u_next = u_next.at[:, :, 0:2].set(0.0)
        u_next = u_next.at[:, :, -2:].set(0.0)
        return u_next

    def get_stepper(self, steps_per_call):
        @jit
        def run_scan(u_init):
            def body(u, _):
                return self.rk4_step(u), None
            u_final, _ = lax.scan(body, u_init, length=steps_per_call)
            return u_final
        return run_scan

# ==============================================================================
# 4. MAIN RUNNER (Same as before)
# ==============================================================================

def main(parfile, output_dir):
    if not os.path.exists(parfile):
        print(f"Error: Parameter file {parfile} not found.")
        return

    with open(parfile, "rb") as f:
        params = tomllib.load(f)
        
    nx, ny = params["Nx"], params["Ny"]
    
    # Must be divisible by 64 (Compute block)
    if nx % 64 != 0 or ny % 64 != 0:
        print("Error: Nx and Ny must be divisible by 64.")
        return

    nt, out_int = params["Nt"], params["output_interval"]
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    sim = LinearWaveSolver(nx, ny, params)
    sim.initialize()
    
    step_fn = sim.get_stepper(out_int)
    names = ["phi", "chi"]
    
    print(f"Starting JAX Ozaki (BFP48 + 64x64 Compute with Shift-Up Optimization) | Grid={nx}x{ny}")
    sim.u = step_fn(sim.u).block_until_ready()
    print("Running...")
    
    sim.initialize()
    start_time = time.time()
    
    for s in range(out_int, nt + 1, out_int):
        sim.u = step_fn(sim.u)
        sim.u.block_until_ready()
        
        u_np = np.array(sim.u)
        phi_norm = np.linalg.norm(u_np[0])
        elapsed = time.time() - start_time
        print(f"Step {s} | Time: {s*sim.dt:.3f} | L2 Norm: {phi_norm:.6e} | Wall: {elapsed:.2f}s")
        
        if iox:
            iox.write_hdf5(s, u_np, np.array(sim.X[:,0]), np.array(sim.Y[0,:]), names, output_dir)
            
    if iox:
        iox.write_xdmf(output_dir, nt, nx, ny, names, out_int, sim.dt)
    
    print("Simulation Complete.")

if __name__ == "__main__":
    par = sys.argv[1] if len(sys.argv) > 1 else "params.toml"
    out = sys.argv[2] if len(sys.argv) > 2 else "data_ozaki_bfp"
    main(par, out)