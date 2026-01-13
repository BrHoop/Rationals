import jax
import jax.numpy as jnp

# --- 1. MEMORY HELPERS ---

def upsample_block(block_data, target_shape):
    h_ratio = target_shape[-2] // block_data.shape[-2]
    w_ratio = target_shape[-1] // block_data.shape[-1]
    
    if h_ratio == 1 and w_ratio == 1:
        return block_data
        
    return jnp.repeat(jnp.repeat(block_data, h_ratio, axis=-2), w_ratio, axis=-1)

def bfp_mul(a, a_exp, b, b_exp, headroom=6):
    """Standard BFP multiply."""
    res_exp = a_exp + b_exp + (64 - headroom)
    lhs = a << headroom
    
    def mul_hi_s64(x, y):
        # Cast to JAX arrays to ensure .astype works if inputs are Python ints
        x = jnp.asarray(x, dtype=jnp.int64)
        y = jnp.asarray(y, dtype=jnp.int64)
        
        x_u = x.astype(jnp.uint64)
        y_u = y.astype(jnp.uint64)
        
        x0, x1 = x_u & 0xFFFFFFFF, x_u >> 32
        y0, y1 = y_u & 0xFFFFFFFF, y_u >> 32
        
        w0 = x0 * y0
        t  = x1 * y0 + (w0 >> 32)
        w1, w2 = t & 0xFFFFFFFF, t >> 32
        w1 = w1 + x0 * y1
        mulhu = (x1 * y1) + w2 + (w1 >> 32)
        
        # Sign correction
        x_neg = (x < 0)
        y_neg = (y < 0)
        return (mulhu - (x_neg * y_u) - (y_neg * x_u)).astype(jnp.int64)

    res_mant = mul_hi_s64(lhs, b)
    return res_mant, res_exp

def bfp_align_val(val, val_exp, target_exp):
    shift = target_exp - val_exp
    shift = jnp.clip(shift, 0, 63)
    rounding = (1 << jnp.maximum(0, shift - 1)) * (shift > 0)
    return (val + rounding) >> shift

def apply_boundaries_internal(m):
    m = m.at[0:2, :].set(0)
    m = m.at[-2:, :].set(0)
    m = m.at[:, 0:2].set(0)
    m = m.at[:, -2:].set(0)
    return m

# --- 2. 4TH ORDER STENCIL KERNEL ---

def get_derivs_4th(m, e_block):
    e_full = upsample_block(e_block, m.shape)
    
    e_l1 = jnp.roll(e_full, 1, axis=0)
    e_l2 = jnp.roll(e_full, 2, axis=0)
    e_r1 = jnp.roll(e_full, -1, axis=0)
    e_r2 = jnp.roll(e_full, -2, axis=0)
    e_u1 = jnp.roll(e_full, 1, axis=1)
    e_u2 = jnp.roll(e_full, 2, axis=1)
    e_d1 = jnp.roll(e_full, -1, axis=1)
    e_d2 = jnp.roll(e_full, -2, axis=1)
    
    e_max = e_full
    for en in [e_l1, e_l2, e_r1, e_r2, e_u1, e_u2, e_d1, e_d2]:
        e_max = jnp.maximum(e_max, en)
    
    c = bfp_align_val(m, e_full, e_max)
    xl1 = bfp_align_val(jnp.roll(m, 1, axis=0), e_l1, e_max)
    xl2 = bfp_align_val(jnp.roll(m, 2, axis=0), e_l2, e_max)
    xr1 = bfp_align_val(jnp.roll(m, -1, axis=0), e_r1, e_max)
    xr2 = bfp_align_val(jnp.roll(m, -2, axis=0), e_r2, e_max)
    yu1 = bfp_align_val(jnp.roll(m, 1, axis=1), e_u1, e_max)
    yu2 = bfp_align_val(jnp.roll(m, 2, axis=1), e_u2, e_max)
    yd1 = bfp_align_val(jnp.roll(m, -1, axis=1), e_d1, e_max)
    yd2 = bfp_align_val(jnp.roll(m, -2, axis=1), e_d2, e_max)

    sum_x = 16*(xl1 + xr1) - (xl2 + xr2) - 30*c
    sum_y = 16*(yu1 + yd1) - (yu2 + yd2) - 30*c
    lap_m = sum_x + sum_y
    
    d4x = (xl2 + xr2) - 4*(xl1 + xr1) + 6*c
    d4y = (yu2 + yd2) - 4*(yu1 + yd1) + 6*c
    diss_m = d4x + d4y
    
    return lap_m, diss_m, e_max

# --- 3. RK4 HELPERS ---

def compute_rhs(phi_m, phi_e, chi_m, chi_e, scale_lap, scale_diss):
    lap_phi, diss_phi_raw, e_deriv_phi = get_derivs_4th(phi_m, phi_e)
    _, diss_chi_raw, e_deriv_chi = get_derivs_4th(chi_m, chi_e)
    
    lap_term_m, lap_term_e = bfp_mul(lap_phi, e_deriv_phi, scale_lap[0], scale_lap[1])
    k_phi_diss_m, k_phi_diss_e = bfp_mul(diss_phi_raw, e_deriv_phi, scale_diss[0], scale_diss[1])
    k_chi_diss_m, k_chi_diss_e = bfp_mul(diss_chi_raw, e_deriv_chi, scale_diss[0], scale_diss[1])
    
    chi_e_expanded = upsample_block(chi_e, chi_m.shape)
    
    return (chi_m, chi_e_expanded, k_phi_diss_m, k_phi_diss_e), \
           (lap_term_m, lap_term_e, k_chi_diss_m, k_chi_diss_e)

def rk_accumulate(u_m, u_e, k_tuple, dt_scaled):
    t1_m, t1_e, t2_m, t2_e = k_tuple
    
    e_rhs_common = jnp.maximum(t1_e, t2_e)
    rhs_val = bfp_align_val(t1_m, t1_e, e_rhs_common) - bfp_align_val(t2_m, t2_e, e_rhs_common)
    
    delta_m, delta_e = bfp_mul(rhs_val, e_rhs_common, dt_scaled[0], dt_scaled[1])
    
    u_e_full = upsample_block(u_e, u_m.shape)
    e_final = jnp.maximum(u_e_full, delta_e)
    
    u_aligned = bfp_align_val(u_m, u_e_full, e_final)
    d_aligned = bfp_align_val(delta_m, delta_e, e_final)
    
    res_m = u_aligned + d_aligned
    res_m = apply_boundaries_internal(res_m)
    
    return res_m, e_final

# --- 4. NORMALIZATION AND RE-BLOCKING ---

@jax.jit
def normalize_grid(grid):
    m = grid.mantissas
    e = grid.exponents
    # Explicitly use the grid's defined block size
    bh, bw = grid.bh, grid.bw
    
    # --- STEP A: COMPRESSION (If exponents are expanded) ---
    # If exponents are pixel-wise (same shape as mantissas), we must re-block them.
    if e.shape[-2] == m.shape[-2] and e.shape[-1] == m.shape[-1]:
        # 1. Reshape to (..., h_blocks, block_h, w_blocks, block_w)
        m_blk = m.reshape(m.shape[0], m.shape[1]//bh, bh, m.shape[2]//bw, bw)
        e_blk = e.reshape(e.shape[0], e.shape[1]//bh, bh, e.shape[2]//bw, bw)
        
        # 2. Find common exponent per block (Max is safe to prevent overflow)
        e_common = jnp.max(e_blk, axis=(2, 4)) # Shape: (C, H_blk, W_blk)
        
        # 3. Align mantissas to this new block exponent
        # shift = block_exp - pixel_exp.  (Always >= 0 because we took max)
        shift = (e_common[:, :, None, :, None] - e_blk)
        m_aligned = m_blk >> shift
        
        # Set working variables
        m_working = m_aligned
        e_working = e_common
    else:
        # Already blocked, just reshape mantissas for processing
        m_working = m.reshape(m.shape[0], e.shape[1], bh, e.shape[2], bw)
        e_working = e

    # --- STEP B: NORMALIZATION (Maximize Precision) ---
    # Find max absolute magnitude in each block
    mag = jnp.abs(m_working)
    max_val = jnp.max(mag, axis=(2, 4), keepdims=True)
    
    target_bit = 50 
    current_bit = jnp.floor(jnp.log2(jnp.maximum(max_val, 1))).astype(jnp.int64)
    shifts = target_bit - current_bit
    
    # Handle completely zero blocks
    shifts = jnp.where(max_val == 0, 0, shifts)
    
    # Apply Shifts
    # If shift > 0 (headroom), shift left. If shift < 0 (overflow), shift right.
    m_new = jnp.where(shifts >= 0, m_working << shifts, m_working >> (-shifts))
    
    # Update Exponents
    shifts_sq = shifts.squeeze((2, 4))
    e_new = e_working - shifts_sq
    
    # Clamp small exponents
    min_safe_exp = jnp.max(e_new) - 1000
    e_clamped = jnp.maximum(e_new, min_safe_exp)
    
    # Compensate clamp in mantissas
    clamp_shift_broad = (e_clamped - e_new)[:, :, None, :, None]
    m_final = (m_new >> clamp_shift_broad).reshape(m.shape)
    
    return grid.replace(mantissas=m_final, exponents=e_clamped)

# --- 5. MAIN FUSED STEP ---

@jax.jit
def fused_rk4_step(grid_m, grid_e, s_dt, s_lap, s_diss):
    phi, chi = grid_m[0], grid_m[1]
    e_phi, e_chi = grid_e[0], grid_e[1]
    
    dt_half = (s_dt[0], s_dt[1] - 1)
    
    # -- K1 --
    kp1, kc1 = compute_rhs(phi, e_phi, chi, e_chi, s_lap, s_diss)
    phi1, ep1 = rk_accumulate(phi, e_phi, kp1, dt_half)
    chi1, ec1 = rk_accumulate(chi, e_chi, kc1, dt_half)
    
    # -- K2 --
    kp2, kc2 = compute_rhs(phi1, ep1, chi1, ec1, s_lap, s_diss)
    phi2, ep2 = rk_accumulate(phi, e_phi, kp2, dt_half)
    chi2, ec2 = rk_accumulate(chi, e_chi, kc2, dt_half)
    
    # -- K3 --
    kp3, kc3 = compute_rhs(phi2, ep2, chi2, ec2, s_lap, s_diss)
    phi3, ep3 = rk_accumulate(phi, e_phi, kp3, s_dt)
    chi3, ec3 = rk_accumulate(chi, e_chi, kc3, s_dt)
    
    # -- K4 --
    kp4, kc4 = compute_rhs(phi3, ep3, chi3, ec3, s_lap, s_diss)
    
    # -- Final Assembly --
    def sum_ks(k1, k2, k3, k4):
        p1, p1e, n1, n1e = k1
        p2, p2e, n2, n2e = k2
        p3, p3e, n3, n3e = k3
        p4, p4e, n4, n4e = k4
        
        em = jnp.maximum(p1e, jnp.maximum(p2e, jnp.maximum(p3e, p4e)))
        sum_p = bfp_align_val(p1, p1e, em) + (bfp_align_val(p2, p2e, em)<<1) + \
                (bfp_align_val(p3, p3e, em)<<1) + bfp_align_val(p4, p4e, em)
                
        en = jnp.maximum(n1e, jnp.maximum(n2e, jnp.maximum(n3e, n4e)))
        sum_n = bfp_align_val(n1, n1e, en) + (bfp_align_val(n2, n2e, en)<<1) + \
                (bfp_align_val(n3, n3e, en)<<1) + bfp_align_val(n4, n4e, en)
        
        e_final = jnp.maximum(em, en)
        total = bfp_align_val(sum_p, em, e_final) - bfp_align_val(sum_n, en, e_final)
        return total, e_final

    k_sum_phi_m, k_sum_phi_e = sum_ks(kp1, kp2, kp3, kp4)
    k_sum_chi_m, k_sum_chi_e = sum_ks(kc1, kc2, kc3, kc4)
    
    inv_6_mant = 192153584101141162
    inv_6_exp = -60
    
    dt6_m, dt6_e = bfp_mul(s_dt[0], s_dt[1], inv_6_mant, inv_6_exp)
    
    dPhi_m, dPhi_e = bfp_mul(k_sum_phi_m, k_sum_phi_e, dt6_m, dt6_e)
    dChi_m, dChi_e = bfp_mul(k_sum_chi_m, k_sum_chi_e, dt6_m, dt6_e)
    
    e_phi_full = upsample_block(e_phi, phi.shape)
    e_final_phi = jnp.maximum(e_phi_full, dPhi_e)
    phi_new = bfp_align_val(phi, e_phi_full, e_final_phi) + bfp_align_val(dPhi_m, dPhi_e, e_final_phi)
    
    e_chi_full = upsample_block(e_chi, chi.shape)
    e_final_chi = jnp.maximum(e_chi_full, dChi_e)
    chi_new = bfp_align_val(chi, e_chi_full, e_final_chi) + bfp_align_val(dChi_m, dChi_e, e_final_chi)
    
    phi_new = apply_boundaries_internal(phi_new)
    chi_new = apply_boundaries_internal(chi_new)
    
    return jnp.stack([phi_new, chi_new]), jnp.stack([e_final_phi, e_final_chi])