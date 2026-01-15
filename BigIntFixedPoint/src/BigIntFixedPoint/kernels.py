import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jax_triton import triton_call
import numpy as np
from typing import Tuple, Union

# ==========================================
# 1. Utilities and Helper Functions
# ==========================================

def _bits_of_dtype(dtype: jnp.dtype) -> int:
    dt = jnp.dtype(dtype)
    if dt == jnp.uint8: return 8
    if dt == jnp.uint16: return 16
    if dt == jnp.uint32: return 32
    raise TypeError(f"Supported limb dtypes: uint8/uint16/uint32. Got {dt}.")

def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p

# --- Python-side helpers for data conversion (optional but useful) ---
def convert_int_vec(A: int, L: int, size: jnp.dtype):
    bits = _bits_of_dtype(size)
    mask = (1 << bits) - 1
    out = np.empty((L,), dtype=np.dtype(size))
    for i in range(L):
        out[i] = A & mask
        A >>= bits
    return jnp.asarray(out, dtype=size)

def convert_vec_int(A: jnp.array):
    w = A.dtype.itemsize * 8
    x = 0
    for v in A[::-1]:
        x = (x << w) | int(v)
    return x

# --- Tensor-based helpers for Division Kernel ---
@triton.jit
def _shl_1_tensor(val_tensor, bit_in, L: tl.constexpr):
    carry = bit_in.to(tl.uint32)
    res = val_tensor
    for i in tl.static_range(L):
        mask = (tl.arange(0, L) == i)
        curr = tl.sum(tl.where(mask, res, 0))
        new_limb = (curr << 1) | carry
        next_carry = (curr >> 31) & 1
        res = tl.where(mask, new_limb, res)
        carry = next_carry
    return res

@triton.jit
def _sub_tensor(A_tensor, B_tensor, L: tl.constexpr):
    borrow = tl.zeros((), dtype=tl.uint32)
    res = A_tensor
    for i in tl.static_range(L):
        mask = (tl.arange(0, L) == i)
        a = tl.sum(tl.where(mask, A_tensor, 0))
        b = tl.sum(tl.where(mask, B_tensor, 0))

        a_64 = a.to(tl.uint64)
        b_64 = b.to(tl.uint64)
        bor_64 = borrow.to(tl.uint64)

        rhs = b_64 + bor_64
        borrow_out = (a_64 < rhs)
        diff_64 = a_64 - rhs
        diff = diff_64.to(tl.uint32)

        res = tl.where(mask, diff, res)
        borrow = borrow_out.to(tl.uint32)
    return res, borrow

# ==========================================
# 2. Addition / Subtraction Kernels
# ==========================================

@triton.jit
def gp_combine(a_g, a_p, b_g, b_p):
    out_g = b_g | (b_p & a_g)
    out_p = b_p & a_p
    return out_g, out_p

@triton.jit
def add_limbs_prefixscan_kernel(
    A_ptr, B_ptr, C_ptr, Carry_ptr,
    N: tl.constexpr, L: tl.constexpr,
    STRIDE_N: tl.constexpr, STRIDE_L: tl.constexpr,
    OUT_STRIDE_N: tl.constexpr, OUT_STRIDE_L: tl.constexpr,
    CARRY_STRIDE: tl.constexpr,
    LIMB_BITS: tl.constexpr, BLOCK_L: tl.constexpr,
    INITIAL_CARRY: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    if pid_n >= N: return

    base = 1 << LIMB_BITS
    base_mask = base - 1
    lane = tl.arange(0, BLOCK_L)
    carry_in = tl.full((), INITIAL_CARRY, dtype=tl.int1)

    for l0 in tl.static_range(0, L, BLOCK_L):
        idx = l0 + lane
        in_bounds = idx < L
        a = tl.load(A_ptr + pid_n * STRIDE_N + idx * STRIDE_L, mask=in_bounds, other=0)
        b = tl.load(B_ptr + pid_n * STRIDE_N + idx * STRIDE_L, mask=in_bounds, other=0)
        x = a.to(tl.uint64) + b.to(tl.uint64)

        g = x >= base
        p = x == (base - 1)
        G, P = tl.associative_scan((g, p), axis=0, combine_fn=gp_combine)
        carry_prefix = G | (P & carry_in)

        idx_prev = idx - 1
        prev_in_block = lane > 0
        prev_bounds = prev_in_block & (idx_prev >= l0) & (idx_prev < L)
        a_prev = tl.load(A_ptr + pid_n * STRIDE_N + idx_prev * STRIDE_L, mask=prev_bounds, other=0)
        b_prev = tl.load(B_ptr + pid_n * STRIDE_N + idx_prev * STRIDE_L, mask=prev_bounds, other=0)
        x_prev = a_prev.to(tl.uint64) + b_prev.to(tl.uint64)
        g_prev = x_prev >= base
        p_prev = x_prev == (base - 1)

        gS = tl.where(lane == 0, False, g_prev)
        pS = tl.where(lane == 0, True,  p_prev)
        GS, PS = tl.associative_scan((gS, pS), axis=0, combine_fn=gp_combine)
        c_vec = GS | (PS & carry_in)

        s = x + c_vec.to(tl.uint64)
        out = s & base_mask
        tl.store(C_ptr + pid_n * OUT_STRIDE_N + idx * OUT_STRIDE_L, out, mask=in_bounds)

        last = tl.minimum(l0 + BLOCK_L, L) - 1
        is_last_lane = idx == last
        carry_in = (tl.sum(tl.where(is_last_lane, carry_prefix.to(tl.int32), 0)) != 0)

    tl.store(Carry_ptr + pid_n * CARRY_STRIDE, carry_in.to(tl.int32))

# ==========================================
# 3. Multiplication Kernel (Full)
# ==========================================

@triton.jit
def _add_u128(acc_lo, acc_hi, x_u64):
    new_lo = acc_lo + x_u64
    carry = new_lo < acc_lo
    new_hi = acc_hi + carry.to(tl.uint64)
    return new_lo, new_hi

@triton.jit
def _shr_u128(acc_lo, acc_hi, shift: tl.constexpr):
    out_lo = (acc_lo >> shift) | (acc_hi << (64 - shift))
    out_hi = (acc_hi >> shift)
    return out_lo, out_hi

@triton.jit
def mul_limbs_full_kernel(
    A_ptr, B_ptr, P_ptr,
    N: tl.constexpr, L: tl.constexpr,
    STRIDE_N: tl.constexpr, STRIDE_L: tl.constexpr,
    OUT_STRIDE_N: tl.constexpr, OUT_STRIDE_L: tl.constexpr,
    LIMB_BITS: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    if pid_n >= N: return

    base_mask = (1 << LIMB_BITS) - 1
    carry_lo = tl.zeros((), dtype=tl.uint64)
    carry_hi = tl.zeros((), dtype=tl.uint64)

    for i in tl.static_range(0, 2 * L):
        acc_lo = carry_lo
        acc_hi = carry_hi
        for k in tl.static_range(0, L):
            j = i - k
            if j >= 0 and j < L:
                a_k = tl.load(A_ptr + pid_n * STRIDE_N + k * STRIDE_L)
                b_j = tl.load(B_ptr + pid_n * STRIDE_N + j * STRIDE_L)
                term = a_k.to(tl.uint64) * b_j.to(tl.uint64)
                acc_lo, acc_hi = _add_u128(acc_lo, acc_hi, term)
        out_val = acc_lo & tl.full((), base_mask, tl.uint64)
        tl.store(P_ptr + pid_n * OUT_STRIDE_N + i * OUT_STRIDE_L, out_val)
        carry_lo, carry_hi = _shr_u128(acc_lo, acc_hi, LIMB_BITS)

# ==========================================
# 4. Division Kernel (Fixed for non-power-of-2 L)
# ==========================================

@triton.jit
def div_limbs_kernel(
    A_ptr, B_ptr, Q_ptr, R_ptr,
    N: tl.constexpr, L: tl.constexpr,
    STRIDE_N: tl.constexpr, STRIDE_L: tl.constexpr,
    LIMB_BITS: tl.constexpr,
    BLOCK_L: tl.constexpr, # Size of vector (power of 2 >= L)
):
    pid_n = tl.program_id(axis=0)
    if pid_n >= N: return

    # We work on a padded vector of size BLOCK_L
    idxs = tl.arange(0, BLOCK_L)
    mask_valid = idxs < L

    offsets = pid_n * STRIDE_N + idxs * STRIDE_L

    # Load input, pad with 0s
    A_vals = tl.load(A_ptr + offsets, mask=mask_valid, other=0)
    B_vals = tl.load(B_ptr + offsets, mask=mask_valid, other=0)

    Q_vals = tl.zeros((BLOCK_L,), dtype=tl.uint32)
    R_vals = tl.zeros((BLOCK_L,), dtype=tl.uint32)

    # We iterate exactly the number of bits in the *valid* numbers.
    total_bits = L * LIMB_BITS

    for step in tl.range(0, total_bits):
        bit_idx = total_bits - 1 - step
        limb_idx = bit_idx // LIMB_BITS
        local_bit_idx = bit_idx % LIMB_BITS

        # Extract the current bit from A (valid part)
        mask_limb = (idxs == limb_idx)
        curr_limb_a = tl.sum(tl.where(mask_limb, A_vals, 0))
        current_bit = (curr_limb_a >> local_bit_idx) & 1

        # Shift R (padded vector)
        R_vals = _shl_1_tensor(R_vals, current_bit, BLOCK_L)
        # Subtract (padded vector)
        # Borrow propagates through zero-padding correctly
        R_sub, borrow_out = _sub_tensor(R_vals, B_vals, BLOCK_L)

        ge = (borrow_out == 0)
        R_vals = tl.where(ge, R_sub, R_vals)
        q_bit = ge
        # Shift Q (padded vector)
        Q_vals = _shl_1_tensor(Q_vals, q_bit, BLOCK_L)

    # Store result, mask valid
    tl.store(Q_ptr + offsets, Q_vals, mask=mask_valid)
    tl.store(R_ptr + offsets, R_vals, mask=mask_valid)

# ==========================================
# 5. JAX Kernel Wrappers
# ==========================================

def _validate_shapes(A, B):
    # Broadcast batch dimensions first
    A, B = jnp.broadcast_arrays(A, B)

    if A.shape != B.shape:
        raise ValueError(f"Shapes mismatch after broadcast: {A.shape} vs {B.shape}")

    # Flatten arbitrary batch dims [..., L] -> [N, L]
    L = A.shape[-1]
    N = A.size // L
    return A.reshape(N, L), B.reshape(N, L), N, L

def run_add_sub(A, B, initial_carry=0):
    A_flat, B_flat, N, L = _validate_shapes(A, B)
    limb_bits = _bits_of_dtype(A.dtype)

    C_spec = jax.ShapeDtypeStruct((N, L), A.dtype)
    carry_spec = jax.ShapeDtypeStruct((N,), jnp.int32)

    C_out, _ = triton_call(
        A_flat, B_flat,
        kernel=add_limbs_prefixscan_kernel,
        out_shape=(C_spec, carry_spec),
        grid=(N,),
        N=N, L=L, STRIDE_N=L, STRIDE_L=1,
        OUT_STRIDE_N=L, OUT_STRIDE_L=1,
        CARRY_STRIDE=1, LIMB_BITS=limb_bits, BLOCK_L=128,
        INITIAL_CARRY=initial_carry
    )
    return C_out.reshape(A.shape)

def add(A, B): return run_add_sub(A, B, 0)

def sub(A, B):
    # A - B = A + (~B) + 1
    return run_add_sub(A, jnp.invert(B), 1)

def mul_unsigned_op(A, B):
    A_flat, B_flat, N, L = _validate_shapes(A, B)
    limb_bits = _bits_of_dtype(A.dtype)
    P_spec = jax.ShapeDtypeStruct((N, 2 * L), A.dtype)

    P_out = triton_call(
        A_flat, B_flat,
        kernel=mul_limbs_full_kernel,
        out_shape=P_spec,
        grid=(N,),
        N=N, L=L, STRIDE_N=L, STRIDE_L=1,
        OUT_STRIDE_N=2*L, OUT_STRIDE_L=1, LIMB_BITS=limb_bits
    )
    return P_out.reshape(tuple(list(A.shape[:-1]) + [2*L]))

def mul_signed(A, B):
    # A, B shape [..., L]
    # Output shape [..., 2L]
    # Implements P = A * B using standard signed multiplication correction

    # 1. Unsigned Multiplication
    P_full = mul_unsigned_op(A, B)

    # 2. Correction for Signed (Two's Complement)
    # If B < 0: High(P) -= A
    # If A < 0: High(P) -= B

    dtype = A.dtype
    bits = _bits_of_dtype(dtype)
    msb_mask = jnp.array(1 << (bits - 1), dtype=dtype)

    a_sign = (A[..., -1] & msb_mask) != 0
    b_sign = (B[..., -1] & msb_mask) != 0

    L = A.shape[-1]

    P_low = P_full[..., :L]
    P_high = P_full[..., L:]

    # Apply corrections sequentially
    # Correction 1: if B < 0, P_high = P_high - A
    P_high_1 = jnp.where(b_sign[..., None], sub(P_high, A), P_high)

    # Correction 2: if A < 0, P_high = P_high - B
    P_high_2 = jnp.where(a_sign[..., None], sub(P_high_1, B), P_high_1)

    return jnp.concatenate([P_low, P_high_2], axis=-1)

def divmod_unsigned_op(A, B):
    A_flat, B_flat, N, L = _validate_shapes(A, B)
    # div kernel currently hardcoded for uint32
    if A.dtype != jnp.uint32: raise TypeError("Div requires uint32")

    Q_spec = jax.ShapeDtypeStruct((N, L), A.dtype)
    R_spec = jax.ShapeDtypeStruct((N, L), A.dtype)

    # Calculate next power of 2 for vector size
    block_l = _next_power_of_2(L)

    Q_out, R_out = triton_call(
        A_flat, B_flat,
        kernel=div_limbs_kernel,
        out_shape=(Q_spec, R_spec),
        grid=(N,),
        N=N, L=L, STRIDE_N=L, STRIDE_L=1, LIMB_BITS=32,
        BLOCK_L=block_l
    )
    return Q_out.reshape(A.shape), R_out.reshape(A.shape)

def divmod_signed(A, B):
    # A, B are JAX arrays with shape [..., L]
    # Implements signed division using floor behavior (Python style)

    dtype = A.dtype
    bits = _bits_of_dtype(dtype)

    # Safe explicit cast to prevent OverflowError in int32 inference
    msb_mask = jnp.array(1 << (bits - 1), dtype=dtype)

    # 1. Determine signs (MSB of last limb)
    a_sign = (A[..., -1] & msb_mask) != 0
    b_sign = (B[..., -1] & msb_mask) != 0

    # 2. Compute absolute values
    # Construct BigInt 0: [0, 0, ...]
    zero = jnp.zeros_like(A)

    # neg_A = 0 - A
    neg_A = sub(zero, A)
    neg_B = sub(zero, B)

    uA = jnp.where(a_sign[..., None], neg_A, A)
    uB = jnp.where(b_sign[..., None], neg_B, B)

    # 3. Unsigned Division
    uQ, uR = divmod_unsigned_op(uA, uB)

    # 4. Fix signs for Truncated Division
    # Q_trunc = -uQ if signs differ
    q_sign = a_sign ^ b_sign
    neg_uQ = sub(zero, uQ)
    neg_uR = sub(zero, uR)

    Q_trunc = jnp.where(q_sign[..., None], neg_uQ, uQ)
    R_trunc = jnp.where(a_sign[..., None], neg_uR, uR)

    # 5. Convert to Floor Division (Python behavior)
    # If signs differ and remainder != 0:
    #   Q = Q_trunc - 1
    #   R = R_trunc + B

    # Construct BigInt 1: [1, 0, 0, ...]
    one = zero.at[..., 0].set(1)

    r_nonzero = jnp.any(R_trunc != 0, axis=-1)
    adjust = q_sign & r_nonzero

    Q_adj = sub(Q_trunc, one)
    R_adj = add(R_trunc, B)

    Q = jnp.where(adjust[..., None], Q_adj, Q_trunc)
    R = jnp.where(adjust[..., None], R_adj, R_trunc)

    return Q, R

add_jit = jax.jit(add)
sub_jit = jax.jit(sub)
mul_jit = jax.jit(mul_signed) # Defaults to signed multiplication
divmod_jit = jax.jit(divmod_signed) # Defaults to signed division