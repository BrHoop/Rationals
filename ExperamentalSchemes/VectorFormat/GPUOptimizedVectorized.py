import jax 
import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jax_triton import triton_call

@triton.jit
def mul_u16_limbs32_mod_kernel(A_ptr, B_ptr, C_ptr, N: tl.constexpr):
    pid = tl.program_id(0)

    # Load 32-limb inputs
    offs = tl.arange(0, 32)
    a = tl.load(A_ptr + pid * 32 + offs, mask=(pid < N), other=0).to(tl.uint32)
    b = tl.load(B_ptr + pid * 32 + offs, mask=(pid < N), other=0).to(tl.uint32)

    # Compute low 32 convolution accumulators in uint64:
    # acc[k] = sum_{i=0..k} a[i]*b[k-i]
    ks = tl.arange(0, 32)
    acc = tl.zeros([32], dtype=tl.uint64)

    # Unrolled loop over i (0..31). Only contributes when i <= k.
    for i in range(32):
        j = ks - i
        valid = (j >= 0) & (j < 32)
        bj = tl.where(valid, b[j], 0).to(tl.uint64)
        ai = a[i].to(tl.uint64)
        acc += ai * bj

    # Carry normalize base 2^16 across 32 limbs; drop carry-out (mod 2^512)
    carry = tl.uint64(0)
    out = tl.zeros([32], dtype=tl.uint16)
    for k in range(32):
        t = acc[k] + carry
        out = tl.where(ks == k, (t & tl.uint64(0xFFFF)).to(tl.uint16), out)
        carry = t >> 16

    # Store result
    tl.store(C_ptr + pid * 32 + ks, out, mask=(pid < N))

@jax.jit
def mul_u16_limbs32_mod_triton(A_u16: jnp.ndarray, B_u16: jnp.ndarray) -> jnp.ndarray:
    if A_u16.dtype != jnp.uint16 or B_u16.dtype != jnp.uint16:
        raise TypeError("Expected uint16 inputs.")
    if A_u16.shape != B_u16.shape or A_u16.shape[-1] != 32:
        raise ValueError("Expected shape (N,32).")
    N = A_u16.shape[0]

    A_flat = A_u16.reshape(-1)
    B_flat = B_u16.reshape(-1)
    C_flat = jnp.empty_like(A_flat)

    C_flat = triton_call(
        A_flat, B_flat, C_flat,
        kernel=mul_u16_limbs32_mod_kernel,
        grid=(N,),
        N=N,
    )
    return C_flat.reshape(N, 32)


@triton.jit
def add_u16_limbs32_mod_kernel(A_ptr, B_ptr, C_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    lanes = tl.arange(0, 32)
    idx = pid * 32 + lanes

    mask = idx < (N * 32)

    a = tl.load(A_ptr + idx, mask=mask, other=0).to(tl.uint32)
    b = tl.load(B_ptr + idx, mask=mask, other=0).to(tl.uint32)
    s = a + b

    G = (s >> 16).to(tl.uint32)
    P = (s == 0xFFFF).to(tl.uint32)

    for off in (1, 2, 4, 8, 16):
        G_left = tl.where(lanes >= off, G[lanes - off], 0)
        P_left = tl.where(lanes >= off, P[lanes - off], 1)
        G = G | (P & G_left)
        P = P & P_left

    c_in = tl.where(lanes == 0, 0, G[lanes - 1])
    out = (s + c_in) & 0xFFFF

    tl.store(C_ptr + idx, out.to(tl.uint16), mask=mask)

@jax.jit
def add_u16_limbs32_mod_triton(A_u16: jnp.ndarray, B_u16: jnp.ndarray) -> jnp.ndarray:
    if A_u16.dtype != jnp.uint16 or B_u16.dtype != jnp.uint16:
        raise TypeError("Expected uint16 inputs.")
    if A_u16.shape != B_u16.shape or A_u16.shape[-1] != 32:
        raise ValueError("Expected shape (N,32).")
    N = A_u16.shape[0]

    A_flat = A_u16.reshape(-1)
    B_flat = B_u16.reshape(-1)
    C_flat = jnp.empty_like(A_flat)

    C_flat = triton_call(
        A_flat, B_flat, C_flat,
        kernel=add_u16_limbs32_mod_kernel,
        grid=(N,),
        N=N,
    )
    return C_flat.reshape(N, 32)
