
import math
import numpy as np
import jax
import jax.numpy as jnp
from .triton_mod_matmul import triton_modular_matmul

# INT8-friendly moduli from the paper (m in eq. (19)); pairwise coprime, all <= 256.
MODULI_INT8 = [
    256, 255, 253, 251, 247, 239, 233, 229,
    227, 223, 217, 211, 199, 197, 193, 191
]

def _symmetric_mod(x, m: int):
    half = m // 2
    return ((x + half) % m) - half

def _crt_coeffs(moduli):
    M = 1
    for m in moduli:
        M *= int(m)
    coeffs = []
    for m in moduli:
        Mi = M // int(m)
        yi = pow(int(Mi), -1, int(m))
        coeffs.append(Mi * yi)
    return M, coeffs

def _compute_scaling(A: jax.Array, B: jax.Array, moduli):
    q = A.shape[1]
    M = 1
    for m in moduli:
        M *= int(m)
    if M <= 2 or q <= 0:
        return 1.0, 1.0, 0, 0, M
    target = (M / 2.0 - 1.0) / float(q)
    k_sum = int(max(0.0, math.floor(math.log2(target)))) if target > 1.0 else 0
    kA = k_sum // 2
    kB = k_sum - kA
    scale_A = float(2 ** kA)
    scale_B = float(2 ** kB)
    return scale_A, scale_B, kA, kB, M

def _crt_reconstruct_host(Cs, coeffs, M: int):
    acc = None
    for C_t, coeff in zip(Cs, coeffs):
        C_np = np.array(jax.device_get(C_t), dtype=object)
        term = C_np * int(coeff)
        acc = term if acc is None else acc + term
    acc = acc % int(M)
    half = M // 2
    acc = (acc + half) % int(M) - half
    return jnp.array(acc, dtype=jnp.float64)

def ozaki_scheme_2_solve(A: jax.Array, B: jax.Array, num_moduli: int = 6, moduli=None):
    """
    Implements Ozaki Scheme II (Algorithm 1 in the paper) using INT8 tensor cores.
    """
    moduli = MODULI_INT8[:num_moduli] if moduli is None else list(moduli)
    if len(moduli) == 0:
        raise ValueError("At least one modulus is required.")
    
    # 1) Determine shift values (D, E) and convert FP64 to integers.
    scale_A, scale_B, _, _, M = _compute_scaling(A, B, moduli)
    A_prime = jnp.trunc(A * scale_A).astype(jnp.int64)
    B_prime = jnp.trunc(B * scale_B).astype(jnp.int64)
    
    # 2) Modular arithmetic and INT8 tensor core GEMM.
    Cs = []
    for m in moduli:
        A_t = _symmetric_mod(A_prime, m).astype(jnp.int8)
        B_t = _symmetric_mod(B_prime, m).astype(jnp.int8)
        C_t = triton_modular_matmul(A_t, B_t, int(m))  # int32 residues
        Cs.append(C_t)
    
    # 3) CRT reconstruction: C ≡ A'B' (mod M)
    M_full, coeffs = _crt_coeffs(moduli)
    if M_full <= 2 ** 55:
        acc = jnp.zeros_like(Cs[0], dtype=jnp.int64)
        for C_t, coeff in zip(Cs, coeffs):
            acc = (acc + C_t.astype(jnp.int64) * jnp.int64(coeff)) % jnp.int64(M_full)
        acc = _symmetric_mod(acc, M_full)
        C = acc.astype(jnp.float64) / (scale_A * scale_B)
        return C
    
    # Fallback: exact host-side CRT for large M (slow but correct).
    C_host = _crt_reconstruct_host(Cs, coeffs, M_full)
    return C_host / (scale_A * scale_B)
