import jax
import jax.numpy as jnp
from .triton_kernel_v1 import triton_matmul

def split_matrix(A: jnp.ndarray, target_bits: int = 10, max_splits: int = 5):
    """
    Decomposes matrix A into a sum of matrices A_k such that A ~ sum(A_k).
    This implements a mantissa-chopping split with a fixed target mantissa width.
    
    Args:
        A: Input matrix (FP64)
        target_bits: Target mantissa bits to retain per split (e.g., 10 for TF32)
        
    Returns:
        List of matrices [A_0, A_1, ...]
    """
    splits = []
    residual = A
    
    for _ in range(max_splits):
        abs_residual = jnp.abs(residual)
        exp = jnp.where(abs_residual > 0, jnp.floor(jnp.log2(abs_residual)), 0.0)
        scale = jnp.exp2(exp - (target_bits - 1))
        split = jnp.round(residual / scale) * scale
        splits.append(split)
        residual = residual - split
        
    return splits

@jax.jit
def ozaki_matmul_v1(A: jnp.ndarray, B: jnp.ndarray, target_bits: int = 10, max_splits: int = 5):
    """
    Performs Matrix Multiplication using a simplified Ozaki-like splitting scheme.
    C = A * B
    
    Note: This is a high-level JAX implementation to verify the logic.
    A Triton implementation would move the inner product computation to a kernel.
    """
    # 1. Split A and B
    # For simplicity, let's split A and keep B or split both.
    # The paper often splits both to ensure error-free accumulation in low precision.
    
    # Epsilon for float32 (23 bits specific). 
    # If we target TensorCores (TF32), mantissa is 10 bits.
    # Let's assume we want to use TF32-like behavior.
    
    A_splits = split_matrix(A, target_bits=target_bits, max_splits=max_splits)
    B_splits = split_matrix(B, target_bits=target_bits, max_splits=max_splits)
    
    C = jnp.zeros((A.shape[0], B.shape[1]), dtype=jnp.float64)
    
    for Ak in A_splits:
        for Bj in B_splits:
            # The core idea: Ak @ Bj can be computed safely with lower precision
            # then accumulated in high precision.
            # In JAX, we can enforce lower precision math:
            # Prefer Triton when available for TF32/FP32 accumulation.
            if triton_matmul is not None:
                partial = triton_matmul(Ak.astype(jnp.float32), Bj.astype(jnp.float32)).astype(jnp.float64)
            else:
                partial = jnp.matmul(Ak.astype(jnp.float32), Bj.astype(jnp.float32)).astype(jnp.float64)
            C += partial
            
    return C
