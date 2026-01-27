import jax
import jax.numpy as jnp
try:
    import triton
    import triton.language as tl
    from jax_triton import triton_call
    
    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ):
        """
        Computes C = A @ B using TF32/FP32 accumulation.
        This kernel is used as the 'exact' low-precision inner multiplication for the Ozaki splits.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        for k in range(0, K, BLOCK_SIZE_K):
            k_mask = (k + offs_k) < K
            a_mask = (offs_m[:, None] < M) & k_mask[None, :]
            b_mask = k_mask[:, None] & (offs_n[None, :] < N)
            
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            accumulator += tl.dot(a, b)
            
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            
        c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
except ImportError:
    triton = None
    triton_call = None
    matmul_kernel = None


def triton_matmul(A: jax.Array, B: jax.Array) -> jax.Array:
    try:
        # Check for triton explicitly
        if triton_call is None:
            raise ImportError("Triton not available")
            
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "Dimension mismatch"
        
        # Block sizes
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        
        # Grid
        grid = (lambda meta: (
            triton.cdiv(M, meta['BLOCK_SIZE_M']),
            triton.cdiv(N, meta['BLOCK_SIZE_N'])
        ))
        
        return triton_call(
            A, B,
            kernel=matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
            grid=grid,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )
    except (ImportError, NameError, RuntimeError):
        # Fallback to JAX native if Triton is missing or fails
        return jnp.matmul(A, B)
