#pragma once
#include "common.hpp"
#include "scaling_fast_real.hpp"

namespace oz2 {
namespace real {
namespace accu {

__forceinline__ __device__ int16_t compute_sft(int amax, const float log2P) {
    const float log2amax = __log2f(__int2float_rn(amax));
    return __float2int_rd(__fmaf_rd(-0x1.0000060000000p-1F, log2amax, log2P));
}

//------------------------------
// Determine row-wise shift values
//------------------------------
template <typename T>
__global__ void compute_sft_extract_kernel(
    const size_t m,                  // size(A,1)
    const size_t k,                  // size(A,2)
    const T *const A,                // input (lda * k)
    const size_t lda,                // leading dimension
    int16_t *const __restrict__ sftA // exponent of shift values
) {
    __shared__ T samax[TILE_DIM][TILE_DIM + 1];

    unsigned row_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    const T amax     = find_amax_tile<T>(m, k, row_idx, A, lda, samax);

    row_idx = blockIdx.x * TILE_DIM + threadIdx.y;
    if (row_idx < m && threadIdx.x == 0) {
        sftA[row_idx] = 5 - Tilogb<T>(amax); // 6-bit
    }
}

//------------------------------
// Extract first 7-bit of abs(A^T)
//------------------------------
template <typename T>
__global__ void extract_A8i_kernel(
    const size_t m,                  // size(A,1)
    const size_t k,                  // size(A,2)
    const T *const __restrict__ A,   // input (lda * k)
    const size_t lda,                // leading dimension
    int8_t *const __restrict__ A8i,  // output (lda8i * (m+pad))
    const size_t lda8i,              // leading dimension
    int16_t *const __restrict__ sftA // exponent of shift values
) {
    __shared__ int8_t tile[TILE_DIM][TILE_DIM + 1];

    const auto rowBase = blockIdx.x * TILE_DIM;
    const auto colBase = blockIdx.y * TILE_DIM;

    const auto in_row = rowBase + threadIdx.x;
    const auto in_col = colBase + threadIdx.y;

    const int16_t sft              = (in_row < m) ? sftA[in_row] : Tconst<int16_t>::zero();
    const T Atmp                   = (in_row < m && in_col < k) ? A[in_col * lda + in_row] : Tconst<T>::zero();
    tile[threadIdx.y][threadIdx.x] = T2int_8i<T>(Atmp, sft); // <= 2^6
    __syncthreads();

    const auto out_row = colBase + threadIdx.x;
    const auto out_col = rowBase + threadIdx.y;
    if (out_col >= m || out_row >= lda8i) return;

    A8i[out_col * lda8i + out_row] = tile[threadIdx.x][threadIdx.y];
}

//------------------------------
// Extract first 7-bit of abs(B)
//------------------------------
template <typename T>
__global__ void extract_B8i_kernel(
    const size_t k,                  // size(B,1)
    const T *const __restrict__ B,   // input (ldb * n)
    const size_t ldb,                // leading dimension
    char4 *const __restrict__ B8i,   // output (ldb8i / 4 * n)
    const size_t ldb8i,              // leading dimension ldb8i / 4
    int16_t *const __restrict__ sftB // exponent of shift values
) {
    __shared__ T shm[32];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    const T amax                   = find_amax<T>(in, k, shm);
    const int16_t sft              = 5 - Tilogb<T>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftB[col_idx] = sft;
    }

    char4 *const __restrict__ out = B8i + col_idx * ldb8i;
    unsigned kmax                 = k >> 2;
    unsigned i                    = threadIdx.x;

    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        T in0 = in[idx];
        T in1 = in[idx + 1];
        T in2 = in[idx + 2];
        T in3 = in[idx + 3];

        char4 out4;
        out4.x = T2int_8i<T>(in0, sft); // <= 2^6
        out4.y = T2int_8i<T>(in1, sft); // <= 2^6
        out4.z = T2int_8i<T>(in2, sft); // <= 2^6
        out4.w = T2int_8i<T>(in3, sft); // <= 2^6

        out[i] = out4;
    }
    for (; i < ldb8i; i += blockDim.x) {
        unsigned idx = i << 2;

        T in0 = (idx < k) ? in[idx] : Tconst<T>::zero();
        T in1 = (idx + 1 < k) ? in[idx + 1] : Tconst<T>::zero();
        T in2 = (idx + 2 < k) ? in[idx + 2] : Tconst<T>::zero();
        T in3 = (idx + 3 < k) ? in[idx + 3] : Tconst<T>::zero();

        char4 out4;
        out4.x = T2int_8i<T>(in0, sft); // <= 2^6
        out4.y = T2int_8i<T>(in1, sft); // <= 2^6
        out4.z = T2int_8i<T>(in2, sft); // <= 2^6
        out4.w = T2int_8i<T>(in3, sft); // <= 2^6

        out[i] = out4;
    }
}

//------------------------------
// Determine row-wise shift values
//------------------------------
__global__ void compute_sft_rowwise_kernel(
    const size_t m,                  // size(C32i,1)
    const size_t n,                  // size(C32i,2)
    const int *const C32i,           // input (ldc32i * n)
    const size_t ldc32i,             // leading dimension
    int16_t *const __restrict__ sft, // exponent of shift values
    const float log2P                // log2(P-1)/2 - 0.5
) {
    __shared__ int samax[TILE_DIM][TILE_DIM + 1];

    unsigned row_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    const int amax   = find_max_tile(m, n, row_idx, C32i, ldc32i, samax);

    row_idx = blockIdx.x * TILE_DIM + threadIdx.y;
    if (row_idx < m && threadIdx.x == 0) {
        int16_t sft_tmp = sft[row_idx];
        sft_tmp += compute_sft(amax, log2P);
        sft[row_idx] = -sft_tmp;
    }
}

//------------------------------
// Determine column-wise shift values
//------------------------------
__global__ void compute_sft_colwise_kernel(
    const size_t m,                  // size(C32i,1)
    const int *const C32i,           // input (ldc32i * n)
    const size_t ldc32i,             // leading dimension
    int16_t *const __restrict__ sft, // exponent of shift values
    const float log2P                // log2(P-1)/2 - 0.5
) {
    __shared__ int shm[32];
    const auto col_idx = blockIdx.x;
    const int amax     = find_max(C32i + col_idx * ldc32i, m, shm);

    if (threadIdx.x == 0) {
        int16_t sft_tmp = sft[col_idx];
        sft_tmp += compute_sft(amax, log2P);
        sft[col_idx] = -sft_tmp;
    }
}

//------------------------------
// Convert trunc(B*diag(2^sftB)) to B8i
//------------------------------
template <typename T, int ITER>
__global__ void scalingB_kernel(
    const size_t k,                  // size(B,1)
    const size_t incB8i,             // ldb8i / 4 * n
    const unsigned num_moduli,       // #moduli
    const T *const __restrict__ B,   // input (ldb * n)
    const size_t ldb,                // leading dimension
    char4 *const __restrict__ B8i,   // output (ldb8i / 4 * n)
    const size_t ldb8i,              // leading dimension ldb8i / 4
    int16_t *const __restrict__ sftB // exponent of shift values
) {
    const auto col_idx = blockIdx.x;
    int16_t sft        = -sftB[col_idx];

    const T *const __restrict__ in = B + col_idx * ldb;
    char4 *const __restrict__ out  = B8i + col_idx * ldb8i;
    oz2::real::fast::scalingB_device<T, ITER>(k, incB8i, num_moduli, in, out, ldb8i, sft);
}

//------------------------------
// Launcher!!
//------------------------------
template <typename T>
__forceinline__ void extract_8i_launch(
    const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
    const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
    const size_t m,               // Number of rows of C
    const size_t n,               // Number of columns of C
    const size_t k,               // Inner dimension
    const T *const A,             // input
    const size_t lda,             // leading dimension
    int8_t *const A8i,            // output (lda8i * (m+pad))
    const size_t lda8i,           // leading dimension
    int16_t *const sftA,          // exponent of shift values for rows of A
    const T *const B,             // input
    const size_t ldb,             // leading dimension
    int8_t *const B8i,            // output (ldb8i * n)
    const size_t ldb8i,           // leading dimension
    int16_t *const sftB,          // exponent of shift values for cols of B
    const bool skip_scalA,        // false (unskip scaling_A) or true (skip scaling_A)
    const bool skip_scalB         // false (unskip scaling_B) or true (skip scaling_B)
) {
    if (!skip_scalA) {
        if (op_A == CUBLAS_OP_N) {
            // m*k -> k*m
            constexpr dim3 threads1(TILE_DIM, TILE_DIM);
            dim3 grid((m + (TILE_DIM - 1)) / TILE_DIM, (lda8i + (TILE_DIM - 1)) / TILE_DIM);
            compute_sft_extract_kernel<T><<<grid.x, threads1>>>(m, k, A, lda, sftA);
            extract_A8i_kernel<T><<<grid, threads1>>>(m, k, A, lda, A8i, lda8i, sftA);
        } else {
            // k*m -> k*m
            extract_B8i_kernel<T><<<m, threads_scaling>>>(k, A, lda, reinterpret_cast<char4 *>(A8i), lda8i >> 2, sftA);
        }
    }

    if (!skip_scalB) {
        if (op_B == CUBLAS_OP_N) {
            // k*n -> k*n
            extract_B8i_kernel<T><<<n, threads_scaling>>>(k, B, ldb, reinterpret_cast<char4 *>(B8i), ldb8i >> 2, sftB);
        } else {
            // n*k -> k*n
            constexpr dim3 threads1(TILE_DIM, TILE_DIM);
            dim3 grid((n + (TILE_DIM - 1)) / TILE_DIM, (ldb8i + (TILE_DIM - 1)) / TILE_DIM);
            compute_sft_extract_kernel<T><<<grid.x, threads1>>>(n, k, B, ldb, sftB);
            extract_A8i_kernel<T><<<grid, threads1>>>(n, k, B, ldb, B8i, ldb8i, sftB);
        }
    }
}

//------------------------------
// Launcher!!
//------------------------------
template <typename T, int ITER>
__forceinline__ void scaling_launch(
    const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
    const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
    const size_t m,               // Number of rows of C
    const size_t n,               // Number of columns of C
    const size_t k,               // Inner dimension
    const unsigned num_moduli,    // #moduli
    const T *const A,             // input
    const size_t lda,             // leading dimension
    int8_t *const A8i,            // output (lda8i * (m+pad))
    const size_t lda8i,           // leading dimension
    const size_t incA8i,          // increment between the A8i
    int16_t *const sftA,          // exponent of shift values for rows of A
    const T *const B,             // input
    const size_t ldb,             // leading dimension
    int8_t *const B8i,            // output (ldb8i * n)
    const size_t ldb8i,           // leading dimension
    const size_t incB8i,          // increment between the B8i
    int16_t *const sftB,          // exponent of shift values for cols of B
    int32_t *const C32i,          // tmp (ldc32i * n)
    const size_t ldc32i,          // ((m + 15) >> 4) << 4
    const float log2P,            // fld(log2(P-1)/2 - 0.5)
    const bool skip_scalA,        // false (unskip scaling_A) or true (skip scaling_A)
    const bool skip_scalB         // false (unskip scaling_B) or true (skip scaling_B)
) {
    if (!skip_scalA) {
        if (op_A == CUBLAS_OP_N) {
            // m*k -> k*m
            dim3 grid((m + (TILE_DIM - 1)) / TILE_DIM, (lda8i + (TILE_DIM - 1)) / TILE_DIM);
            constexpr dim3 threads1(TILE_DIM, TILE_DIM);
            compute_sft_rowwise_kernel<<<grid.x, threads1>>>(m, n, C32i, ldc32i, sftA, log2P);
            oz2::real::fast::scalingA_kernel<T, ITER><<<grid, threads1>>>(m, k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA);
        } else {
            // k*m -> k*m
            constexpr dim3 threads1(TILE_DIM, TILE_DIM);
            compute_sft_rowwise_kernel<<<(m + (TILE_DIM - 1)) / TILE_DIM, threads1>>>(m, n, C32i, ldc32i, sftA, log2P);
            scalingB_kernel<T, ITER><<<m, threads_scaling>>>(k, incA8i >> 2, num_moduli, A, lda, reinterpret_cast<char4 *>(A8i), lda8i >> 2, sftA);
        }
    }

    if (!skip_scalB) {
        if (op_B == CUBLAS_OP_N) {
            // k*n -> k*n
            compute_sft_colwise_kernel<<<n, threads_scaling>>>(m, C32i, ldc32i, sftB, log2P);
            scalingB_kernel<T, ITER><<<n, threads_scaling>>>(k, incB8i >> 2, num_moduli, B, ldb, reinterpret_cast<char4 *>(B8i), ldb8i >> 2, sftB);
        } else {
            // n*k -> k*n
            compute_sft_colwise_kernel<<<n, threads_scaling>>>(m, C32i, ldc32i, sftB, log2P);
            dim3 grid((n + (TILE_DIM - 1)) / TILE_DIM, (ldb8i + (TILE_DIM - 1)) / TILE_DIM);
            constexpr dim3 threads2(TILE_DIM, TILE_DIM);
            oz2::real::fast::scalingA_kernel<T, ITER><<<grid, threads2>>>(n, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB);
        }
    }
}

//------------------------------
// Interface!!
//------------------------------
template <typename T>
__inline__ void scaling(
    cublasHandle_t handle,        // Handle to the cuBLAS library context
    const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
    const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
    const size_t m,               // Number of rows of C
    const size_t n,               // Number of columns of C
    const size_t k,               // Inner dimension
    const unsigned num_moduli,    // #moduli
    const T *const A,             // input
    const size_t lda,             // leading dimension
    int8_t *const A8i,            // output (lda8i * m)
    int8_t *const A8i_high,       // work/output (lda8i * (m+pad))
    const size_t lda8i,           // leading dimension
    const size_t incA8i,          // increment between the A8i
    int16_t *const sftA,          // exponent of shift values for rows of A
    const T *const B,             // input
    const size_t ldb,             // leading dimension
    int8_t *const B8i,            // output (ldb8i * n)
    int8_t *const B8i_high,       // work/output (lda8i * n)
    const size_t ldb8i,           // leading dimension
    const size_t incB8i,          // increment between the B8i
    int16_t *const sftB,          // exponent of shift values for cols of B
    int32_t *const C32i,          // tmp (ldc32i * n)
    const size_t ldc32i,          // ((m + 15) >> 4) << 4
    const unsigned table_idx,     //
    const bool skip_scalA,        // false (unskip scaling_A) or true (skip scaling_A)
    const bool skip_scalB         // false (unskip scaling_B) or true (skip scaling_B)
) {
    // Extract first 7-bit from A and B
    extract_8i_launch<T>(op_A, op_B, m, n, k,
                         A, lda, A8i_high, lda8i, sftA,
                         B, ldb, B8i_high, ldb8i, sftB,
                         skip_scalA, skip_scalB);

    // C32i := A8i^T*B8i
    constexpr int32_t alpha = 1;
    constexpr int32_t beta  = 0;

    int blk    = 8192;
    int rem    = n;
    int offset = 0;

    while (rem > 0) {
        size_t nn = (rem <= 12288) ? rem : blk;

        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     ldc32i, nn, lda8i,
                     &alpha,
                     A8i_high, CUDA_R_8I, lda8i,
                     B8i_high + offset * ldb8i, CUDA_R_8I, ldb8i,
                     &beta,
                     C32i + offset * ldc32i, CUDA_R_32I, ldc32i,
                     CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);

        offset += nn;
        rem -= nn;
    }

    const float log2P = table::accu::log2P[table_idx]; // fld(log2(P-1)/2 - 0.5)

    // Convert A and B to INT8 matrices
    if (num_moduli <= threshold<T>::iter1) {
        scaling_launch<T, 1>(op_A, op_B, m, n, k, num_moduli,
                             A, lda, A8i, lda8i, incA8i, sftA,
                             B, ldb, B8i, ldb8i, incB8i, sftB,
                             C32i, ldc32i,
                             log2P, skip_scalA, skip_scalB);

    } else if (num_moduli <= threshold<T>::iter2) {
        scaling_launch<T, 2>(op_A, op_B, m, n, k, num_moduli,
                             A, lda, A8i, lda8i, incA8i, sftA,
                             B, ldb, B8i, ldb8i, incB8i, sftB,
                             C32i, ldc32i,
                             log2P, skip_scalA, skip_scalB);

    } else if (num_moduli <= threshold<T>::iter3) {
        scaling_launch<T, 3>(op_A, op_B, m, n, k, num_moduli,
                             A, lda, A8i, lda8i, incA8i, sftA,
                             B, ldb, B8i, ldb8i, incB8i, sftB,
                             C32i, ldc32i,
                             log2P, skip_scalA, skip_scalB);

    } else {
        scaling_launch<T, 4>(op_A, op_B, m, n, k, num_moduli,
                             A, lda, A8i, lda8i, incA8i, sftA,
                             B, ldb, B8i, ldb8i, incB8i, sftB,
                             C32i, ldc32i,
                             log2P, skip_scalA, skip_scalB);
    }
}

} // namespace accu
} // namespace real
} // namespace oz2
