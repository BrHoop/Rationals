#pragma once
#include "common.hpp"
#include "template_math.hpp"

namespace oz2 {
namespace real {
namespace fast {

template <typename T> __forceinline__ __device__ int16_t compute_sft(T amax, T vecnrm, const float log2P);
template <> __forceinline__ __device__ int16_t compute_sft<double>(double amax, double vecnrm, const float log2P) {
    const int exponent   = Tilogb<double>(vecnrm);
    const float vecnrmf  = __double2float_ru(scalbn(vecnrm, -exponent));
    const float log2vnrm = __fadd_ru(__log2f(vecnrmf), exponent);
    const float tmp      = __fmaf_rd(-0x1.0000060000000p-1F, log2vnrm, log2P);
    const int sft2       = __float2int_rd(tmp);
    return min(__float2int_rd(log2P - 1.0f), sft2) - Tilogb<double>(amax);
}
template <> __forceinline__ __device__ int16_t compute_sft<float>(float amax, float vecnrm, const float log2P) {
    const float log2vnrm = __log2f(vecnrm);
    const float tmp      = __fmaf_rd(-0x1.0000060000000p-1F, log2vnrm, log2P);
    const int sft2       = __float2int_rd(tmp);
    return min(__float2int_rd(log2P - 1.0f), sft2) - Tilogb<float>(amax);
}

//------------------------------
// Determine row-wise shift values
//------------------------------
template <typename T>
__global__ void compute_sftA_kernel(
    const size_t m,                   // size(A,1)
    const size_t k,                   // size(A,2)
    const T *const A,                 // input (lda * k)
    const size_t lda,                 // leading dimension
    int16_t *const __restrict__ sftA, // exponent of shift values
    const float log2P                 // log2(P-1)/2 - 1.5
) {
    __shared__ T samax[TILE_DIM][TILE_DIM + 1];
    __shared__ T ssum[TILE_DIM][TILE_DIM + 1];

    unsigned row_idx = blockIdx.x * TILE_DIM + threadIdx.x;

    T sum;
    T amax = find_amax_and_nrm_tile<T>(m, k, row_idx, A, lda, samax, ssum, sum);

    row_idx = blockIdx.x * TILE_DIM + threadIdx.y;
    if (row_idx < m && threadIdx.x == 0) {
        int16_t sft   = compute_sft<T>(amax, sum, log2P);
        sftA[row_idx] = -sft;
    }
}

//------------------------------
// Convert trunc(diag(2^sftA)*A) to A8i
//------------------------------
template <typename T, int ITER>
__global__ void scalingA_kernel(
    const size_t m,                  // size(A,1)
    const size_t k,                  // size(A,2)
    const size_t incA8i,             // lda8i * (m+pad)
    const unsigned num_moduli,       // #moduli
    const T *const __restrict__ A,   // input (lda * k)
    const size_t lda,                // leading dimension
    int8_t *const __restrict__ A8i,  // output (lda8i * (m+pad))
    const size_t lda8i,              // leading dimension
    int16_t *const __restrict__ sftA // exponent of shift values (m+pad)
) {
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];

    const auto rowBase = blockIdx.x * TILE_DIM;
    const auto colBase = blockIdx.y * TILE_DIM;

    const auto in_row              = rowBase + threadIdx.x;
    const auto in_col              = colBase + threadIdx.y;
    const int16_t sft              = (in_row < m) ? -sftA[in_row] : Tconst<int16_t>::zero();
    const T Atmp                   = (in_row < m && in_col < k) ? A[in_col * lda + in_row] : Tconst<T>::zero();
    tile[threadIdx.y][threadIdx.x] = T2int_fp<T>(Atmp, sft);
    __syncthreads();

    const T in = tile[threadIdx.x][threadIdx.y];

    const auto out_col = rowBase + threadIdx.y;
    const auto out_row = colBase + threadIdx.x;
    if (out_col >= m || out_row >= lda8i) return;

    int8_t *const __restrict__ out = A8i + out_col * lda8i + out_row;
    {
        // mod 256
        out[0] = mod_8i_256<T, ITER>(in);
    }
    for (unsigned j = 1; j < num_moduli; ++j) {
        auto val        = table::readtab<T>(j - 1);
        out[j * incA8i] = mod_8i<T, ITER>(in, val);
    }
}

//------------------------------
// Convert trunc(B*diag(2^sftB)) to B8i
//------------------------------
template <typename T, int ITER>
__forceinline__ __device__ void scalingB_device(
    const size_t k,                 // size(B,1)
    const size_t incB8i,            // ldb8i / 4 * n
    const unsigned num_moduli,      // #moduli
    const T *const __restrict__ in, // input (ldb * n)
    char4 *const __restrict__ out,  // output (ldb8i / 4 * n)
    const size_t ldb8i,             // leading dimension ldb8i / 4
    const int16_t sft               // shift value
) {
    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;

    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        T in0 = in[idx];
        T in1 = in[idx + 1];
        T in2 = in[idx + 2];
        T in3 = in[idx + 3];

        in0 = T2int_fp<T>(in0, sft);
        in1 = T2int_fp<T>(in1, sft);
        in2 = T2int_fp<T>(in2, sft);
        in3 = T2int_fp<T>(in3, sft);

        {
            // mod 256
            char4 out4;
            out4.x = mod_8i_256<T, ITER>(in0);
            out4.y = mod_8i_256<T, ITER>(in1);
            out4.z = mod_8i_256<T, ITER>(in2);
            out4.w = mod_8i_256<T, ITER>(in3);

            out[i] = out4;
        }
        for (unsigned j = 1; j < num_moduli; ++j) {
            const auto val = table::readtab<T>(j - 1);

            char4 out4;
            out4.x = mod_8i<T, ITER>(in0, val);
            out4.y = mod_8i<T, ITER>(in1, val);
            out4.z = mod_8i<T, ITER>(in2, val);
            out4.w = mod_8i<T, ITER>(in3, val);

            out[j * incB8i + i] = out4;
        }
    }
    for (; i < ldb8i; i += blockDim.x) {
        unsigned idx = i << 2;

        T in0 = (idx < k) ? in[idx] : Tconst<T>::zero();
        T in1 = (idx + 1 < k) ? in[idx + 1] : Tconst<T>::zero();
        T in2 = (idx + 2 < k) ? in[idx + 2] : Tconst<T>::zero();
        T in3 = (idx + 3 < k) ? in[idx + 3] : Tconst<T>::zero();

        in0 = T2int_fp<T>(in0, sft);
        in1 = T2int_fp<T>(in1, sft);
        in2 = T2int_fp<T>(in2, sft);
        in3 = T2int_fp<T>(in3, sft);

        {
            // mod 256
            char4 out4;
            out4.x = mod_8i_256<T, ITER>(in0);
            out4.y = mod_8i_256<T, ITER>(in1);
            out4.z = mod_8i_256<T, ITER>(in2);
            out4.w = mod_8i_256<T, ITER>(in3);

            out[i] = out4;
        }
        for (unsigned j = 1; j < num_moduli; ++j) {
            const auto val = table::readtab<T>(j - 1);

            char4 out4;
            out4.x = mod_8i<T, ITER>(in0, val);
            out4.y = mod_8i<T, ITER>(in1, val);
            out4.z = mod_8i<T, ITER>(in2, val);
            out4.w = mod_8i<T, ITER>(in3, val);

            out[j * incB8i + i] = out4;
        }
    }
}

//------------------------------
// Convert trunc(B*diag(2^sftB)) to B8i
//------------------------------
template <typename T, int ITER>
__global__ void scalingB_kernel(
    const size_t k,                   // size(B,1)
    const size_t incB8i,              // ldb8i / 4 * n
    const unsigned num_moduli,        // #moduli
    const T *const __restrict__ B,    // input (ldb * n)
    const size_t ldb,                 // leading dimension
    char4 *const __restrict__ B8i,    // output (ldb8i / 4 * n)
    const size_t ldb8i,               // leading dimension ldb8i / 4
    int16_t *const __restrict__ sftB, // exponent of shift values
    const float log2P                 // log2(P-1)/2 - 1.5
) {
    __shared__ T shm[64];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    T vecnrm;
    const T amax      = find_amax_and_nrm<T>(in, k, shm, vecnrm);
    const int16_t sft = compute_sft<T>(amax, vecnrm, log2P);
    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }

    char4 *const __restrict__ out = B8i + col_idx * ldb8i;
    scalingB_device<T, ITER>(k, incB8i, num_moduli, in, out, ldb8i, sft);
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
    const float log2P,            // fld(log2(P-1)/2 - 1.5)
    const bool skip_scalA,        // false (unskip scaling_A) or true (skip scaling_A)
    const bool skip_scalB         // false (unskip scaling_B) or true (skip scaling_B)
) {
    if (!skip_scalA) {
        if (op_A == CUBLAS_OP_N) {
            // m*k -> k*m
            dim3 grid((m + (TILE_DIM - 1)) / TILE_DIM, (lda8i + (TILE_DIM - 1)) / TILE_DIM);
            constexpr dim3 threads1(TILE_DIM, TILE_DIM);
            compute_sftA_kernel<T><<<grid.x, threads1>>>(m, k, A, lda, sftA, log2P);
            scalingA_kernel<T, ITER><<<grid, threads1>>>(m, k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA);
        } else {
            // k*m -> k*m
            scalingB_kernel<T, ITER><<<m, threads_scaling>>>(k, incA8i >> 2, num_moduli, A, lda, reinterpret_cast<char4 *>(A8i), lda8i >> 2, sftA, log2P);
        }
    }

    if (!skip_scalB) {
        if (op_B == CUBLAS_OP_N) {
            // k*n -> k*n
            scalingB_kernel<T, ITER><<<n, threads_scaling>>>(k, incB8i >> 2, num_moduli, B, ldb, reinterpret_cast<char4 *>(B8i), ldb8i >> 2, sftB, log2P);
        } else {
            // n*k -> k*n
            dim3 grid((n + (TILE_DIM - 1)) / TILE_DIM, (ldb8i + (TILE_DIM - 1)) / TILE_DIM);
            constexpr dim3 threads1(TILE_DIM, TILE_DIM);
            compute_sftA_kernel<T><<<grid.x, threads1>>>(n, k, B, ldb, sftB, log2P);
            scalingA_kernel<T, ITER><<<grid, threads1>>>(n, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB);
        }
    }
}

//------------------------------
// Interface!!
//------------------------------
template <typename T>
__inline__ void scaling(
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
    const unsigned table_idx,     // index for table
    const bool skip_scalA,        // false (unskip scaling_A) or true (skip scaling_A)
    const bool skip_scalB         // false (unskip scaling_B) or true (skip scaling_B)
) {
    const float log2P = table::fast::log2P[table_idx]; // fld(log2(P-1)/2 - 1.5)

    if (num_moduli <= threshold<T>::iter1) {
        scaling_launch<T, 1>(op_A, op_B, m, n, k, num_moduli,
                             A, lda, A8i, lda8i, incA8i, sftA,
                             B, ldb, B8i, ldb8i, incB8i, sftB,
                             log2P, skip_scalA, skip_scalB);

    } else if (num_moduli <= threshold<T>::iter2) {
        scaling_launch<T, 2>(op_A, op_B, m, n, k, num_moduli,
                             A, lda, A8i, lda8i, incA8i, sftA,
                             B, ldb, B8i, ldb8i, incB8i, sftB,
                             log2P, skip_scalA, skip_scalB);

    } else if (num_moduli <= threshold<T>::iter3) {
        scaling_launch<T, 3>(op_A, op_B, m, n, k, num_moduli,
                             A, lda, A8i, lda8i, incA8i, sftA,
                             B, ldb, B8i, ldb8i, incB8i, sftB,
                             log2P, skip_scalA, skip_scalB);

    } else {
        scaling_launch<T, 4>(op_A, op_B, m, n, k, num_moduli,
                             A, lda, A8i, lda8i, incA8i, sftA,
                             B, ldb, B8i, ldb8i, incB8i, sftB,
                             log2P, skip_scalA, skip_scalB);
    }
}

} // namespace fast
} // namespace real
} // namespace oz2
