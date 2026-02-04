#pragma once
#include "common.hpp"
#include "template_math.hpp"

namespace oz2 {
namespace complex {
namespace fast {

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
    using U = underlying_t<T>;
    __shared__ U samax[TILE_DIM][TILE_DIM + 1];
    __shared__ U ssum[TILE_DIM][TILE_DIM + 1];

    unsigned row_idx = blockIdx.x * TILE_DIM + threadIdx.x;

    U sum;
    U amax = find_amax_and_nrm_tile<T>(m, k, row_idx, A, lda, samax, ssum, sum);

    row_idx = blockIdx.x * TILE_DIM + threadIdx.y;
    if (row_idx < m && threadIdx.x == 0) {
        int16_t sft   = oz2::real::fast::compute_sft<U>(amax, sum, log2P);
        sftA[row_idx] = -sft;
    }
}

//------------------------------
// Convert trunc(diag(2^sftA)*A) to A8i
//------------------------------
template <typename T, int ITER, bool CONJ = false>
__global__ void scalingA_kernel(
    const size_t m,                   // size(A,1)
    const size_t k,                   // size(A,2)
    const size_t incA8i,              // lda8i * (m+pad)
    const unsigned num_moduli,        // #moduli
    const T *const __restrict__ A,    // input (lda * k)
    const size_t lda,                 // leading dimension
    int8_t *const __restrict__ A8i_1, // output (lda8i * (m+pad))
    int8_t *const __restrict__ A8i_2, // output (lda8i * (m+pad))
    int8_t *const __restrict__ A8i_3, // output (lda8i * (m+pad))
    const size_t lda8i,               // leading dimension
    int16_t *const __restrict__ sftA  // exponent of shift values (m+pad)
) {
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];

    const auto rowBase = blockIdx.x * TILE_DIM;
    const auto colBase = blockIdx.y * TILE_DIM;

    const auto in_row              = rowBase + threadIdx.x;
    const auto in_col              = colBase + threadIdx.y;
    const int16_t sft              = (in_row < m) ? -sftA[in_row] : Tconst<int16_t>::zero();
    const T Atmp                   = (in_row < m && in_col < k) ? A[in_col * lda + in_row] : Tconst<T>::zero();
    tile[threadIdx.y][threadIdx.x] = T2int_fp<T>(conj<T, CONJ>(Atmp), sft);
    __syncthreads();

    const T in = tile[threadIdx.x][threadIdx.y];

    const auto out_col = rowBase + threadIdx.y;
    const auto out_row = colBase + threadIdx.x;
    if (out_col >= m || out_row >= lda8i) return;

    int8_t *const __restrict__ out_1 = A8i_1 + out_col * lda8i + out_row;
    int8_t *const __restrict__ out_2 = A8i_2 + out_col * lda8i + out_row;
    int8_t *const __restrict__ out_3 = A8i_3 + out_col * lda8i + out_row;

    {
        // mod 256
        char3 out_val = mod_8i_256_complex<T, ITER>(in);

        out_1[0] = out_val.x;
        out_2[0] = out_val.y;
        out_3[0] = out_val.z;
    }
    for (unsigned j = 1; j < num_moduli; ++j) {
        auto val      = table::readtab<underlying_t<T>>(j - 1);
        int p         = table::MODULI_I[j - 1].x;
        char3 out_val = mod_8i_complex<T, ITER>(in, val, p);

        out_1[j * incA8i] = out_val.x;
        out_2[j * incA8i] = out_val.y;
        out_3[j * incA8i] = out_val.z;
    }
}

//------------------------------
// Convert trunc(B*diag(2^sftB)) to B8i
//------------------------------
template <typename T, int ITER, bool CONJ = false>
__forceinline__ __device__ void scalingB_device(
    const size_t k,                  // size(B,1)
    const size_t incB8i,             // ldb8i / 4 * n
    const unsigned num_moduli,       // #moduli
    const T *const __restrict__ in,  // input (ldb * n)
    char4 *const __restrict__ out_1, // output (ldb8i / 4 * n)
    char4 *const __restrict__ out_2, // output (ldb8i / 4 * n)
    char4 *const __restrict__ out_3, // output (ldb8i / 4 * n)
    const size_t ldb8i,              // leading dimension ldb8i / 4
    const int16_t sft                // shift value
) {
    using U       = underlying_t<T>;
    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;

    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        T in0 = conj<T, CONJ>(in[idx]);
        T in1 = conj<T, CONJ>(in[idx + 1]);
        T in2 = conj<T, CONJ>(in[idx + 2]);
        T in3 = conj<T, CONJ>(in[idx + 3]);

        in0 = T2int_fp<T>(in0, sft);
        in1 = T2int_fp<T>(in1, sft);
        in2 = T2int_fp<T>(in2, sft);
        in3 = T2int_fp<T>(in3, sft);

        {
            // mod 256
            char3 out0 = mod_8i_256_complex<T, ITER>(in0);
            char3 out1 = mod_8i_256_complex<T, ITER>(in1);
            char3 out2 = mod_8i_256_complex<T, ITER>(in2);
            char3 out3 = mod_8i_256_complex<T, ITER>(in3);

            char4 out4_1{out0.x, out1.x, out2.x, out3.x};
            char4 out4_2{out0.y, out1.y, out2.y, out3.y};
            char4 out4_3{out0.z, out1.z, out2.z, out3.z};

            out_1[i] = out4_1;
            out_2[i] = out4_2;
            out_3[i] = out4_3;
        }
        for (unsigned j = 1; j < num_moduli; ++j) {
            auto val = table::readtab<U>(j - 1);
            int p    = table::MODULI_I[j - 1].x;

            char3 out0 = mod_8i_complex<T, ITER>(in0, val, p);
            char3 out1 = mod_8i_complex<T, ITER>(in1, val, p);
            char3 out2 = mod_8i_complex<T, ITER>(in2, val, p);
            char3 out3 = mod_8i_complex<T, ITER>(in3, val, p);

            char4 out4_1{out0.x, out1.x, out2.x, out3.x};
            char4 out4_2{out0.y, out1.y, out2.y, out3.y};
            char4 out4_3{out0.z, out1.z, out2.z, out3.z};

            out_1[j * incB8i + i] = out4_1;
            out_2[j * incB8i + i] = out4_2;
            out_3[j * incB8i + i] = out4_3;
        }
    }
    for (; i < ldb8i; i += blockDim.x) {
        unsigned idx = i << 2;

        T in0 = (idx < k) ? conj<T, CONJ>(in[idx]) : Tconst<T>::zero();
        T in1 = (idx + 1 < k) ? conj<T, CONJ>(in[idx + 1]) : Tconst<T>::zero();
        T in2 = (idx + 2 < k) ? conj<T, CONJ>(in[idx + 2]) : Tconst<T>::zero();
        T in3 = (idx + 3 < k) ? conj<T, CONJ>(in[idx + 3]) : Tconst<T>::zero();

        in0 = T2int_fp<T>(in0, sft);
        in1 = T2int_fp<T>(in1, sft);
        in2 = T2int_fp<T>(in2, sft);
        in3 = T2int_fp<T>(in3, sft);

        {
            // mod 256
            char3 out0 = mod_8i_256_complex<T, ITER>(in0);
            char3 out1 = mod_8i_256_complex<T, ITER>(in1);
            char3 out2 = mod_8i_256_complex<T, ITER>(in2);
            char3 out3 = mod_8i_256_complex<T, ITER>(in3);

            char4 out4_1{out0.x, out1.x, out2.x, out3.x};
            char4 out4_2{out0.y, out1.y, out2.y, out3.y};
            char4 out4_3{out0.z, out1.z, out2.z, out3.z};

            out_1[i] = out4_1;
            out_2[i] = out4_2;
            out_3[i] = out4_3;
        }
        for (unsigned j = 1; j < num_moduli; ++j) {
            auto val = table::readtab<U>(j - 1);
            int p    = table::MODULI_I[j - 1].x;

            char3 out0 = mod_8i_complex<T, ITER>(in0, val, p);
            char3 out1 = mod_8i_complex<T, ITER>(in1, val, p);
            char3 out2 = mod_8i_complex<T, ITER>(in2, val, p);
            char3 out3 = mod_8i_complex<T, ITER>(in3, val, p);

            char4 out4_1{out0.x, out1.x, out2.x, out3.x};
            char4 out4_2{out0.y, out1.y, out2.y, out3.y};
            char4 out4_3{out0.z, out1.z, out2.z, out3.z};

            out_1[j * incB8i + i] = out4_1;
            out_2[j * incB8i + i] = out4_2;
            out_3[j * incB8i + i] = out4_3;
        }
    }
}

//------------------------------
// Convert trunc(B*diag(2^sftB)) to B8i
//------------------------------
template <typename T, int ITER, bool CONJ = false>
__global__ void scalingB_kernel(
    const size_t k,                   // size(B,1)
    const size_t incB8i,              // ldb8i / 4 * n
    const unsigned num_moduli,        // #moduli
    const T *const __restrict__ B,    // input (ldb * n)
    const size_t ldb,                 // leading dimension
    char4 *const __restrict__ B8i_1,  // output (ldb8i / 4 * n)
    char4 *const __restrict__ B8i_2,  // output (ldb8i / 4 * n)
    char4 *const __restrict__ B8i_3,  // output (ldb8i / 4 * n)
    const size_t ldb8i,               // leading dimension ldb8i / 4
    int16_t *const __restrict__ sftB, // exponent of shift values
    const float log2P                 // log2(P-1)/2 - 1.5
) {
    using U = underlying_t<T>;
    __shared__ U shm[64];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    U vecnrm;
    const U amax      = find_amax_and_nrm<T>(in, k, shm, vecnrm);
    const int16_t sft = oz2::real::fast::compute_sft<U>(amax, vecnrm, log2P);
    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }

    char4 *const __restrict__ out_1 = B8i_1 + col_idx * ldb8i;
    char4 *const __restrict__ out_2 = B8i_2 + col_idx * ldb8i;
    char4 *const __restrict__ out_3 = B8i_3 + col_idx * ldb8i;
    scalingB_device<T, ITER, CONJ>(k, incB8i, num_moduli, in, out_1, out_2, out_3, ldb8i, sft);
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
    int8_t *const *const A8i,     // output (lda8i * (m+pad))
    const size_t lda8i,           // leading dimension
    const size_t incA8i,          // increment between the A8i
    int16_t *const sftA,          // exponent of shift values for rows of A
    const T *const B,             // input
    const size_t ldb,             // leading dimension
    int8_t *const *const B8i,     // output (ldb8i * n)
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
            scalingA_kernel<T, ITER><<<grid, threads1>>>(m, k, incA8i, num_moduli, A, lda, A8i[0], A8i[1], A8i[2], lda8i, sftA);
        } else {
            // k*m -> k*m
            if (op_A == CUBLAS_OP_T) {
                scalingB_kernel<T, ITER, false><<<m, threads_scaling>>>(k, incA8i >> 2, num_moduli, A, lda,
                                                                        reinterpret_cast<char4 *>(A8i[0]),
                                                                        reinterpret_cast<char4 *>(A8i[1]),
                                                                        reinterpret_cast<char4 *>(A8i[2]),
                                                                        lda8i >> 2, sftA, log2P);
            } else {
                scalingB_kernel<T, ITER, true><<<m, threads_scaling>>>(k, incA8i >> 2, num_moduli, A, lda,
                                                                       reinterpret_cast<char4 *>(A8i[0]),
                                                                       reinterpret_cast<char4 *>(A8i[1]),
                                                                       reinterpret_cast<char4 *>(A8i[2]),
                                                                       lda8i >> 2, sftA, log2P);
            }
        }
    }

    if (!skip_scalB) {
        if (op_B == CUBLAS_OP_N) {
            // k*n -> k*n
            scalingB_kernel<T, ITER><<<n, threads_scaling>>>(k, incB8i >> 2, num_moduli, B, ldb,
                                                             reinterpret_cast<char4 *>(B8i[0]),
                                                             reinterpret_cast<char4 *>(B8i[1]),
                                                             reinterpret_cast<char4 *>(B8i[2]),
                                                             ldb8i >> 2, sftB, log2P);
        } else {
            // n*k -> k*n
            dim3 grid((n + (TILE_DIM - 1)) / TILE_DIM, (ldb8i + (TILE_DIM - 1)) / TILE_DIM);
            constexpr dim3 threads1(TILE_DIM, TILE_DIM);
            compute_sftA_kernel<T><<<grid.x, threads1>>>(n, k, B, ldb, sftB, log2P);
            if (op_B == CUBLAS_OP_T) {
                scalingA_kernel<T, ITER, false><<<grid, threads1>>>(n, k, incB8i, num_moduli, B, ldb, B8i[0], B8i[1], B8i[2], ldb8i, sftB);
            } else {
                scalingA_kernel<T, ITER, true><<<grid, threads1>>>(n, k, incB8i, num_moduli, B, ldb, B8i[0], B8i[1], B8i[2], ldb8i, sftB);
            }
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
    int8_t *const *const A8i,     // output (lda8i * (m+pad))
    const size_t lda8i,           // leading dimension
    const size_t incA8i,          // increment between the A8i
    int16_t *const sftA,          // exponent of shift values for rows of A
    const T *const B,             // input
    const size_t ldb,             // leading dimension
    int8_t *const *const B8i,     // output (ldb8i * n)
    const size_t ldb8i,           // leading dimension
    const size_t incB8i,          // increment between the B8i
    int16_t *const sftB,          // exponent of shift values for cols of B
    const unsigned table_idx,     // index for table
    const bool skip_scalA,        // false (unskip scaling_A) or true (skip scaling_A)
    const bool skip_scalB         // false (unskip scaling_B) or true (skip scaling_B)
) {
    const float log2P = table::fast::log2P[table_idx]; // fld(log2(P-1)/2 - 1.5)

    using U = underlying_t<T>;
    if (num_moduli <= threshold<U>::iter1) {
        scaling_launch<T, 1>(op_A, op_B, m, n, k, num_moduli,
                             A, lda, A8i, lda8i, incA8i, sftA,
                             B, ldb, B8i, ldb8i, incB8i, sftB,
                             log2P, skip_scalA, skip_scalB);

    } else if (num_moduli <= threshold<U>::iter2) {
        scaling_launch<T, 2>(op_A, op_B, m, n, k, num_moduli,
                             A, lda, A8i, lda8i, incA8i, sftA,
                             B, ldb, B8i, ldb8i, incB8i, sftB,
                             log2P, skip_scalA, skip_scalB);

    } else if (num_moduli <= threshold<U>::iter3) {
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
} // namespace complex
} // namespace oz2
