#pragma once
#include "self_hipify.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <vector>

namespace eval {

//------------------------------
// dd
//------------------------------
namespace dd {

#pragma clang optimize off
inline void two_sum(const double a, const double b, double &c, double &d) {
    c        = a + b;
    double s = c - a;
    double t = b - s;
    double u = c - s;
    d        = (a - u) + t;
}
#pragma clang optimize on

#pragma clang optimize off
inline void two_sub(const double a, const double b, double &c, double &d) {
    c        = a - b;
    double s = c - a;
    double t = b + s;
    double u = c - s;
    d        = (a - u) - t;
}
#pragma clang optimize on

#pragma clang optimize off
inline void fast_two_sum(const double a, const double b, double &c, double &d) {
    c = a + b;
    d = (a - c) + b;
}
#pragma clang optimize on

#pragma clang optimize off
inline void two_prod(const double a, const double b, double &c, double &d) {
    c = a * b;
    d = std::fma(a, b, -c);
}
#pragma clang optimize on

#pragma clang optimize off
inline void add(const double a1,
                const double a2,
                const double b1,
                const double b2,
                double &c1,
                double &c2) {
    dd::two_sum(a1, b1, c1, c2);
    c2 += a2;
    c2 += b2;
    dd::fast_two_sum(c1, c2, c1, c2);
}
#pragma clang optimize on

#pragma clang optimize off
inline void sub(const double a1,
                const double a2,
                const double b1,
                const double b2,
                double &c1,
                double &c2) {
    dd::two_sub(a1, b1, c1, c2);
    c2 += a2;
    c2 -= b2;
    dd::fast_two_sum(c1, c2, c1, c2);
}
#pragma clang optimize on

#pragma clang optimize off
inline void mul(const double a1,
                const double a2,
                const double b1,
                const double b2,
                double &c1,
                double &c2) {
    dd::two_prod(a1, b1, c1, c2);
    c2 = std::fma(a2, b1, std::fma(a1, b2, c2));
    dd::fast_two_sum(c1, c2, c1, c2);
}
#pragma clang optimize on

#pragma clang optimize off
inline void div(const double a1,
                const double a2,
                const double b1,
                const double b2,
                double &c1,
                double &c2) {
    double s, t;
    c1 = a1 / b1;
    dd::two_prod(c1, b1, s, t);
    double u = a1 - s;
    u -= t;
    u += a2;
    u = std::fma(-c1, b2, u);
    u /= b1;
    dd::fast_two_sum(c1, u, c1, c2);
}
#pragma clang optimize on

void simple_gemm(
    size_t m,
    size_t p,
    size_t n,
    double *A,  // m*k
    double *B,  // k*n
    double *C1, // m*n result
    double *C2) // m*n result
{
    constexpr size_t block_size = 64;
    constexpr double dzero      = 0.0;

#pragma omp parallel for
    for (size_t i = 0; i < m * p; i++) {
        C1[i] = 0.0;
        C2[i] = 0.0;
    }

#pragma omp parallel
    {
        double *C1_local = (double *)calloc(m * p, sizeof(double));
        double *C2_local = (double *)calloc(m * p, sizeof(double));

#pragma omp for collapse(2) schedule(static)
        for (int ii = 0; ii < m; ii += block_size) {
            for (int jj = 0; jj < p; jj += block_size) {
                for (int kk = 0; kk < n; kk += block_size) {
                    for (int i = ii; i < ii + block_size && i < m; i++) {
                        for (int j = jj; j < jj + block_size && j < p; j++) {
                            double sum1 = 0.0;
                            double sum2 = 0.0;
                            for (int k = kk; k < kk + block_size && k < n; k++) {
                                double ab1, ab2;
                                dd::mul(A[i * n + k], dzero, B[k * p + j], dzero, ab1, ab2);
                                dd::add(ab1, ab2, sum1, sum2, sum1, sum2);
                            }
                            dd::add(C1_local[i * p + j], C2_local[i * p + j], sum1, sum2, C1_local[i * p + j], C2_local[i * p + j]);
                        }
                    }
                }
            }
        }

#pragma omp critical
        for (size_t i = 0; i < m * p; i++) {
            dd::add(C1[i], C2[i], C1_local[i], C2_local[i], C1[i], C2[i]);
        }

        free(C1_local);
        free(C2_local);
    }
}

} // namespace dd

//------------------------------
// dd on gpu
//------------------------------
namespace dd_gpu {

#pragma clang optimize off
__device__ __forceinline__ void two_sum(const double a, const double b, double &c, double &d) {
    c        = a + b;
    double s = c - a;
    double t = b - s;
    double u = c - s;
    d        = (a - u) + t;
}
#pragma clang optimize on

#pragma clang optimize off
__device__ __forceinline__ void two_sub(const double a, const double b, double &c, double &d) {
    c        = a - b;
    double s = c - a;
    double t = b + s;
    double u = c - s;
    d        = (a - u) - t;
}
#pragma clang optimize on

#pragma clang optimize off
__device__ __forceinline__ void fast_two_sum(const double a, const double b, double &c, double &d) {
    c = a + b;
    d = (a - c) + b;
}
#pragma clang optimize on

#pragma clang optimize off
__device__ __forceinline__ void two_prod(const double a, const double b, double &c, double &d) {
    c = a * b;
    d = fma(a, b, -c);
}
#pragma clang optimize on

#pragma clang optimize off
__device__ __forceinline__ void add(const double a1,
                                    const double a2,
                                    const double b1,
                                    const double b2,
                                    double &c1,
                                    double &c2) {
    dd_gpu::two_sum(a1, b1, c1, c2);
    c2 += a2;
    c2 += b2;
    dd_gpu::fast_two_sum(c1, c2, c1, c2);
}
#pragma clang optimize on

#pragma clang optimize off
__device__ __forceinline__ void sub(const double a1,
                                    const double a2,
                                    const double b1,
                                    const double b2,
                                    double &c1,
                                    double &c2) {
    dd_gpu::two_sub(a1, b1, c1, c2);
    c2 += a2;
    c2 -= b2;
    dd_gpu::fast_two_sum(c1, c2, c1, c2);
}
#pragma clang optimize on

#pragma clang optimize off
__device__ __forceinline__ void mul(const double a1,
                                    const double a2,
                                    const double b1,
                                    const double b2,
                                    double &c1,
                                    double &c2) {
    dd_gpu::two_prod(a1, b1, c1, c2);
    c2 = fma(a2, b1, fma(a1, b2, c2));
    dd_gpu::fast_two_sum(c1, c2, c1, c2);
}
#pragma clang optimize on

#pragma clang optimize off
__device__ __forceinline__ void div(const double a1,
                                    const double a2,
                                    const double b1,
                                    const double b2,
                                    double &c1,
                                    double &c2) {
    double s, t;
    c1 = a1 / b1;
    dd_gpu::two_prod(c1, b1, s, t);
    double u = a1 - s;
    u -= t;
    u += a2;
    u = fma(-c1, b2, u);
    u /= b1;
    dd_gpu::fast_two_sum(c1, u, c1, c2);
}
#pragma clang optimize on

template <typename T>
__device__ __forceinline__ T load_A_element(const T *A,
                                            size_t m,
                                            size_t k,
                                            size_t row,
                                            size_t t,
                                            cublasOperation_t op_A) //
{
    if (op_A == CUBLAS_OP_N) {
        return __ldg(A + row + t * m);
    } else {
        return __ldg(A + t + row * k);
    }
}

template <typename T>
__device__ __forceinline__ T load_B_element(const T *B,
                                            size_t k,
                                            size_t n,
                                            size_t t,
                                            size_t col,
                                            cublasOperation_t op_B) //
{
    if (op_B == CUBLAS_OP_N) {
        return __ldg(B + t + col * k);
    } else {
        return __ldg(B + col + t * n);
    }
}

__global__ void simple_gemm_device(size_t m,
                                   size_t n,
                                   size_t k,
                                   const double *A,
                                   const double *B,
                                   double *C1,
                                   double *C2,
                                   cublasOperation_t op_A,
                                   cublasOperation_t op_B) //
{
    const int TILE = 32;
    __shared__ double Asub[TILE][TILE + 1];
    __shared__ double Bsub[TILE][TILE + 1];

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    double sum1 = 0.0;
    double sum2 = 0.0;

    int numTiles = (int)((k + TILE - 1) / TILE);

    for (int t = 0; t < numTiles; ++t) {
        size_t a_col = t * TILE + threadIdx.x;
        if (row < m && a_col < k) {
            Asub[threadIdx.y][threadIdx.x] = load_A_element(A, m, k, row, a_col, op_A);
        } else {
            Asub[threadIdx.y][threadIdx.x] = 0.0;
        }

        size_t b_row = t * TILE + threadIdx.y;
        if (col < n && b_row < k) {
            Bsub[threadIdx.y][threadIdx.x] = load_B_element(B, k, n, b_row, col, op_B);
        } else {
            Bsub[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE; ++i) {
            double a = Asub[threadIdx.y][i];
            double b = Bsub[i][threadIdx.x];
            double ab1, ab2;
            dd_gpu::two_prod(a, b, ab1, ab2);
            dd_gpu::add(ab1, ab2, sum1, sum2, sum1, sum2);
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C1[col * m + row] = sum1;
        C2[col * m + row] = sum2;
    }
}

void simple_gemm(size_t m,
                 size_t n,
                 size_t k,
                 const double *A,
                 const double *B,
                 double *C1,
                 double *C2,
                 cublasOperation_t op_A = CUBLAS_OP_N,
                 cublasOperation_t op_B = CUBLAS_OP_N) //
{
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    simple_gemm_device<<<numBlocks, threadsPerBlock>>>(m, n, k, A, B, C1, C2, op_A, op_B);
}

__global__ void simple_gemm_device(size_t m,
                                   size_t n,
                                   size_t k,
                                   const cuDoubleComplex *A,
                                   const cuDoubleComplex *B,
                                   cuDoubleComplex *C1,
                                   cuDoubleComplex *C2,
                                   cublasOperation_t op_A,
                                   cublasOperation_t op_B) //
{
    const int TILE = 32;
    __shared__ cuDoubleComplex Asub[TILE][TILE + 1];
    __shared__ cuDoubleComplex Bsub[TILE][TILE + 1];

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    double sum1_r = 0.0, sum2_r = 0.0;
    double sum1_i = 0.0, sum2_i = 0.0;

    int numTiles = (int)((k + TILE - 1) / TILE);

    for (int t = 0; t < numTiles; ++t) {
        size_t a_col = t * TILE + threadIdx.x;
        if (row < m && a_col < k) {
            Asub[threadIdx.y][threadIdx.x] = load_A_element(A, m, k, row, a_col, op_A);
        } else {
            Asub[threadIdx.y][threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);
        }

        size_t b_row = t * TILE + threadIdx.y;
        if (col < n && b_row < k) {
            Bsub[threadIdx.y][threadIdx.x] = load_B_element(B, k, n, b_row, col, op_B);
        } else {
            Bsub[threadIdx.y][threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE; ++i) {
            cuDoubleComplex a = Asub[threadIdx.y][i];
            cuDoubleComplex b = Bsub[i][threadIdx.x];

            // (ar + i ai)(br + i bi)
            double ar = cuCreal(a), ai = cuCimag(a);
            double br = cuCreal(b), bi = cuCimag(b);

            // --- real contribution: ar*br - ai*bi ---
            double p1, p2;
            dd_gpu::two_prod(ar, br, p1, p2);
            double q1, q2;
            dd_gpu::two_prod(ai, bi, q1, q2);

            double r1, r2;
            dd_gpu::sub(p1, p2, q1, q2, r1, r2);
            dd_gpu::add(sum1_r, sum2_r, r1, r2, sum1_r, sum2_r);

            // --- imag contribution: ar*bi + ai*br ---
            dd_gpu::two_prod(ar, bi, p1, p2);
            dd_gpu::two_prod(ai, br, q1, q2);
            dd_gpu::add(p1, p2, q1, q2, r1, r2);
            dd_gpu::add(sum1_i, sum2_i, r1, r2, sum1_i, sum2_i);
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C1[col * m + row] = make_cuDoubleComplex(sum1_r, sum1_i);
        C2[col * m + row] = make_cuDoubleComplex(sum2_r, sum2_i);
    }
}

void simple_gemm(size_t m,
                 size_t n,
                 size_t k,
                 const cuDoubleComplex *A,
                 const cuDoubleComplex *B,
                 cuDoubleComplex *C1,
                 cuDoubleComplex *C2,
                 cublasOperation_t op_A = CUBLAS_OP_N,
                 cublasOperation_t op_B = CUBLAS_OP_N) //
{
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    simple_gemm_device<<<numBlocks, threadsPerBlock>>>(m, n, k, A, B, C1, C2, op_A, op_B);
}

} // namespace dd_gpu

//------------------------------
// evaluate error
//------------------------------
namespace err {

void gemm_err(const size_t m,
              const size_t n,
              double *const C,        // calculated value
              const double *const C1, // true value
              const double *const C2, // true value
              double &err1,           // max
              double &err2)           // median
{
    size_t sizeC = m * n;

#pragma omp parallel for
    for (size_t i = 0; i < sizeC; i++) {
        double tmp1, tmp2, tmp3 = 0.0, tmp4;
        dd::sub(C[i], tmp3, C1[i], C2[i], tmp1, tmp2);
        dd::div(tmp1, tmp2, C1[i], C2[i], tmp3, tmp4);
        C[i] = std::fabs(tmp3);
    }

    std::sort(C, C + sizeC);
    err1 = C[sizeC - 1];
    err2 = (sizeC & 1) ? C[sizeC / 2] : ((C[sizeC / 2] + C[sizeC / 2 - 1]) * 0.5);
}

void gemm_err(const size_t m,
              const size_t n,
              float *const C,         // calculated value
              const double *const C1, // true value
              double &err1,           // max
              double &err2)           // median
{
    size_t sizeC = m * n;

#pragma omp parallel for
    for (size_t i = 0; i < sizeC; i++) {
        double tmp = (double(C[i]) - C1[i]) / C1[i];
        C[i]       = float(std::fabs(tmp));
    }

    std::sort(C, C + sizeC);
    err1 = double(C[sizeC - 1]);
    err2 = (sizeC & 1) ? double(C[sizeC / 2]) : ((double(C[sizeC / 2]) + double(C[sizeC / 2 - 1])) * 0.5);
}

void gemm_err(const size_t m,
              const size_t n,
              cuDoubleComplex *const C,        // calculated value
              const cuDoubleComplex *const C1, // true value
              const cuDoubleComplex *const C2, // true value
              double &err1,                    // max
              double &err2)                    // median
{
    size_t sizeC = m * n;

#pragma omp parallel for
    for (size_t i = 0; i < sizeC; i++) {
        double tmp1, tmp2, tmp3, tmp4, res1, res2;
        tmp3 = 0.0;
        dd::sub(cuCreal(C[i]), tmp3, cuCreal(C1[i]), cuCreal(C2[i]), tmp1, tmp2);
        dd::div(tmp1, tmp2, cuCreal(C1[i]), cuCreal(C2[i]), res1, tmp4);

        tmp3 = 0.0;
        dd::sub(cuCimag(C[i]), tmp3, cuCimag(C1[i]), cuCimag(C2[i]), tmp1, tmp2);
        dd::div(tmp1, tmp2, cuCimag(C1[i]), cuCimag(C2[i]), res2, tmp4);

        res1 = std::fabs(res1);
        res2 = std::fabs(res2);
        C[i] = make_cuDoubleComplex(res1, res2);
    }

    double *D    = reinterpret_cast<double *>(C);
    size_t sizeD = sizeC * 2;
    std::sort(D, D + sizeD);
    err1 = D[sizeD - 1];
    err2 = (sizeD & 1) ? D[sizeD / 2] : ((D[sizeD / 2] + D[sizeD / 2 - 1]) * 0.5);
}

void gemm_err(const size_t m,
              const size_t n,
              cuComplex *const C,              // calculated value
              const cuDoubleComplex *const C1, // true value
              double &err1,                    // max
              double &err2)                    // median
{
    size_t sizeC = m * n;

#pragma omp parallel for
    for (size_t i = 0; i < sizeC; i++) {
        double res1 = (double(cuCrealf(C[i])) - cuCreal(C1[i])) / cuCreal(C1[i]);
        double res2 = (double(cuCimagf(C[i])) - cuCimag(C1[i])) / cuCimag(C1[i]);

        res1 = float(std::fabs(res1));
        res2 = float(std::fabs(res2));
        C[i] = make_cuComplex(res1, res2);
    }

    float *D     = reinterpret_cast<float *>(C);
    size_t sizeD = sizeC * 2;
    std::sort(D, D + sizeD);
    err1 = double(D[sizeD - 1]);
    err2 = (sizeD & 1) ? double(D[sizeD / 2]) : ((double(D[sizeD / 2]) + double(D[sizeD / 2 - 1])) * 0.5);
}

} // namespace err

void data_analysis(const size_t m,
                   const size_t n,
                   cuDoubleComplex *A,
                   double &maxA,
                   double &minA,
                   double &medA,       //
                   double &quartile1A, //
                   double &quartile3A  //
) {
    size_t sizeD = m * n * 2;
    double *D    = reinterpret_cast<double *>(A);
    std::sort(D, D + sizeD, [](double a, double b) {
        return std::abs(a) < std::abs(b);
    });
    maxA       = std::abs(D[sizeD - 1]);
    medA       = std::abs(D[sizeD / 2]);
    minA       = std::abs(D[0]);
    quartile1A = std::abs(D[sizeD / 4]);
    quartile3A = std::abs(D[sizeD * 3 / 4]);
}

void data_analysis(const size_t m,
                   const size_t n,
                   double *A,
                   double &maxA,
                   double &minA,
                   double &medA,       //
                   double &quartile1A, //
                   double &quartile3A  //
) {
    size_t sizeD = m * n;
    double *D    = A;
    std::sort(D, D + sizeD, [](double a, double b) {
        return std::abs(a) < std::abs(b);
    });
    maxA       = std::abs(D[sizeD - 1]);
    medA       = std::abs(D[sizeD / 2]);
    minA       = std::abs(D[0]);
    quartile1A = std::abs(D[sizeD / 4]);
    quartile3A = std::abs(D[sizeD * 3 / 4]);
}

} // namespace eval
