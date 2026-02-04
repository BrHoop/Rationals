#pragma once
#include "self_hipify.hpp"

namespace makemat {

#pragma clang optimize off
__global__ void randmat_kernel(size_t m,                      // rows of A
                               size_t n,                      // columns of A
                               float *const A,                // output
                               float phi,                     // difficulty for matrix multiplication
                               const unsigned long long seed) // seed for random numbers
{
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= m * n) return;
    curandState state;
    curand_init(seed, idx, 0, &state);
    const float rand  = static_cast<float>(curand_uniform_double(&state));
    const float randn = static_cast<float>(curand_normal_double(&state));
    A[idx]            = (rand - 0.5f) * expf(randn * phi);
}
#pragma clang optimize on

void randmat(size_t m,                      // rows of A
             size_t n,                      // columns of A
             float *const A,                // output
             float phi,                     // difficulty for matrix multiplication
             const unsigned long long seed) // seed for random numbers
{
    constexpr size_t block_size = 256;
    const size_t grid_size      = (m * n + block_size - 1) / block_size;
    randmat_kernel<<<grid_size, block_size>>>(m, n, A, phi, seed);
    cudaDeviceSynchronize();
}

#pragma clang optimize off
__global__ void randmat_kernel(size_t m,                      // rows of A
                               size_t n,                      // columns of A
                               double *const A,               // output
                               double phi,                    // difficulty for matrix multiplication
                               const unsigned long long seed) // seed for random numbers
{
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= m * n) return;
    curandState state;
    curand_init(seed, idx, 0, &state);
    const double rand  = curand_uniform_double(&state);
    const double randn = curand_normal_double(&state);
    A[idx]             = (rand - 0.5) * exp(randn * phi);
}
#pragma clang optimize on

void randmat(size_t m,                      // rows of A
             size_t n,                      // columns of A
             double *const A,               // output
             double phi,                    // difficulty for matrix multiplication
             const unsigned long long seed) // seed for random numbers
{
    constexpr size_t block_size = 256;
    const size_t grid_size      = (m * n + block_size - 1) / block_size;
    randmat_kernel<<<grid_size, block_size>>>(m, n, A, phi, seed);
    cudaDeviceSynchronize();
}

#pragma clang optimize off
__global__ void randmat_kernel(size_t m,
                               size_t n,
                               cuComplex *const A,
                               float phi,
                               unsigned long long seed) //
{
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= m * n) return;
    curandState state;
    curand_init(seed, idx, 0, &state);

    // real
    float rand_r  = static_cast<float>(curand_uniform_double(&state));
    float randn_r = static_cast<float>(curand_normal_double(&state));
    float real    = (rand_r - 0.5f) * expf(randn_r * phi);

    // img
    float rand_i  = static_cast<float>(curand_uniform_double(&state));
    float randn_i = static_cast<float>(curand_normal_double(&state));
    float imag    = (rand_i - 0.5f) * expf(randn_i * phi);

    A[idx] = make_cuComplex(real, imag);
}
#pragma clang optimize on

void randmat(size_t m,                      // rows of A
             size_t n,                      // columns of A
             cuComplex *const A,            // output
             float phi,                     // difficulty for matrix multiplication
             const unsigned long long seed) // seed for random numbers
{
    constexpr size_t block_size = 256;
    const size_t grid_size      = (m * n + block_size - 1) / block_size;
    randmat_kernel<<<grid_size, block_size>>>(m, n, A, phi, seed);
    cudaDeviceSynchronize();
}

#pragma clang optimize off
__global__ void randmat_kernel(size_t m,
                               size_t n,
                               cuDoubleComplex *const A,
                               double phi,
                               unsigned long long seed) //
{
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= m * n) return;
    curandState state;
    curand_init(seed, idx, 0, &state);

    // real
    double rand_r  = curand_uniform_double(&state);
    double randn_r = curand_normal_double(&state);
    double real    = (rand_r - 0.5) * exp(randn_r * phi);

    // img
    double rand_i  = curand_uniform_double(&state);
    double randn_i = curand_normal_double(&state);
    double imag    = (rand_i - 0.5) * exp(randn_i * phi);

    A[idx] = make_cuDoubleComplex(real, imag);
}
#pragma clang optimize on

void randmat(size_t m,                      // rows of A
             size_t n,                      // columns of A
             cuDoubleComplex *const A,      // output
             double phi,                    // difficulty for matrix multiplication
             const unsigned long long seed) // seed for random numbers
{
    constexpr size_t block_size = 256;
    const size_t grid_size      = (m * n + block_size - 1) / block_size;
    randmat_kernel<<<grid_size, block_size>>>(m, n, A, phi, seed);
    cudaDeviceSynchronize();
}

__global__ void ones_kernel(size_t sizeA, int8_t *const __restrict__ A) {
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= sizeA) return;
    A[idx] = 1;
}

void ones(size_t sizeA, int8_t *const A) {
    constexpr size_t block_size = 256;
    const size_t grid_size      = (sizeA + block_size - 1) / block_size;
    ones_kernel<<<grid_size, block_size>>>(sizeA, A);
    cudaDeviceSynchronize();
}

__global__ void f2d_kernel(size_t sizeA, const float *const __restrict__ in, double *const __restrict__ out) {
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= sizeA) return;
    out[idx] = static_cast<double>(in[idx]);
}

void f2d(size_t m,              // rows of A
         size_t n,              // columns of A
         const float *const in, // input
         double *const out)     // output
{
    constexpr size_t block_size = 256;
    const size_t grid_size      = (m * n + block_size - 1) / block_size;
    f2d_kernel<<<grid_size, block_size>>>(m * n, in, out);
    cudaDeviceSynchronize();
}

__global__ void f2d_kernel(size_t sizeA, const cuComplex *const __restrict__ in, cuDoubleComplex *const __restrict__ out) {
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= sizeA) return;
    cuComplex intmp = in[idx];
    cuDoubleComplex outtmp;
    outtmp.x = static_cast<double>(intmp.x);
    outtmp.y = static_cast<double>(intmp.y);
    out[idx] = outtmp;
}

void f2d(size_t m,                   // rows of A
         size_t n,                   // columns of A
         const cuComplex *const in,  // input
         cuDoubleComplex *const out) // output
{
    constexpr size_t block_size = 256;
    const size_t grid_size      = (m * n + block_size - 1) / block_size;
    f2d_kernel<<<grid_size, block_size>>>(m * n, in, out);
    cudaDeviceSynchronize();
}

} // namespace makemat
