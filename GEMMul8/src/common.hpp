#pragma once
#if defined(__NVCC__)
    #include <cuComplex.h>
    #include <cublas_v2.h>
    #include <cuda_runtime.h>
#endif
#include "self_hipify.hpp"
#include "table.hpp"
#include "template_type.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#ifndef _WIN32
    #include <dlfcn.h>
#endif
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace oz2 {

//------------------------------
// CUDA grid and thread configuration
//------------------------------
size_t grid_invscal;
size_t grid_conv32i8i;
inline constexpr size_t threads_scaling   = 256;
inline constexpr size_t threads_conv32i8i = 256;
inline constexpr size_t threads_invscal   = 128;
inline constexpr int TILE_DIM             = 32; // better than 16 for A100, GH200
inline constexpr int PAD_SIZE             = 32;

//------------------------------
// Iteration threshold for modular reduction
// Used to decide ITER count based on num_moduli
//------------------------------
template <typename T> struct threshold;
template <> struct threshold<double> {
    static constexpr unsigned iter1 = 12u;
    static constexpr unsigned iter2 = 18u;
    static constexpr unsigned iter3 = 25u;
};
template <> struct threshold<float> {
    static constexpr unsigned iter1 = 5u;
    static constexpr unsigned iter2 = 11u;
    static constexpr unsigned iter3 = 18u;
};

//------------------------------
// Pad size to multiple of 16 (for alignment)
//------------------------------
__forceinline__ __host__ __device__ size_t padding(const size_t n) { return PAD_SIZE * ((n + (PAD_SIZE - 1)) / PAD_SIZE); }

//------------------------------
// Start timing measurement
//------------------------------
void timing(std::chrono::system_clock::time_point &time_stamp) {
    cudaDeviceSynchronize();
    time_stamp = std::chrono::system_clock::now();
}

//------------------------------
// Stop timing and accumulate elapsed time (ns)
//------------------------------
void timing(std::chrono::system_clock::time_point &time_stamp, double &timer) {
    cudaDeviceSynchronize();
    std::chrono::system_clock::time_point time_now = std::chrono::system_clock::now();
    timer += std::chrono::duration_cast<std::chrono::nanoseconds>(time_now - time_stamp).count();
    time_stamp = time_now;
}

} // namespace oz2
