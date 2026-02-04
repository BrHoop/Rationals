#pragma once
#include "common.hpp"
#include "template_math.hpp"

namespace oz2 {
namespace real {

//------------------------------
// C8i = mod(C32i, 256)
//------------------------------
__device__ __forceinline__ double conv_32i_2_8i_256_scal(int32_t a) {
    return __int2double_rn(a & 255);
}

__global__ void conv_32i_2_8i_256_kernel(
    const size_t sizeC,                     // padding(m*n)/4
    const int32_t *const __restrict__ C32i, // input
    char4 *const __restrict__ C8i           // output
) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;

    int4 in = reinterpret_cast<const int4 *>(C32i)[idx];

    char4 out{static_cast<int8_t>(in.x), static_cast<int8_t>(in.y),
              static_cast<int8_t>(in.z), static_cast<int8_t>(in.w)};

    C8i[idx] = out;
}

//------------------------------
// C8i = mod(C32i, p)
//------------------------------
__device__ __forceinline__ double conv_32i_2_8i_not256_scal(int32_t a, unsigned table_idx) {
    int2 p_invp = table::MODULI_I[table_idx];
    int32_t b   = a - __mulhi(a, p_invp.y) * p_invp.x;
    b -= (b > 127) * p_invp.x;
    b += (b < -128) * p_invp.x;
    return __int2double_rn(b);
}

__global__ void conv_32i_2_8i_not256_kernel(
    const size_t sizeC,                     // padding(m*n)/4
    const int32_t *const __restrict__ C32i, // input
    char4 *const __restrict__ C8i,          // output
    const unsigned table_idx                //
) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;

    int2 p_invp = table::MODULI_I[table_idx]; // {p, invp}
    int4 in     = reinterpret_cast<const int4 *>(C32i)[idx];

    in.x -= __mulhi(in.x, p_invp.y) * p_invp.x;
    in.y -= __mulhi(in.y, p_invp.y) * p_invp.x;
    in.z -= __mulhi(in.z, p_invp.y) * p_invp.x;
    in.w -= __mulhi(in.w, p_invp.y) * p_invp.x;

    in.x -= (in.x > 127) * p_invp.x;
    in.y -= (in.y > 127) * p_invp.x;
    in.z -= (in.z > 127) * p_invp.x;
    in.w -= (in.w > 127) * p_invp.x;

    in.x += (in.x < -128) * p_invp.x;
    in.y += (in.y < -128) * p_invp.x;
    in.z += (in.z < -128) * p_invp.x;
    in.w += (in.w < -128) * p_invp.x;

    char4 out{static_cast<int8_t>(in.x), static_cast<int8_t>(in.y),
              static_cast<int8_t>(in.z), static_cast<int8_t>(in.w)};

    C8i[idx] = out;
}

//------------------------------
// Interface!!
//------------------------------
__inline__ void conv_32i_2_8i(
    const unsigned i,          //
    const size_t sizeC,        // padding(m*n) / 4
    const int32_t *const C32i, // input
    int8_t *const C8i          // output
) {
    if (i == 0) {
        conv_32i_2_8i_256_kernel<<<grid_conv32i8i, threads_conv32i8i>>>(sizeC, C32i, reinterpret_cast<char4 *>(C8i));
    } else {
        conv_32i_2_8i_not256_kernel<<<grid_conv32i8i, threads_conv32i8i>>>(sizeC, C32i, reinterpret_cast<char4 *>(C8i), i - 1);
    }
}

} // namespace real
} // namespace oz2
