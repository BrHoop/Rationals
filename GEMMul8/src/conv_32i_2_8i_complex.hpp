#pragma once
#include "common.hpp"
#include "template_math.hpp"

namespace oz2 {
namespace complex {

//------------------------------
// C8i = mod(C32i, 256)
//------------------------------
__device__ __forceinline__ double2 conv_32i_2_8i_256_scal(int32_t a, int32_t b, int32_t c) {
    double2 d;
    d.x = __int2double_rn((a - b) & 255);
    d.y = __int2double_rn((c - a - b) & 255);
    return d;
}

__global__ void conv_32i_2_8i_256_kernel(
    const size_t sizeC,                       // padding(m*n)/4
    const int32_t *const __restrict__ C32i_1, // input
    const int32_t *const __restrict__ C32i_2, // input
    const int32_t *const __restrict__ C32i_3, // input
    int2 *const __restrict__ C8i              // output
) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;

    int4 in1 = reinterpret_cast<const int4 *>(C32i_1)[idx];
    int4 in2 = reinterpret_cast<const int4 *>(C32i_2)[idx];
    int4 in3 = reinterpret_cast<const int4 *>(C32i_3)[idx];

    // Im(C8i)
    in3.x = in3.x - in1.x - in2.x;
    in3.y = in3.y - in1.y - in2.y;
    in3.z = in3.z - in1.z - in2.z;
    in3.w = in3.w - in1.w - in2.w;

    // Re(C8i)
    in1.x -= in2.x;
    in1.y -= in2.y;
    in1.z -= in2.z;
    in1.w -= in2.w;

    char4 part1{static_cast<int8_t>(in1.x), static_cast<int8_t>(in3.x),
                static_cast<int8_t>(in1.y), static_cast<int8_t>(in3.y)};

    char4 part2{static_cast<int8_t>(in1.z), static_cast<int8_t>(in3.z),
                static_cast<int8_t>(in1.w), static_cast<int8_t>(in3.w)};

    int2 out{*reinterpret_cast<int32_t *>(&part1),
             *reinterpret_cast<int32_t *>(&part2)};

    C8i[idx] = out;
}

//------------------------------
// C8i = mod(C32i, p)
//------------------------------
__device__ __forceinline__ double2 conv_32i_2_8i_not256_scal(int32_t a, int32_t b, int32_t c, unsigned table_idx) {
    int2 p_invp = table::MODULI_I[table_idx];
    a -= __mulhi(a, p_invp.y) * p_invp.x;
    b -= __mulhi(b, p_invp.y) * p_invp.x;
    c -= __mulhi(c, p_invp.y) * p_invp.x;

    c = c - a - b;
    c -= __mulhi(c, p_invp.y) * p_invp.x;
    c -= (c > 127) * p_invp.x;
    c += (c < -128) * p_invp.x;

    a -= b;
    a -= __mulhi(a, p_invp.y) * p_invp.x;
    a -= (a > 127) * p_invp.x;
    a += (a < -128) * p_invp.x;

    double2 d;
    d.x = __int2double_rn(a);
    d.y = __int2double_rn(c);
    return d;
}

__global__ void conv_32i_2_8i_not256_kernel(
    const size_t sizeC,                       // padding(m*n)/4
    const int32_t *const __restrict__ C32i_1, // input
    const int32_t *const __restrict__ C32i_2, // input
    const int32_t *const __restrict__ C32i_3, // input
    int2 *const __restrict__ C8i,             // output
    const unsigned table_idx                  //
) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;

    int2 p_invp = table::MODULI_I[table_idx]; // {p, invp}
    int4 in1    = reinterpret_cast<const int4 *>(C32i_1)[idx];
    int4 in2    = reinterpret_cast<const int4 *>(C32i_2)[idx];
    int4 in3    = reinterpret_cast<const int4 *>(C32i_3)[idx];

    in1.x -= __mulhi(in1.x, p_invp.y) * p_invp.x;
    in1.y -= __mulhi(in1.y, p_invp.y) * p_invp.x;
    in1.z -= __mulhi(in1.z, p_invp.y) * p_invp.x;
    in1.w -= __mulhi(in1.w, p_invp.y) * p_invp.x;

    in2.x -= __mulhi(in2.x, p_invp.y) * p_invp.x;
    in2.y -= __mulhi(in2.y, p_invp.y) * p_invp.x;
    in2.z -= __mulhi(in2.z, p_invp.y) * p_invp.x;
    in2.w -= __mulhi(in2.w, p_invp.y) * p_invp.x;

    in3.x -= __mulhi(in3.x, p_invp.y) * p_invp.x;
    in3.y -= __mulhi(in3.y, p_invp.y) * p_invp.x;
    in3.z -= __mulhi(in3.z, p_invp.y) * p_invp.x;
    in3.w -= __mulhi(in3.w, p_invp.y) * p_invp.x;

    // Im(C8i)
    in3.x = in3.x - in1.x - in2.x;
    in3.y = in3.y - in1.y - in2.y;
    in3.z = in3.z - in1.z - in2.z;
    in3.w = in3.w - in1.w - in2.w;

    in3.x -= __mulhi(in3.x, p_invp.y) * p_invp.x;
    in3.y -= __mulhi(in3.y, p_invp.y) * p_invp.x;
    in3.z -= __mulhi(in3.z, p_invp.y) * p_invp.x;
    in3.w -= __mulhi(in3.w, p_invp.y) * p_invp.x;

    in3.x -= (in3.x > 127) * p_invp.x;
    in3.y -= (in3.y > 127) * p_invp.x;
    in3.z -= (in3.z > 127) * p_invp.x;
    in3.w -= (in3.w > 127) * p_invp.x;

    in3.x += (in3.x < -128) * p_invp.x;
    in3.y += (in3.y < -128) * p_invp.x;
    in3.z += (in3.z < -128) * p_invp.x;
    in3.w += (in3.w < -128) * p_invp.x;

    // Re(C8i)
    in1.x -= in2.x;
    in1.y -= in2.y;
    in1.z -= in2.z;
    in1.w -= in2.w;

    in1.x -= __mulhi(in1.x, p_invp.y) * p_invp.x;
    in1.y -= __mulhi(in1.y, p_invp.y) * p_invp.x;
    in1.z -= __mulhi(in1.z, p_invp.y) * p_invp.x;
    in1.w -= __mulhi(in1.w, p_invp.y) * p_invp.x;

    in1.x -= (in1.x > 127) * p_invp.x;
    in1.y -= (in1.y > 127) * p_invp.x;
    in1.z -= (in1.z > 127) * p_invp.x;
    in1.w -= (in1.w > 127) * p_invp.x;

    in1.x += (in1.x < -128) * p_invp.x;
    in1.y += (in1.y < -128) * p_invp.x;
    in1.z += (in1.z < -128) * p_invp.x;
    in1.w += (in1.w < -128) * p_invp.x;

    char4 part1{static_cast<int8_t>(in1.x), static_cast<int8_t>(in3.x),
                static_cast<int8_t>(in1.y), static_cast<int8_t>(in3.y)};

    char4 part2{static_cast<int8_t>(in1.z), static_cast<int8_t>(in3.z),
                static_cast<int8_t>(in1.w), static_cast<int8_t>(in3.w)};

    int2 out{*reinterpret_cast<int32_t *>(&part1),
             *reinterpret_cast<int32_t *>(&part2)};

    C8i[idx] = out;
}

//------------------------------
// Interface!!
//------------------------------
__inline__ void conv_32i_2_8i(
    const unsigned i,           //
    const size_t sizeC,         // padding(m*n) / 4
    int32_t *const *const C32i, // input
    char2 *const C8i            // output
) {
    if (i == 0) {
        conv_32i_2_8i_256_kernel<<<grid_conv32i8i, threads_conv32i8i>>>(sizeC, C32i[0], C32i[1], C32i[2], reinterpret_cast<int2 *>(C8i));
    } else {
        conv_32i_2_8i_not256_kernel<<<grid_conv32i8i, threads_conv32i8i>>>(sizeC, C32i[0], C32i[1], C32i[2], reinterpret_cast<int2 *>(C8i), i - 1);
    }
}

} // namespace complex
} // namespace oz2
