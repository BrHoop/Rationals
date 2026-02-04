#pragma once
#if defined(__NVCC__)
    #include <cuComplex.h>
    #include <cublas_v2.h>
    #include <cuda_runtime.h>
#endif
#include "common.hpp"

namespace oz2 {

//------------------------------
// abs(x)
//------------------------------
template <typename T> __forceinline__ __device__ T Tabs(T in);
template <> __forceinline__ __device__ double Tabs<double>(double in) { return fabs(in); }
template <> __forceinline__ __device__ float Tabs<float>(float in) { return fabsf(in); }
template <> __forceinline__ __device__ int32_t Tabs<int32_t>(int32_t in) { return abs(in); }
template <> __forceinline__ __device__ cuDoubleComplex Tabs<cuDoubleComplex>(cuDoubleComplex in) { return make_cuDoubleComplex(fabs(in.x), fabs(in.y)); }
template <> __forceinline__ __device__ cuFloatComplex Tabs<cuFloatComplex>(cuFloatComplex in) { return make_cuFloatComplex(fabsf(in.x), fabsf(in.y)); }

//------------------------------
// x+y
//------------------------------
template <typename T> __forceinline__ __device__ T Tadd(T a, T b) { return a + b; }
template <> __forceinline__ __device__ cuDoubleComplex Tadd<cuDoubleComplex>(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }
template <> __forceinline__ __device__ cuFloatComplex Tadd<cuFloatComplex>(cuFloatComplex a, cuFloatComplex b) { return cuCaddf(a, b); }

//------------------------------
// x-y
//------------------------------
template <typename T> __forceinline__ __device__ T Tsub(T a, T b) { return a - b; }
template <> __forceinline__ __device__ cuDoubleComplex Tsub<cuDoubleComplex>(cuDoubleComplex a, cuDoubleComplex b) { return cuCsub(a, b); }
template <> __forceinline__ __device__ cuFloatComplex Tsub<cuFloatComplex>(cuFloatComplex a, cuFloatComplex b) { return cuCsubf(a, b); }

//------------------------------
// x*y
//------------------------------
template <typename T1, typename T2 = T1> __forceinline__ __device__ T2 Tmul(T1 a, T2 b) { return a * b; }
template <> __forceinline__ __device__ cuDoubleComplex Tmul<cuDoubleComplex, cuDoubleComplex>(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }
template <> __forceinline__ __device__ cuFloatComplex Tmul<cuFloatComplex, cuFloatComplex>(cuFloatComplex a, cuFloatComplex b) { return cuCmulf(a, b); }
template <> __forceinline__ __device__ cuDoubleComplex Tmul<double, cuDoubleComplex>(double a, cuDoubleComplex b) { return make_cuDoubleComplex(a * b.x, a * b.y); }

//------------------------------
// -x
//------------------------------
template <typename T> __forceinline__ __device__ T Tneg(T a) { return -a; }
template <> __forceinline__ __device__ cuDoubleComplex Tneg<cuDoubleComplex>(cuDoubleComplex a) { return make_cuDoubleComplex(-a.x, -a.y); }
template <> __forceinline__ __device__ cuFloatComplex Tneg<cuFloatComplex>(cuFloatComplex a) { return make_cuFloatComplex(-a.x, -a.y); }

//------------------------------
// x^2 + y
//------------------------------
template <typename T> __forceinline__ __device__ underlying_t<T> Tsqr_add_ru(T in1, underlying_t<T> in2);
template <> __forceinline__ __device__ double Tsqr_add_ru<double>(double in1, double in2) { return __fma_ru(in1, in1, in2); }
template <> __forceinline__ __device__ float Tsqr_add_ru<float>(float in1, float in2) { return __fmaf_ru(in1, in1, in2); }
template <> __forceinline__ __device__ double Tsqr_add_ru<cuDoubleComplex>(cuDoubleComplex in1, double in2) { return __fma_ru(in1.y, in1.y, __fma_ru(in1.x, in1.x, in2)); }
template <> __forceinline__ __device__ float Tsqr_add_ru<cuFloatComplex>(cuFloatComplex in1, float in2) { return __fmaf_ru(in1.y, in1.y, __fmaf_ru(in1.x, in1.x, in2)); }

//------------------------------
// x+y in round-up mode
//------------------------------
template <typename T> __forceinline__ __device__ T __Tadd_ru(T in1, T in2);
template <> __forceinline__ __device__ double __Tadd_ru<double>(double in1, double in2) { return __dadd_ru(in1, in2); }
template <> __forceinline__ __device__ float __Tadd_ru<float>(float in1, float in2) { return __fadd_ru(in1, in2); }

//------------------------------
// a*x + b*y
//------------------------------
template <typename T> __forceinline__ __device__ T Taxpby_scal(T a, T x, T b, T y);
template <> __forceinline__ __device__ double Taxpby_scal<double>(double a, double x, double b, double y) { return fma(b, y, a * x); }
template <> __forceinline__ __device__ float Taxpby_scal<float>(float a, float x, float b, float y) { return fmaf(b, y, a * x); }
template <> __device__ __forceinline__ cuDoubleComplex Taxpby_scal<cuDoubleComplex>(cuDoubleComplex a, cuDoubleComplex x, cuDoubleComplex b, cuDoubleComplex y) {
    double2 out;
    out.x = fma(-b.y, y.y, fma(b.x, y.x, fma(-a.y, x.y, a.x * x.x))); // a.x*x.x - a.y*x.y + b.x*y.x - b.y*y.y
    out.y = fma(b.y, y.x, fma(b.x, y.y, fma(a.y, x.x, a.x * x.y)));   // a.x*x.y + a.y*x.x + b.x*y.y + b.y*y.x
    return out;
}
template <> __device__ __forceinline__ cuFloatComplex Taxpby_scal<cuFloatComplex>(cuFloatComplex a, cuFloatComplex x, cuFloatComplex b, cuFloatComplex y) {
    float2 out;
    out.x = fmaf(-b.y, y.y, fmaf(b.x, y.x, fmaf(-a.y, x.y, a.x * x.x))); // a.x*x.x - a.y*x.y + b.x*y.x - b.y*y.y
    out.y = fmaf(b.y, y.x, fmaf(b.x, y.y, fmaf(a.y, x.x, a.x * x.y)));   // a.x*x.y + a.y*x.x + b.x*y.y + b.y*y.x
    return out;
}

//------------------------------
// x*2^y
//------------------------------
template <typename T> __forceinline__ __device__ T Tscalbn(T in, int sft);
template <> __forceinline__ __device__ double Tscalbn<double>(double in, int sft) { return scalbn(in, sft); }
template <> __forceinline__ __device__ float Tscalbn<float>(float in, int sft) { return scalbnf(in, sft); }
template <> __forceinline__ __device__ cuDoubleComplex Tscalbn<cuDoubleComplex>(cuDoubleComplex in, int sft) { return make_cuDoubleComplex(scalbn(in.x, sft), scalbn(in.y, sft)); }
template <> __forceinline__ __device__ cuFloatComplex Tscalbn<cuFloatComplex>(cuFloatComplex in, int sft) { return make_cuFloatComplex(scalbnf(in.x, sft), scalbnf(in.y, sft)); }

//------------------------------
// rint(x) = round(x)
//------------------------------
template <typename T> __forceinline__ __device__ T Trint(T in);
template <> __forceinline__ __device__ double Trint<double>(double in) { return rint(in); }
template <> __forceinline__ __device__ cuDoubleComplex Trint<cuDoubleComplex>(cuDoubleComplex in) { return make_cuDoubleComplex(rint(in.x), rint(in.y)); }

//------------------------------
// ilogb(x) = floor(log2(x))
//------------------------------
template <typename T> __forceinline__ __device__ int Tilogb(T in);
template <> __forceinline__ __device__ int Tilogb<double>(double in) { return (in == 0.0) ? 0 : ilogb(in); }
template <> __forceinline__ __device__ int Tilogb<float>(float in) { return (in == 0.0F) ? 0 : ilogbf(in); }
//------------------------------
// max(x,y)
//------------------------------
template <typename T> __forceinline__ __device__ underlying_t<T> Tmax(T in1, underlying_t<T> in2);
template <> __forceinline__ __device__ double Tmax<double>(double in1, double in2) { return max(in1, in2); }
template <> __forceinline__ __device__ float Tmax<float>(float in1, float in2) { return max(in1, in2); }
template <> __forceinline__ __device__ int32_t Tmax<int32_t>(int32_t in1, int32_t in2) { return max(in1, in2); }
template <> __forceinline__ __device__ double Tmax<cuDoubleComplex>(cuDoubleComplex in1, double in2) { return max(max(in1.x, in1.y), in2); }
template <> __forceinline__ __device__ float Tmax<cuFloatComplex>(cuFloatComplex in1, float in2) { return max(max(in1.x, in1.y), in2); }

//------------------------------
// Cast Tin to Tout
//------------------------------
template <typename Tin, typename Tout> __forceinline__ __device__ Tout Tcast(Tin in);
template <> __forceinline__ __device__ double Tcast<double, double>(double in) { return in; }
template <> __forceinline__ __device__ float Tcast<double, float>(double in) { return __double2float_rn(in); }
template <> __forceinline__ __device__ cuDoubleComplex Tcast<cuDoubleComplex, cuDoubleComplex>(cuDoubleComplex in) { return in; }
template <> __forceinline__ __device__ cuFloatComplex Tcast<cuDoubleComplex, cuFloatComplex>(cuDoubleComplex in) { return make_cuFloatComplex(__double2float_rn(in.x), __double2float_rn(in.y)); }
template <> __forceinline__ __device__ cuDoubleComplex Tcast<uchar2, cuDoubleComplex>(uchar2 in) { return make_cuDoubleComplex(static_cast<double>(in.x), static_cast<double>(in.y)); }
template <> __forceinline__ __device__ double Tcast<uint8_t, double>(uint8_t in) { return static_cast<double>(in); }

//------------------------------
// static_cast (fp -> int)
//------------------------------
template <typename T> __forceinline__ __device__ int_t<T> __fp2int_rz(T in);
template <> __forceinline__ __device__ int_t<double> __fp2int_rz<double>(double in) { return __double2ll_rz(in); }
template <> __forceinline__ __device__ int_t<float> __fp2int_rz<float>(float in) { return __float2int_rz(in); }

//------------------------------
// reinterpret_cast (fp -> int)
//------------------------------
template <typename T> __forceinline__ __device__ int_t<T> __fp_as_int(T in);
template <> __forceinline__ __device__ int_t<double> __fp_as_int<double>(double in) { return __double_as_longlong(in); }
template <> __forceinline__ __device__ int_t<float> __fp_as_int<float>(float in) { return __float_as_int(in); }

//------------------------------
// reinterpret_cast (fp <- int)
//------------------------------
template <typename T> __forceinline__ __device__ fp_t<T> __int_as_fp(int_t<T> in);
template <> __forceinline__ __device__ fp_t<double> __int_as_fp<double>(int_t<double> in) { return __longlong_as_double(in); }
template <> __forceinline__ __device__ fp_t<float> __int_as_fp<float>(int_t<float> in) { return __int_as_float(in); }

//------------------------------
// Extract sign part
//------------------------------
template <typename T> __forceinline__ __device__ int_t<T> extract_sign(int_t<T> in);
template <> __forceinline__ __device__ int_t<double> extract_sign<double>(int_t<double> in) { return in & 0x8000000000000000LL; }
template <> __forceinline__ __device__ int_t<float> extract_sign<float>(int_t<float> in) { return in & 0x80000000; }

//------------------------------
// Extract exponent part
//------------------------------
template <typename T> __forceinline__ __device__ int_t<T> extract_exp(int_t<T> in);
template <> __forceinline__ __device__ int_t<double> extract_exp<double>(int_t<double> in) { return (in >> 52) & 0x7FF; }
template <> __forceinline__ __device__ int_t<float> extract_exp<float>(int_t<float> in) { return (in >> 23) & 0xFF; }

//------------------------------
// Extract significand part
//------------------------------
template <typename T> __forceinline__ __device__ int_t<T> extract_significand(int_t<T> in);
template <> __forceinline__ __device__ int_t<double> extract_significand<double>(int_t<double> in) { return in & 0x000FFFFFFFFFFFFFULL; }
template <> __forceinline__ __device__ int_t<float> extract_significand<float>(int_t<float> in) { return in & 0x007FFFFF; }

//------------------------------
// Count the number of consecutive high-order zero bits
//------------------------------
template <typename T> __forceinline__ __device__ int32_t countlz(int_t<T> in);
template <> __forceinline__ __device__ int32_t countlz<double>(int_t<double> in) { return __clzll(in); }
template <> __forceinline__ __device__ int32_t countlz<float>(int_t<float> in) { return __clz(in); }

//------------------------------
// Retrieves the complex conjugate of a complex number.
//------------------------------
template <typename T, bool CONJ> __forceinline__ __device__ T conj(T in) { return in; };
template <> __forceinline__ __device__ cuDoubleComplex conj<cuDoubleComplex, true>(cuDoubleComplex in) { return make_cuDoubleComplex(in.x, -in.y); };
template <> __forceinline__ __device__ cuFloatComplex conj<cuFloatComplex, true>(cuFloatComplex in) { return make_cuFloatComplex(in.x, -in.y); };

//------------------------------
// Warp reduction (max)
//------------------------------
template <typename T, int width = 32> __forceinline__ __device__ T inner_warp_max(T amax) {
#pragma unroll
    for (unsigned i = width / 2; i > 0; i >>= 1) {
        amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, i, width));
    }
    return amax;
}
template <> __forceinline__ __device__ double inner_warp_max<double, 32>(double amax) {
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 16u, 32));
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 8u, 32));
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 4u, 32));
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 2u, 32));
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 1u, 32));
    return amax;
}
template <> __forceinline__ __device__ float inner_warp_max<float, 32>(float amax) {
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 16u, 32));
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 8u, 32));
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 4u, 32));
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 2u, 32));
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 1u, 32));
    return amax;
}
template <> __forceinline__ __device__ int inner_warp_max<int, 32>(int amax) {
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 16u, 32));
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 8u, 32));
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 4u, 32));
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 2u, 32));
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 1u, 32));
    return amax;
}

//------------------------------
// Warp reduction (sum in round-up mode)
//------------------------------
template <typename T, int width = 32> __forceinline__ __device__ T inner_warp_sum(T sum) {
#pragma unroll
    for (unsigned i = width / 2; i > 0; i >>= 1) {
        sum = __Tadd_ru<T>(sum, __shfl_down_sync(0xFFFFFFFFu, sum, i, width));
    }
    return sum;
}
template <> __forceinline__ __device__ double inner_warp_sum<double, 32>(double sum) {
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 16u, 32));
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 8u, 32));
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 4u, 32));
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 2u, 32));
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 1u, 32));
    return sum;
}
template <> __forceinline__ __device__ float inner_warp_sum<float, 32>(float sum) {
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 16u, 32));
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 8u, 32));
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 4u, 32));
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 2u, 32));
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 1u, 32));
    return sum;
}

//------------------------------
// Calculate mod: a - round(a/p(j))*p(j)
//------------------------------
template <typename T, int ITER> __forceinline__ __device__ int8_t mod_8i(T a, table::tab_t<T> val);
template <> __forceinline__ __device__ int8_t mod_8i<double, 1>(double a, table::tab_t<double> val) {
    double tmp_d = fma(rint(a * val.f64.y), val.f64.x, a);
    return static_cast<int8_t>(tmp_d);
}
template <> __forceinline__ __device__ int8_t mod_8i<double, 2>(double a, table::tab_t<double> val) {
    float tmp1 = __double2float_rn(fma(rint(a * val.f64.y), val.f64.x, a));
    float tmp2 = fmaf(rintf(tmp1 * val.f32.y), val.f32.x, tmp1);
    return static_cast<int8_t>(tmp2);
}
template <> __forceinline__ __device__ int8_t mod_8i<double, 3>(double a, table::tab_t<double> val) {
    float tmp1 = __double2float_rn(fma(rint(a * val.f64.y), val.f64.x, a));
    float tmp2 = fmaf(rintf(tmp1 * val.f32.y), val.f32.x, tmp1);
    tmp1       = fmaf(rintf(tmp2 * val.f32.y), val.f32.x, tmp2);
    return static_cast<int8_t>(tmp1);
}
template <> __forceinline__ __device__ int8_t mod_8i<double, 4>(double a, table::tab_t<double> val) {
    float tmp1 = __double2float_rn(fma(rint(a * val.f64.y), val.f64.x, a));
    float tmp2 = fmaf(rintf(tmp1 * val.f32.y), val.f32.x, tmp1);
    tmp1       = fmaf(rintf(tmp2 * val.f32.y), val.f32.x, tmp2);
    tmp2       = fmaf(rintf(tmp1 * val.f32.y), val.f32.x, tmp1);
    return static_cast<int8_t>(tmp2);
}
template <> __forceinline__ __device__ int8_t mod_8i<float, 1>(float a, table::tab_t<float> val) {
    float tmp1 = fmaf(rintf(a * val.y), val.x, a);
    return static_cast<int8_t>(tmp1);
}
template <> __forceinline__ __device__ int8_t mod_8i<float, 2>(float a, table::tab_t<float> val) {
    float tmp1 = fmaf(rintf(a * val.y), val.x, a);
    float tmp2 = fmaf(rintf(tmp1 * val.y), val.x, tmp1);
    return static_cast<int8_t>(tmp2);
}
template <> __forceinline__ __device__ int8_t mod_8i<float, 3>(float a, table::tab_t<float> val) {
    float tmp1 = fmaf(rintf(a * val.y), val.x, a);
    float tmp2 = fmaf(rintf(tmp1 * val.y), val.x, tmp1);
    tmp1       = fmaf(rintf(tmp2 * val.y), val.x, tmp2);
    return static_cast<int8_t>(tmp1);
}
template <> __forceinline__ __device__ int8_t mod_8i<float, 4>(float a, table::tab_t<float> val) {
    float tmp1 = fmaf(rintf(a * val.y), val.x, a);
    float tmp2 = fmaf(rintf(tmp1 * val.y), val.x, tmp1);
    tmp1       = fmaf(rintf(tmp2 * val.y), val.x, tmp2);
    tmp2       = fmaf(rintf(tmp1 * val.y), val.x, tmp1);
    return static_cast<int8_t>(tmp2);
}

template <typename T, int ITER> __forceinline__ __device__ char3 mod_8i_complex(T a, table::tab_t<underlying_t<T>> val, int p) {
    int tmp1 = mod_8i<underlying_t<T>, ITER>(a.x, val);
    int tmp2 = mod_8i<underlying_t<T>, ITER>(a.y, val);
    int tmp3 = tmp1 + tmp2;
    tmp3 -= (tmp3 > 127) * p;
    tmp3 += (tmp3 < -128) * p;
    char3 out{static_cast<int8_t>(tmp1), static_cast<int8_t>(tmp2), static_cast<int8_t>(tmp3)};
    return out;
};

template <typename T, int ITER> __forceinline__ __device__ int8_t mod_8i_256(T a) {
    if constexpr (ITER == 1) {
        // |a| < 2^31 holds!
        auto i = __fp2int_rz<T>(a);
        return static_cast<int8_t>(i);
    } else {
        // if |a| >= fp<T>::th, LSB 8-bit is 0000'0000!
        T abs_a       = Tabs<T>(a);
        int_t<T> i    = __fp2int_rz<T>(a);
        int_t<T> mask = -static_cast<int_t<T>>(abs_a < fp<T>::th);
        return static_cast<int8_t>(i & mask);
    }
}

template <typename T, int ITER> __forceinline__ __device__ char3 mod_8i_256_complex(T a) {
    int tmp1 = mod_8i_256<underlying_t<T>, ITER>(a.x);
    int tmp2 = mod_8i_256<underlying_t<T>, ITER>(a.y);
    int tmp3 = tmp1 + tmp2;
    char3 out{static_cast<int8_t>(tmp1), static_cast<int8_t>(tmp2), static_cast<int8_t>(tmp3)};
    return out;
}

//------------------------------
// Return trunc(scalbn(in, sft))
//------------------------------
template <typename T> __forceinline__ __device__ T T2int_fp(T in, const int sft) {
    int_t<T> bits        = __fp_as_int<T>(in);
    const int_t<T> sign  = extract_sign<T>(bits);
    int exp_biased       = (int)extract_exp<T>(bits);
    int_t<T> significand = extract_significand<T>(bits);

    if (exp_biased != 0) {
        exp_biased += sft;
        if (exp_biased < fp<T>::bias) {
            return __int_as_fp<T>(sign);
        }
        if (exp_biased >= (fp<T>::bias + fp<T>::prec)) {
            bits = sign | ((int_t<T>)exp_biased << fp<T>::prec) | significand;
            return __int_as_fp<T>(bits);
        }

        significand |= ((int_t<T>)1 << fp<T>::prec);
        int chop_bits = (fp<T>::bias + fp<T>::prec) - exp_biased;
        int_t<T> mask = (int_t<T>)(-1) << chop_bits;
        significand   = extract_significand<T>(significand & mask);
        bits          = sign | (int_t<T>)exp_biased << fp<T>::prec | significand;
        return __int_as_fp<T>(bits);
    }

    if (significand == 0) {
        return in;
    }

    int lz = (fp<T>::bits - fp<T>::prec) - countlz<T>(significand);
    int e  = lz + sft;

    if (e < fp<T>::bias) {
        return __int_as_fp<T>(sign);
    }

    int_t<T> frac_full = (significand << (2 - lz)) ^ (Tconst<int_t<T>>::one() << fp<T>::prec);
    int_t<T> mask      = Tconst<int_t<T>>::mone() << max((int_t<T>)(fp<T>::bias + fp<T>::prec - e), Tconst<int_t<T>>::zero());
    bits               = sign | (int_t<T>)e << fp<T>::prec | (frac_full & mask);
    return __int_as_fp<T>(bits);
}
template <> __forceinline__ __device__ cuDoubleComplex T2int_fp<cuDoubleComplex>(cuDoubleComplex in, const int sft) {
    cuDoubleComplex out;
    out.x = T2int_fp<double>(in.x, sft);
    out.y = T2int_fp<double>(in.y, sft);
    return out;
}
template <> __forceinline__ __device__ cuFloatComplex T2int_fp<cuFloatComplex>(cuFloatComplex in, const int sft) {
    cuFloatComplex out;
    out.x = T2int_fp<float>(in.x, sft);
    out.y = T2int_fp<float>(in.y, sft);
    return out;
}

//------------------------------
// Return int8_t(ceil(scalbn(fabs(in),sft)))
//------------------------------
template <typename T> __forceinline__ __device__ int8x2_t<T> T2int_8i(T in, const int sft) {
    using I = int_t<T>;

    // fabs (sign clear)
    I bits = __fp_as_int<T>(in);
    bits &= ~extract_sign<T>(bits);

    // zero
    if (bits == 0) {
        return int8_t(0);
    }

    // exp / frac
    int exp = (int)extract_exp<T>(bits);
    I frac  = extract_significand<T>(bits);

    // mantissa + unbiased exponent
    I mant;
    int e;

    if (exp) {
        // normal
        mant = frac | (I(1) << fp<T>::prec);
        e    = exp - fp<T>::bias;
    } else {
        // subnormal (only place using clz)
        int k = countlz<T>(frac) - (fp<T>::bits - fp<T>::prec);
        mant  = (frac << k) | (I(1) << fp<T>::prec);
        e     = (1 - fp<T>::bias) - k;
    }

    // apply scalbn
    e += sft;

    // integer exponent alignment
    int shift = fp<T>::prec - e;

    // integer result (no rounding needed)
    if (shift <= 0) {
        return static_cast<int8_t>(mant << (-shift));
    }

    // small values
    if (shift >= fp<T>::prec + 1) {
        return int8_t(1);
    }

    // compute ceil
    I mask     = (I(1) << shift) - 1;
    I floor    = mant >> shift;
    I has_frac = (mant & mask) != 0;

    return static_cast<int8_t>(floor + has_frac);
};
template <> __forceinline__ __device__ char2 T2int_8i<cuDoubleComplex>(cuDoubleComplex in, const int sft) {
    char2 out;
    out.x = T2int_8i<double>(in.x, sft);
    out.y = T2int_8i<double>(in.y, sft);
    return out;
}
template <> __forceinline__ __device__ char2 T2int_8i<cuFloatComplex>(cuFloatComplex in, const int sft) {
    char2 out;
    out.x = T2int_8i<float>(in.x, sft);
    out.y = T2int_8i<float>(in.y, sft);
    return out;
}

//------------------------------
// Column-wise amax
//------------------------------
template <typename T> __forceinline__ __device__ underlying_t<T> find_amax(
    const T *const ptr,    //
    const unsigned length, //
    underlying_t<T> *samax // shared memory (workspace)
) {
    // max in thread
    underlying_t<T> amax = Tconst<underlying_t<T>>::zero();
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
        const T tmp = Tabs<T>(ptr[i]);
        amax        = Tmax<T>(tmp, amax);
    }

    // inner-warp reduction
    amax = inner_warp_max(amax);

    // inner-threadblock reduction
    if ((threadIdx.x & 0x1f) == 0) samax[threadIdx.x >> 5] = amax; // samax[warp-id] = max in warp

    __syncthreads();
    if (threadIdx.x < 32) {
        if (threadIdx.x < (blockDim.x >> 5)) amax = samax[threadIdx.x];
        amax = inner_warp_max(amax);
        if (threadIdx.x == 0) samax[0] = amax;
    }

    __syncthreads();
    return samax[0];
}

//------------------------------
// Column-wise max
//------------------------------
__forceinline__ __device__ int find_max(
    const int *const ptr,  //
    const unsigned length, //
    int *samax             // shared memory (workspace)
) {
    // max in thread
    int amax = 0;
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
        const int tmp = ptr[i];
        amax          = max(tmp, amax);
    }

    // inner-warp reduction
    amax = inner_warp_max(amax);

    // inner-threadblock reduction
    if ((threadIdx.x & 0x1f) == 0) samax[threadIdx.x >> 5] = amax; // samax[warp-id] = max in warp

    __syncthreads();
    if (threadIdx.x < 32) {
        if (threadIdx.x < (blockDim.x >> 5)) amax = samax[threadIdx.x];
        amax = inner_warp_max(amax);
        if (threadIdx.x == 0) samax[0] = amax;
    }

    __syncthreads();
    return samax[0];
}

__forceinline__ __device__ int find_max_complex(
    const int *const ptr_1, //
    const int *const ptr_2, //
    const unsigned length,  //
    int *samax              // shared memory (workspace)
) {
    // max in thread
    int amax = 0;
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
        const int tmp1 = ptr_1[i];
        const int tmp2 = ptr_2[i];
        amax           = max(max(tmp1 + tmp2, tmp2), amax);
    }

    // inner-warp reduction
    amax = inner_warp_max(amax);

    // inner-threadblock reduction
    if ((threadIdx.x & 0x1f) == 0) {
        samax[threadIdx.x >> 5] = amax; // samax[warp-id] = max in warp
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        if (threadIdx.x < (blockDim.x >> 5)) {
            amax = samax[threadIdx.x];
        }
        amax = inner_warp_max(amax);
        if (threadIdx.x == 0) {
            samax[0] = amax;
        }
    }

    __syncthreads();
    return samax[0];
}

//------------------------------
// Column-wise amax & sum of squares
//------------------------------
template <typename T> __forceinline__ __device__ underlying_t<T> find_amax_and_nrm(
    const T *const ptr,     //
    const unsigned length,  //
    underlying_t<T> *shm,   // shared memory (workspace)
    underlying_t<T> &vecnrm // 2-norm^2
) {
    using U = underlying_t<T>;

    U *samax = shm;
    U *ssum  = shm + 32;

    // max in thread
    U amax = Tconst<U>::zero();
    U sum  = Tconst<U>::zero();
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
        T tmp = Tabs<T>(ptr[i]);
        amax  = Tmax<T>(tmp, amax);
        sum   = Tsqr_add_ru<T>(tmp, sum); // round-up mode
    }

    // inner-warp reduction
    amax = inner_warp_max(amax);
    sum  = inner_warp_sum(sum);

    // inner-threadblock reduction
    if ((threadIdx.x & 31) == 0) {
        samax[threadIdx.x >> 5] = amax; // samax[warp-id] = max in warp
        ssum[threadIdx.x >> 5]  = sum;  // ssum[warp-id] = sum in warp
    }

    __syncthreads();
    sum = Tconst<U>::zero();
    if (threadIdx.x < 32) {
        if (threadIdx.x < (blockDim.x >> 5)) {
            amax = samax[threadIdx.x];
            sum  = ssum[threadIdx.x];
        }
        amax = inner_warp_max(amax);
        sum  = inner_warp_sum(sum);
        if (threadIdx.x == 0) {
            samax[0] = amax;
            ssum[0]  = sum;
        }
    }

    __syncthreads();
    vecnrm = ssum[0];
    return samax[0];
}

//------------------------------
// Row-wise amax
//------------------------------
template <typename T> __forceinline__ __device__ underlying_t<T> find_amax_tile(
    const size_t m, const size_t k,       // size(A)
    const unsigned row_idx,               //
    const T *const A,                     // input (lda * k)
    const size_t lda,                     // leading dimension
    underlying_t<T> samax[][TILE_DIM + 1] // shared memory (workspace)
) {
    using U = underlying_t<T>;

    U amax = Tconst<U>::zero();
    if (row_idx < m) {
        const T *row_ptr = A + row_idx;
        for (unsigned col = threadIdx.y; col < k; col += blockDim.y) {
            const T tmp = Tabs<T>(row_ptr[col * lda]);
            amax        = Tmax<T>(tmp, amax);
        }
    }
    samax[threadIdx.y][threadIdx.x] = amax;
    __syncthreads();

    amax = samax[threadIdx.x][threadIdx.y];
    amax = inner_warp_max<U, TILE_DIM>(amax);
    return amax;
}

//------------------------------
// Row-wise max
//------------------------------
__forceinline__ __device__ int find_max_tile(
    const size_t m, const size_t k, // size(A)
    const unsigned row_idx,         //
    const int *const A,             // input (lda * k)
    const size_t lda,               // leading dimension
    int samax[][TILE_DIM + 1]       // shared memory (workspace)
) {
    int amax = 0;
    if (row_idx < m) {
        const int *row_ptr = A + row_idx;
        for (unsigned col = threadIdx.y; col < k; col += blockDim.y) {
            const int tmp = row_ptr[col * lda];
            amax          = max(tmp, amax);
        }
    }
    samax[threadIdx.y][threadIdx.x] = amax;
    __syncthreads();

    amax = samax[threadIdx.x][threadIdx.y];
    amax = inner_warp_max<int, TILE_DIM>(amax);
    return amax;
}

__forceinline__ __device__ int find_max_tile_complex(
    const size_t m, const size_t k, // size(A)
    const unsigned row_idx,         //
    const int *const A1,            // input (lda * k)
    const int *const A2,            // input (lda * k)
    const size_t lda,               // leading dimension
    int samax[][TILE_DIM + 1]       // shared memory (workspace)
) {
    int amax = 0;
    if (row_idx < m) {
        const int *row_ptr1 = A1 + row_idx;
        const int *row_ptr2 = A2 + row_idx;
        for (unsigned col = threadIdx.y; col < k; col += blockDim.y) {
            const int tmp1 = row_ptr1[col * lda];
            const int tmp2 = row_ptr2[col * lda];
            amax           = max(max(tmp1 + tmp2, tmp2), amax);
        }
    }
    samax[threadIdx.y][threadIdx.x] = amax;
    __syncthreads();

    amax = samax[threadIdx.x][threadIdx.y];
    amax = inner_warp_max<int, TILE_DIM>(amax);

    return amax;
}

//------------------------------
// Row-wise amax & sum of squares
//------------------------------
template <typename T> __forceinline__ __device__ underlying_t<T> find_amax_and_nrm_tile(
    const size_t m, const size_t k,        // size(A)
    const unsigned row_idx,                //
    const T *const A,                      // input (lda * k)
    const size_t lda,                      // leading dimension
    underlying_t<T> samax[][TILE_DIM + 1], // shared memory (workspace)
    underlying_t<T> ssum[][TILE_DIM + 1],  // shared memory (workspace)
    underlying_t<T> &vecnrm                // 2-norm^2
) {
    using U = underlying_t<T>;

    U amax = Tconst<U>::zero();
    U sum  = Tconst<U>::zero();
    if (row_idx < m) {
        const T *row_ptr = A + row_idx;
        for (unsigned col = threadIdx.y; col < k; col += blockDim.y) {
            const T tmp = Tabs<T>(row_ptr[col * lda]);
            amax        = Tmax<T>(tmp, amax);
            sum         = Tsqr_add_ru<T>(tmp, sum); // round-up mode
        }
    }
    samax[threadIdx.y][threadIdx.x] = amax;
    ssum[threadIdx.y][threadIdx.x]  = sum;
    __syncthreads();

    sum    = ssum[threadIdx.x][threadIdx.y];
    vecnrm = inner_warp_sum<U, TILE_DIM>(sum);

    amax = samax[threadIdx.x][threadIdx.y];
    amax = inner_warp_max<U, TILE_DIM>(amax);

    return amax;
}

} // namespace oz2
