#pragma once
#if defined(__NVCC__)
    #include <cuComplex.h>
    #include <cublas_v2.h>
    #include <cuda_runtime.h>
#endif
#include "self_hipify.hpp"

namespace oz2 {

//==========
// Underlying table type
//==========
namespace table {
template <typename T> struct tab_t_mult;
template <> struct tab_t_mult<double> {
    double2 f64;
    float2 f32;
};
template <> struct tab_t_mult<cuDoubleComplex> {
    double2 f64;
    float2 f32;
    int modulus;
};
template <> struct tab_t_mult<cuFloatComplex> {
    float2 f32;
    int modulus;
};
template <typename T> struct tab_type {
    using type = tab_t_mult<T>;
};
template <> struct tab_type<float> {
    using type = float2;
};
template <typename T> using tab_t = typename tab_type<T>::type;
} // namespace table

//==========
// Map type to same size integer type
//==========
template <typename T>
struct samesize_int;
template <> struct samesize_int<int32_t> {
    using type = int32_t;
};
template <> struct samesize_int<int64_t> {
    using type = int64_t;
};
template <> struct samesize_int<float> {
    using type = int32_t;
};
template <> struct samesize_int<double> {
    using type = int64_t;
};
template <typename T> using int_t = typename samesize_int<T>::type;

//==========
// Map type to same size floating-point type
//==========
template <typename T> struct samesize_fp;
template <> struct samesize_fp<int32_t> {
    using type = float;
};
template <> struct samesize_fp<int64_t> {
    using type = double;
};
template <> struct samesize_fp<float> {
    using type = float;
};
template <> struct samesize_fp<double> {
    using type = double;
};
template <typename T> using fp_t = typename samesize_fp<T>::type;

//==========
// Map type to underlying scalar type
//==========
template <typename T> struct underlying_type;
template <> struct underlying_type<int32_t> {
    using type = int32_t;
};
template <> struct underlying_type<float> {
    using type = float;
};
template <> struct underlying_type<double> {
    using type = double;
};
template <> struct underlying_type<cuFloatComplex> {
    using type = float;
};
template <> struct underlying_type<cuDoubleComplex> {
    using type = double;
};
template <typename T> using underlying_t = typename underlying_type<T>::type;

//==========
// Map type to single- or triple-int8 representation
//==========
template <typename T> struct int8x3_type;
template <> struct int8x3_type<float> {
    using type = int8_t;
};
template <> struct int8x3_type<double> {
    using type = int8_t;
};
template <> struct int8x3_type<cuFloatComplex> {
    using type = char3;
};
template <> struct int8x3_type<cuDoubleComplex> {
    using type = char3;
};
template <typename T> using int8x3_t = typename int8x3_type<T>::type;

//==========
// Map type to single- or triple-int8 representation
//==========
template <typename T> struct int8x2_type;
template <> struct int8x2_type<float> {
    using type = int8_t;
};
template <> struct int8x2_type<double> {
    using type = int8_t;
};
template <> struct int8x2_type<cuFloatComplex> {
    using type = char2;
};
template <> struct int8x2_type<cuDoubleComplex> {
    using type = char2;
};
template <typename T> using int8x2_t = typename int8x2_type<T>::type;

//==========
// Map type to single- or double-int16 representation
//==========
template <typename T> struct int16x2_type;
template <> struct int16x2_type<float> {
    using type = int16_t;
};
template <> struct int16x2_type<double> {
    using type = int16_t;
};
template <> struct int16x2_type<cuFloatComplex> {
    using type = short2;
};
template <> struct int16x2_type<cuDoubleComplex> {
    using type = short2;
};
template <typename T> using int16x2_t = typename int16x2_type<T>::type;

//==========
// Map type to single- or double-double representation
//==========
template <typename T> struct doublex_type;
template <> struct doublex_type<float> {
    using type = double;
};
template <> struct doublex_type<double> {
    using type = double;
};
template <> struct doublex_type<cuFloatComplex> {
    using type = cuDoubleComplex;
};
template <> struct doublex_type<cuDoubleComplex> {
    using type = cuDoubleComplex;
};
template <typename T> using doublex_t = typename doublex_type<T>::type;

//==========
// Check if type is complex
//==========
template <typename T> inline constexpr bool isComplex        = false;
template <> inline constexpr bool isComplex<cuDoubleComplex> = true;
template <> inline constexpr bool isComplex<cuFloatComplex>  = true;

//==========
// Floating-point traits
//==========
template <typename T> struct fp;
template <> struct fp<double> {
    static constexpr int32_t bias = 1023;
    static constexpr int32_t prec = 52;
    static constexpr int32_t bits = 64;
    static constexpr double th    = 0x1p+60;
};
template <> struct fp<float> {
    static constexpr int32_t bias = 127;
    static constexpr int32_t prec = 23;
    static constexpr int32_t bits = 32;
    static constexpr float th     = 0x1p+31f;
};

//==========
// Type-specific constants
//==========
template <typename T> struct Tconst {
    __device__ __host__ __forceinline__ static constexpr T zero() { return static_cast<T>(0); }
    __device__ __host__ __forceinline__ static constexpr T one() { return static_cast<T>(1.0); }
    __device__ __host__ __forceinline__ static constexpr T mone() { return static_cast<T>(-1.0); }
};

template <> struct Tconst<double2> {
    __device__ __host__ __forceinline__ static constexpr double2 zero() { return {0.0, 0.0}; }
    __device__ __host__ __forceinline__ static constexpr double2 one() { return {1.0, 0.0}; }
    __device__ __host__ __forceinline__ static constexpr double2 mone() { return {-1.0, 0.0}; }
};

template <> struct Tconst<float2> {
    __device__ __host__ __forceinline__ static constexpr float2 zero() { return {0.0f, 0.0f}; }
    __device__ __host__ __forceinline__ static constexpr float2 one() { return {1.0f, 0.0f}; }
    __device__ __host__ __forceinline__ static constexpr float2 mone() { return {-1.0f, 0.0f}; }
};

template <> struct Tconst<char2> {
    __device__ __host__ __forceinline__ static constexpr char2 zero() {
        return {static_cast<int8_t>(0), static_cast<int8_t>(0)};
    }
};

template <> struct Tconst<char3> {
    __device__ __host__ __forceinline__ static constexpr char3 zero() {
        return {static_cast<int8_t>(0), static_cast<int8_t>(0), static_cast<int8_t>(0)};
    }
};

} // namespace oz2
