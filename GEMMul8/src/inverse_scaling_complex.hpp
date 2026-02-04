#pragma once
#include "common.hpp"
#include "inverse_scaling_real.hpp"
#include "template_math.hpp"

namespace oz2 {
namespace complex {

//------------------------------
// Accumulation, final reduction & Undo scaling
//------------------------------
template <typename TC, typename TP>
__forceinline__ __device__ TC invscal_device(
    const unsigned num_moduli,           // number of moduli
    const size_t incC8i,                 // increment
    const char2 *const __restrict__ C8i, // input
    const TP P,                          // prod(moduli)
    const double invP,                   // 1/prod(moduli)
    const int16_t sft                    // exponent of shift values
) {
    if constexpr (std::is_same_v<TP, double>) {

        // sum(qi*Pi*C8i[i])
        double2 C64f{};
        for (unsigned i = 0; i < num_moduli; ++i) {
            double2 C8i_d;
            char2 C8i_tmp = C8i[i * incC8i];
            C8i_d.x       = static_cast<double>(C8i_tmp.x);
            C8i_d.y       = static_cast<double>(C8i_tmp.y);
            double qPi    = table::qPi[i];
            C64f.x        = fma(qPi, C8i_d.x, C64f.x);
            C64f.y        = fma(qPi, C8i_d.y, C64f.y);
        }

        // round(C64f/P)
        double2 quot;
        quot.x = rint(invP * C64f.x);
        quot.y = rint(invP * C64f.y);

        // C64f - P*round(C64f/P)
        double2 CRT_ans;
        CRT_ans.x = fma(P, quot.x, C64f.x);
        CRT_ans.y = fma(P, quot.y, C64f.y);

        // undo scaling
        using U = underlying_t<TC>;
        TC C;
        C.x = Tscalbn<U>(static_cast<U>(CRT_ans.x), sft);
        C.y = Tscalbn<U>(static_cast<U>(CRT_ans.y), sft);

        return C;

    } else if constexpr (std::is_same_v<TP, double2>) {

        // sum(qi*Pi*C8i[i])
        double2 C64f_hi{};
        double2 C64f_lo{};
        for (unsigned i = 0; i < num_moduli; ++i) {
            double2 C8i_d;
            char2 C8i_tmp = C8i[i * incC8i];
            C8i_d.x       = static_cast<double>(C8i_tmp.x);
            C8i_d.y       = static_cast<double>(C8i_tmp.y);
            double2 qPi   = reinterpret_cast<double2 *>(table::qPi)[i];
            C64f_hi.x     = fma(qPi.x, C8i_d.x, C64f_hi.x); // error-free
            C64f_hi.y     = fma(qPi.x, C8i_d.y, C64f_hi.y); // error-free
            C64f_lo.x     = fma(qPi.y, C8i_d.x, C64f_lo.x); // non-error-free
            C64f_lo.y     = fma(qPi.y, C8i_d.y, C64f_lo.y); // non-error-free
        }

        // round(C64f/P)
        double2 quot;
        quot.x = rint(invP * C64f_hi.x);
        quot.y = rint(invP * C64f_hi.y);

        // C64f - P*round(C64f/P)
        double2 CRT_ans;
        CRT_ans.x = fma(P.y, quot.x, fma(P.x, quot.x, C64f_hi.x) + C64f_lo.x);
        CRT_ans.y = fma(P.y, quot.y, fma(P.x, quot.y, C64f_hi.y) + C64f_lo.y);

        // undo scaling
        using U = underlying_t<TC>;
        TC C;
        C.x = Tscalbn<U>(static_cast<U>(CRT_ans.x), sft);
        C.y = Tscalbn<U>(static_cast<U>(CRT_ans.y), sft);

        return C;

    } else {
        return Tconst<TC>::zero();
    }
}

template <typename TC, typename TP>
__forceinline__ __device__ TC invscal_device(
    const unsigned num_moduli,            // number of moduli
    const size_t incCtmp,                 // increment
    const int32_t *const __restrict__ C0, // input
    const int32_t *const __restrict__ C1, // input
    const int32_t *const __restrict__ C2, // input
    const TP P,                           // prod(moduli)
    const double invP,                    // 1/prod(moduli)
    const int16_t sft                     // exponent of shift values
) {
    if constexpr (std::is_same_v<TP, double>) {

        // sum(qi*Pi*C8i[i])
        double2 C64f;
        {
            double2 C8i_d = conv_32i_2_8i_256_scal(C0[0], C1[0], C2[0]);
            double qPi    = table::qPi[0];
            C64f.x        = qPi * C8i_d.x;
            C64f.y        = qPi * C8i_d.y;
        }
        for (unsigned i = 1; i < num_moduli; ++i) {
            double2 C8i_d = conv_32i_2_8i_not256_scal(C0[i * incCtmp], C1[i * incCtmp], C2[i * incCtmp], i - 1);
            double qPi    = table::qPi[i];
            C64f.x        = fma(qPi, C8i_d.x, C64f.x);
            C64f.y        = fma(qPi, C8i_d.y, C64f.y);
        }

        // round(C64f/P)
        double2 quot;
        quot.x = rint(invP * C64f.x);
        quot.y = rint(invP * C64f.y);

        // C64f - P*round(C64f/P)
        double2 CRT_ans;
        CRT_ans.x = fma(P, quot.x, C64f.x);
        CRT_ans.y = fma(P, quot.y, C64f.y);

        // undo scaling
        using U = underlying_t<TC>;
        TC C;
        C.x = Tscalbn<U>(static_cast<U>(CRT_ans.x), sft);
        C.y = Tscalbn<U>(static_cast<U>(CRT_ans.y), sft);

        return C;

    } else if constexpr (std::is_same_v<TP, double2>) {

        // sum(qi*Pi*C8i[i])
        double2 C64f_hi;
        double2 C64f_lo;
        {
            double2 C8i_d = conv_32i_2_8i_256_scal(C0[0], C1[0], C2[0]);
            double2 qPi   = reinterpret_cast<double2 *>(table::qPi)[0];
            C64f_hi.x     = qPi.x * C8i_d.x; // error-free
            C64f_hi.y     = qPi.x * C8i_d.y; // error-free
            C64f_lo.x     = qPi.y * C8i_d.x; // non-error-free
            C64f_lo.y     = qPi.y * C8i_d.y; // non-error-free
        }
        for (unsigned i = 1; i < num_moduli; ++i) {
            double2 C8i_d = conv_32i_2_8i_not256_scal(C0[i * incCtmp], C1[i * incCtmp], C2[i * incCtmp], i - 1);
            double2 qPi   = reinterpret_cast<double2 *>(table::qPi)[i];
            C64f_hi.x     = fma(qPi.x, C8i_d.x, C64f_hi.x); // error-free
            C64f_hi.y     = fma(qPi.x, C8i_d.y, C64f_hi.y); // error-free
            C64f_lo.x     = fma(qPi.y, C8i_d.x, C64f_lo.x); // non-error-free
            C64f_lo.y     = fma(qPi.y, C8i_d.y, C64f_lo.y); // non-error-free
        }

        // round(C64f/P)
        double2 quot;
        quot.x = rint(invP * C64f_hi.x);
        quot.y = rint(invP * C64f_hi.y);

        // C64f - P*round(C64f/P)
        double2 CRT_ans;
        CRT_ans.x = fma(P.y, quot.x, fma(P.x, quot.x, C64f_hi.x) + C64f_lo.x);
        CRT_ans.y = fma(P.y, quot.y, fma(P.x, quot.y, C64f_hi.y) + C64f_lo.y);

        // undo scaling
        using U = underlying_t<TC>;
        TC C;
        C.x = Tscalbn<U>(static_cast<U>(CRT_ans.x), sft);
        C.y = Tscalbn<U>(static_cast<U>(CRT_ans.y), sft);

        return C;

    } else {
        return Tconst<TC>::zero();
    }
}

//------------------------------
// C := alpha*AB + beta*C
//------------------------------
template <typename TC, typename TP>
__global__ void invscal_kernel_general(
    const TC alpha, const TC beta,          //
    const unsigned num_moduli,              // number of moduli
    const size_t m,                         // size(C64f,1)
    const size_t sizeC,                     // m*n
    const size_t incCtmp,                   // increment
    const char2 *const __restrict__ Ctmp,   // input
    const size_t ldctmp,                    // leading dim of Ctmp
    TC *const __restrict__ C,               // output
    const size_t ldc,                       // leading dimension
    const TP P,                             // -prod(moduli)
    const double invP,                      // 1/prod(moduli)
    const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
    const int16_t *const __restrict__ sftB  // exponent of shift values for cols of B
) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col     = idx / m;
    const auto row     = idx - col * m;
    const auto mem_idx = col * ldctmp + row;
    TC AB              = invscal_device<TC, TP>(num_moduli, incCtmp, Ctmp + mem_idx, P, invP, sftA[row] + sftB[col]);

    const auto idxC = col * ldc + row;
    C[idxC]         = Taxpby_scal<TC>(alpha, AB, beta, C[idxC]);
}

template <typename TC, typename TP>
__global__ void invscal_kernel_general(
    const TC alpha, const TC beta,          //
    const unsigned num_moduli,              // number of moduli
    const size_t m,                         // size(C64f,1)
    const size_t sizeC,                     // m*n
    const size_t incCtmp,                   // increment
    const int32_t *const __restrict__ C0,   // input
    const int32_t *const __restrict__ C1,   // input
    const int32_t *const __restrict__ C2,   // input
    const size_t ldctmp,                    // leading dim of Ctmp
    TC *const __restrict__ C,               // output
    const size_t ldc,                       // leading dimension
    const TP P,                             // -prod(moduli)
    const double invP,                      // 1/prod(moduli)
    const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
    const int16_t *const __restrict__ sftB  // exponent of shift values for cols of B
) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col     = idx / m;
    const auto row     = idx - col * m;
    const auto mem_idx = col * ldctmp + row;
    TC AB              = invscal_device<TC, TP>(num_moduli, incCtmp, C0 + mem_idx, C1 + mem_idx, C2 + mem_idx, P, invP, sftA[row] + sftB[col]);

    const auto idxC = col * ldc + row;
    C[idxC]         = Taxpby_scal<TC>(alpha, AB, beta, C[idxC]);
}

//------------------------------
// C := alpha*AB + beta*C
//------------------------------
template <typename TC, typename TP, int ALPHA, int BETA>
__global__ void invscal_kernel_special(
    const unsigned num_moduli,              // number of moduli
    const size_t m,                         // size(C64f,1)
    const size_t sizeC,                     // m*n
    const size_t incCtmp,                   // increment
    const char2 *const __restrict__ C8i,    // input
    const size_t ldctmp,                    // leading dim of Ctmp
    TC *const __restrict__ C,               // output
    const size_t ldc,                       // leading dimension
    const TP P,                             // -prod(moduli)
    const double invP,                      // 1/prod(moduli)
    const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
    const int16_t *const __restrict__ sftB  // exponent of shift values for cols of B
) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col     = idx / m;
    const auto row     = idx - col * m;
    const auto mem_idx = col * ldctmp + row;
    TC AB              = invscal_device<TC, TP>(num_moduli, incCtmp, C8i + mem_idx, P, invP, sftA[row] + sftB[col]);

    const auto idxC = col * ldc + row;
    if constexpr (ALPHA == 1 && BETA == 0) {
        // C[idxC] =  1*AB + 0*C[idxC] = AB
        C[idxC] = AB;

    } else if constexpr (ALPHA == 1 && BETA == 1) {
        // C[idxC] =  1*AB + 1*C[idxC] = C[idxC] + AB
        TC Ctmp = C[idxC];
        Ctmp.x += AB.x;
        Ctmp.y += AB.y;
        C[idxC] = Ctmp;

    } else if constexpr (ALPHA == -1 && BETA == 0) {
        // C[idxC] = -1*AB + 0*C[idxC] = -AB
        AB.x    = -AB.x;
        AB.y    = -AB.y;
        C[idxC] = AB;

    } else if constexpr (ALPHA == -1 && BETA == 1) {
        // C[idxC] = -1*AB + 1*C[idxC] = C[idxC] - AB
        TC Ctmp = C[idxC];
        Ctmp.x -= AB.x;
        Ctmp.y -= AB.y;
        C[idxC] = Ctmp;
    }
}

template <typename TC, typename TP, int ALPHA, int BETA>
__global__ void invscal_kernel_special(
    const unsigned num_moduli,              // number of moduli
    const size_t m,                         // size(C64f,1)
    const size_t sizeC,                     // m*n
    const size_t incCtmp,                   // increment
    const int32_t *const __restrict__ C0,   // input
    const int32_t *const __restrict__ C1,   // input
    const int32_t *const __restrict__ C2,   // input
    const size_t ldctmp,                    // leading dim of Ctmp
    TC *const __restrict__ C,               // output
    const size_t ldc,                       // leading dimension
    const TP P,                             // -prod(moduli)
    const double invP,                      // 1/prod(moduli)
    const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
    const int16_t *const __restrict__ sftB  // exponent of shift values for cols of B
) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col     = idx / m;
    const auto row     = idx - col * m;
    const auto mem_idx = col * ldctmp + row;
    TC AB              = invscal_device<TC, TP>(num_moduli, incCtmp, C0 + mem_idx, C1 + mem_idx, C2 + mem_idx, P, invP, sftA[row] + sftB[col]);

    const auto idxC = col * ldc + row;
    if constexpr (ALPHA == 1 && BETA == 0) {
        // C[idxC] =  1*AB + 0*C[idxC] = AB
        C[idxC] = AB;

    } else if constexpr (ALPHA == 1 && BETA == 1) {
        // C[idxC] =  1*AB + 1*C[idxC] = C[idxC] + AB
        TC Ctmp = C[idxC];
        Ctmp.x += AB.x;
        Ctmp.y += AB.y;
        C[idxC] = Ctmp;

    } else if constexpr (ALPHA == -1 && BETA == 0) {
        // C[idxC] = -1*AB + 0*C[idxC] = -AB
        AB.x    = -AB.x;
        AB.y    = -AB.y;
        C[idxC] = AB;

    } else if constexpr (ALPHA == -1 && BETA == 1) {
        // C[idxC] = -1*AB + 1*C[idxC] = C[idxC] - AB
        TC Ctmp = C[idxC];
        Ctmp.x -= AB.x;
        Ctmp.y -= AB.y;
        C[idxC] = Ctmp;
    }
}

//------------------------------
// Interface!!
//------------------------------
template <typename T, typename TCtmp = const char2 *const>
__inline__ void inverse_scaling(
    const bool P_is_double,
    const unsigned num_moduli,      // number of moduli
    const size_t m, const size_t n, // size(C)
    TCtmp Ctmp,                     // input
    const size_t ldctmp,            // leading dim of Ctmp
    const size_t incCtmp,           // increment
    T *const C,                     // output
    const size_t ldc,               // leading dimension
    const int16_t *const sftA,      // exponent of shift values for rows of A
    const int16_t *const sftB,      // exponent of shift values for cols of B
    const T alpha, const T beta     //
) {
    using U                  = underlying_t<T>;
    const unsigned table_idx = num_moduli - 2;
    const size_t sizeC       = m * n;
    const double invP        = table::invP[table_idx];

    if constexpr (std::is_same_v<TCtmp, const char2 *const>) {

        if (P_is_double) {
            const double P = table::P[table_idx].x;

            if (alpha.x == Tconst<U>::one() && alpha.y == Tconst<U>::zero()) {
                if (beta.x == Tconst<U>::zero() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double, 1, 0><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                } else if (beta.x == Tconst<U>::one() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double, 1, 1><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                }
            } else if (alpha.x == Tconst<U>::mone() && alpha.y == Tconst<U>::zero()) {
                if (beta.x == Tconst<U>::zero() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double, -1, 0><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                } else if (beta.x == Tconst<U>::one() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double, -1, 1><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                }
            }
            invscal_kernel_general<T, double><<<grid_invscal, threads_invscal>>>(alpha, beta, num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);

        } else {
            const double2 P = table::P[table_idx];

            if (alpha.x == Tconst<U>::one() && alpha.y == Tconst<U>::zero()) {
                if (beta.x == Tconst<U>::zero() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double2, 1, 0><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                } else if (beta.x == Tconst<U>::one() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double2, 1, 1><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                }
            } else if (alpha.x == Tconst<U>::mone() && alpha.y == Tconst<U>::zero()) {
                if (beta.x == Tconst<U>::zero() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double2, -1, 0><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                } else if (beta.x == Tconst<U>::one() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double2, -1, 1><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                }
            }
            invscal_kernel_general<T, double2><<<grid_invscal, threads_invscal>>>(alpha, beta, num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
        }

    } else {

        if (P_is_double) {
            const double P = table::P[table_idx].x;

            if (alpha.x == Tconst<U>::one() && alpha.y == Tconst<U>::zero()) {
                if (beta.x == Tconst<U>::zero() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double, 1, 0><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp[0], Ctmp[1], Ctmp[2], ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                } else if (beta.x == Tconst<U>::one() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double, 1, 1><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp[0], Ctmp[1], Ctmp[2], ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                }
            } else if (alpha.x == Tconst<U>::mone() && alpha.y == Tconst<U>::zero()) {
                if (beta.x == Tconst<U>::zero() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double, -1, 0><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp[0], Ctmp[1], Ctmp[2], ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                } else if (beta.x == Tconst<U>::one() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double, -1, 1><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp[0], Ctmp[1], Ctmp[2], ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                }
            }
            invscal_kernel_general<T, double><<<grid_invscal, threads_invscal>>>(alpha, beta, num_moduli, m, sizeC, incCtmp, Ctmp[0], Ctmp[1], Ctmp[2], ldctmp, C, ldc, P, invP, sftA, sftB);

        } else {
            const double2 P = table::P[table_idx];

            if (alpha.x == Tconst<U>::one() && alpha.y == Tconst<U>::zero()) {
                if (beta.x == Tconst<U>::zero() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double2, 1, 0><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp[0], Ctmp[1], Ctmp[2], ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                } else if (beta.x == Tconst<U>::one() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double2, 1, 1><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp[0], Ctmp[1], Ctmp[2], ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                }
            } else if (alpha.x == Tconst<U>::mone() && alpha.y == Tconst<U>::zero()) {
                if (beta.x == Tconst<U>::zero() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double2, -1, 0><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp[0], Ctmp[1], Ctmp[2], ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                } else if (beta.x == Tconst<U>::one() && beta.y == Tconst<U>::zero()) {
                    invscal_kernel_special<T, double2, -1, 1><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp[0], Ctmp[1], Ctmp[2], ldctmp, C, ldc, P, invP, sftA, sftB);
                    return;
                }
            }
            invscal_kernel_general<T, double2><<<grid_invscal, threads_invscal>>>(alpha, beta, num_moduli, m, sizeC, incCtmp, Ctmp[0], Ctmp[1], Ctmp[2], ldctmp, C, ldc, P, invP, sftA, sftB);
        }
    }
}

} // namespace complex
} // namespace oz2
