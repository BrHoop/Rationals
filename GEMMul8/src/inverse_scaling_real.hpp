#pragma once
#include "common.hpp"
#include "conv_32i_2_8i_real.hpp"
#include "template_math.hpp"

namespace oz2 {
namespace real {

//------------------------------
// Accumulation, final reduction & Undo scaling
//------------------------------
template <typename TC, typename TP, typename TCtmp>
__forceinline__ __device__ TC invscal_device(
    const unsigned num_moduli,            // number of moduli
    const size_t incCtmp,                 // increment
    const TCtmp *const __restrict__ Ctmp, // input
    const TP P,                           // prod(moduli)
    const double invP,                    // 1/prod(moduli)
    const int16_t sft                     // exponent of shift values
) {
    if constexpr (std::is_same_v<TP, double>) {

        // sum(qi*Pi*C8i[i])
        double C64f = 0.0;
        if constexpr (std::is_same_v<TCtmp, int8_t>) {
            for (unsigned i = 0; i < num_moduli; ++i) {
                double C8i_d = static_cast<double>(Ctmp[i * incCtmp]);
                double qPi   = table::qPi[i];
                C64f         = fma(qPi, C8i_d, C64f);
            }
        } else {
            {
                double C8i_d = conv_32i_2_8i_256_scal(Ctmp[0]);
                double qPi   = table::qPi[0];
                C64f         = qPi * C8i_d;
            }
            for (unsigned i = 1; i < num_moduli; ++i) {
                double C8i_d = conv_32i_2_8i_not256_scal(Ctmp[i * incCtmp], i - 1);
                double qPi   = table::qPi[i];
                C64f         = fma(qPi, C8i_d, C64f);
            }
        }

        double quot    = rint(invP * C64f);                          // round(C64f/P)
        double CRT_ans = fma(P, quot, C64f);                         // C64f - P*round(C64f/P)
        TC C           = Tscalbn<TC>(static_cast<TC>(CRT_ans), sft); // undo scaling
        return C;

    } else if constexpr (std::is_same_v<TP, double2>) {

        // sum(qi*Pi*C8i[i])
        double2 C64f{};
        if constexpr (std::is_same_v<TCtmp, int8_t>) {
            for (unsigned i = 0; i < num_moduli; ++i) {
                double C8i_d = static_cast<double>(Ctmp[i * incCtmp]);
                double2 qPi  = reinterpret_cast<double2 *>(table::qPi)[i];
                C64f.x       = fma(qPi.x, C8i_d, C64f.x); // error-free
                C64f.y       = fma(qPi.y, C8i_d, C64f.y); // non-error-free
            }
        } else {
            {
                double C8i_d = conv_32i_2_8i_256_scal(Ctmp[0]);
                double2 qPi  = reinterpret_cast<double2 *>(table::qPi)[0];
                C64f.x       = qPi.x * C8i_d; // error-free
                C64f.y       = qPi.y * C8i_d; // non-error-free
            }
            for (unsigned i = 1; i < num_moduli; ++i) {
                double C8i_d = conv_32i_2_8i_not256_scal(Ctmp[i * incCtmp], i - 1);
                double2 qPi  = reinterpret_cast<double2 *>(table::qPi)[i];
                C64f.x       = fma(qPi.x, C8i_d, C64f.x); // error-free
                C64f.y       = fma(qPi.y, C8i_d, C64f.y); // non-error-free
            }
        }

        double quot    = rint(invP * C64f.x);                             // round(C64f/P)
        double CRT_ans = fma(P.y, quot, fma(P.x, quot, C64f.x) + C64f.y); // C64f - P*round(C64f/P)
        TC C           = Tscalbn<TC>(static_cast<TC>(CRT_ans), sft);      // undo scaling
        return C;

    } else {
        return Tconst<TC>::zero();
    }
}

//------------------------------
// C := alpha*AB + beta*C
//------------------------------
template <typename TC, typename TP, typename TCtmp>
__global__ void invscal_kernel_general(
    const TC alpha, const TC beta,          //
    const unsigned num_moduli,              // number of moduli
    const size_t m,                         // size(C64f,1)
    const size_t sizeC,                     // m*n
    const size_t incCtmp,                   // increment
    const TCtmp *const __restrict__ Ctmp,   // input
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
    TC AB              = invscal_device<TC, TP, TCtmp>(num_moduli, incCtmp, Ctmp + mem_idx, P, invP, sftA[row] + sftB[col]);

    const auto idxC = col * ldc + row;
    C[idxC]         = Taxpby_scal<TC>(alpha, AB, beta, C[idxC]);
}

//------------------------------
// C := alpha*AB + beta*C
//------------------------------
template <typename TC, typename TP, typename TCtmp, int ALPHA, int BETA>
__global__ void invscal_kernel_special(
    const unsigned num_moduli,              // number of moduli
    const size_t m,                         // size(C64f,1)
    const size_t sizeC,                     // m*n
    const size_t incCtmp,                   // increment
    const TCtmp *const __restrict__ Ctmp,   // input
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
    TC AB              = invscal_device<TC, TP, TCtmp>(num_moduli, incCtmp, Ctmp + mem_idx, P, invP, sftA[row] + sftB[col]);

    const auto idxC = col * ldc + row;
    if constexpr (ALPHA == 1 && BETA == 0) {
        // C[idxC] =  1*AB + 0*C[idxC] = AB
        C[idxC] = AB;

    } else if constexpr (ALPHA == 1 && BETA == 1) {
        // C[idxC] =  1*AB + 1*C[idxC] = C[idxC] + AB
        C[idxC] += AB;

    } else if constexpr (ALPHA == -1 && BETA == 0) {
        // C[idxC] = -1*AB + 0*C[idxC] = -AB
        C[idxC] = -AB;

    } else if constexpr (ALPHA == -1 && BETA == 1) {
        // C[idxC] = -1*AB + 1*C[idxC] = C[idxC] - AB
        C[idxC] -= AB;
    }
}

//------------------------------
// Interface!!
//------------------------------
template <typename T, typename TCtmp = int8_t>
__inline__ void inverse_scaling(
    const bool P_is_double,
    const unsigned num_moduli,      // number of moduli
    const size_t m, const size_t n, // size(C)
    const TCtmp *const Ctmp,        // input
    const size_t ldctmp,            // leading dim of Ctmp
    const size_t incCtmp,           // increment
    T *const C,                     // output
    const size_t ldc,               // leading dimension
    const int16_t *const sftA,      // exponent of shift values for rows of A
    const int16_t *const sftB,      // exponent of shift values for cols of B
    const T alpha, const T beta     //
) {
    const unsigned table_idx = num_moduli - 2;
    const size_t sizeC       = m * n;
    const double invP        = table::invP[table_idx];

    if (P_is_double) {
        const double P = table::P[table_idx].x;

        if (alpha == Tconst<T>::one()) {
            if (beta == Tconst<T>::zero()) {
                invscal_kernel_special<T, double, TCtmp, 1, 0><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                return;
            } else if (beta == Tconst<T>::one()) {
                invscal_kernel_special<T, double, TCtmp, 1, 1><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                return;
            }
        } else if (alpha == Tconst<T>::mone()) {
            if (beta == Tconst<T>::zero()) {
                invscal_kernel_special<T, double, TCtmp, -1, 0><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                return;
            } else if (beta == Tconst<T>::one()) {
                invscal_kernel_special<T, double, TCtmp, -1, 1><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                return;
            }
        }
        invscal_kernel_general<T, double, TCtmp><<<grid_invscal, threads_invscal>>>(alpha, beta, num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);

    } else {
        const double2 P = table::P[table_idx];

        if (alpha == Tconst<T>::one()) {
            if (beta == Tconst<T>::zero()) {
                invscal_kernel_special<T, double2, TCtmp, 1, 0><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                return;
            } else if (beta == Tconst<T>::one()) {
                invscal_kernel_special<T, double2, TCtmp, 1, 1><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                return;
            }
        } else if (alpha == Tconst<T>::mone()) {
            if (beta == Tconst<T>::zero()) {
                invscal_kernel_special<T, double2, TCtmp, -1, 0><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                return;
            } else if (beta == Tconst<T>::one()) {
                invscal_kernel_special<T, double2, TCtmp, -1, 1><<<grid_invscal, threads_invscal>>>(num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
                return;
            }
        }
        invscal_kernel_general<T, double2, TCtmp><<<grid_invscal, threads_invscal>>>(alpha, beta, num_moduli, m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, P, invP, sftA, sftB);
    }
}

} // namespace real
} // namespace oz2
