#pragma once

#if defined(__HIPCC__)
    #include <amd_smi/amdsmi.h>
    #include <hip/hip_complex.h>
    #include <hip/hip_runtime.h>
    #include <hipblas/hipblas.h>
    #include <hiprand/hiprand_kernel.h>

    #define cublasCreate                hipblasCreate
    #define cublasDestroy               hipblasDestroy
    #define cublasHandle_t              hipblasHandle_t
    #define cublasOperation_t           hipblasOperation_t
    #define cublasStatus_t              hipblasStatus_t
    #define cublasComputeType_t         hipblasComputeType_t
    #define cublasGemmAlgo_t            hipblasGemmAlgo_t
    #define CUBLAS_STATUS_SUCCESS       HIPBLAS_STATUS_SUCCESS
    #define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
    #if defined(HIPBLAS_V2)
        #define cublasGemmEx hipblasGemmEx
        #define cublasCgemm  hipblasCgemm
        #define cublasZgemm  hipblasZgemm
    #else
        #define cublasGemmEx hipblasGemmEx_v2
        #define cublasCgemm  hipblasCgemm_v2
        #define cublasZgemm  hipblasZgemm_v2
    #endif
    #define cublasSgemm_v2                 hipblasSgemm
    #define cublasDgemm_v2                 hipblasDgemm
    #define CUBLAS_OP_N                    HIPBLAS_OP_N
    #define CUBLAS_OP_T                    HIPBLAS_OP_T
    #define CUBLAS_OP_C                    HIPBLAS_OP_C
    #define CUDA_R_8I                      HIP_R_8I
    #define CUDA_R_32I                     HIP_R_32I
    #define CUDA_R_32F                     HIP_R_32F
    #define CUDA_R_64F                     HIP_R_64F
    #define CUDA_C_32F                     HIP_C_32F
    #define CUDA_C_64F                     HIP_C_64F
    #define CUBLAS_COMPUTE_32I             HIPBLAS_COMPUTE_32I
    #define CUBLAS_GEMM_DEFAULT            HIPBLAS_GEMM_DEFAULT
    #define CUBLAS_COMPUTE_32F             HIPBLAS_COMPUTE_32F
    #define CUBLAS_COMPUTE_64F             HIPBLAS_COMPUTE_64F
    #define CUBLAS_COMPUTE_32F_FAST_TF32   HIPBLAS_COMPUTE_32F_FAST_TF32
    #define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
    #define CUBLAS_STATUS_INVALID_VALUE    HIPBLAS_STATUS_INVALID_VALUE

    #define cudaDeviceSynchronize                  hipDeviceSynchronize
    #define cudaMemcpyToSymbol(symbol, src, count) hipMemcpyToSymbol(HIP_SYMBOL(symbol), src, count)
    #define cudaDeviceProp                         hipDeviceProp_t
    #define cudaGetDeviceProperties                hipGetDeviceProperties
    #define cudaMalloc                             hipMalloc
    #define cudaMemcpy                             hipMemcpy
    #define cudaMemcpyDeviceToHost                 hipMemcpyDeviceToHost
    #define cudaFree                               hipFree
    #define cudaError_t                            hipError_t
    #define cudaSuccess                            hipSuccess
    #define cudaDataType                           hipDataType
    #define cudaGetErrorString                     hipGetErrorString

    #define cuComplex              hipFloatComplex
    #define cuFloatComplex         hipFloatComplex
    #define cuDoubleComplex        hipDoubleComplex
    #define cuCreal                hipCreal
    #define cuCrealf               hipCrealf
    #define cuCimag                hipCimag
    #define cuCimagf               hipCimagf
    #define make_cuComplex         make_hipFloatComplex
    #define make_cuFloatComplex    make_hipFloatComplex
    #define make_cuDoubleComplex   make_hipDoubleComplex
    #define cuCmul                 hipCmul
    #define cuCmulf                hipCmulf
    #define cuCadd                 hipCadd
    #define cuCaddf                hipCaddf
    #define cuCfma                 hipCfma
    #define cuCfmaf                hipCfmaf
    #define cuCabs                 hipCabs
    #define cuCsub                 hipCsub
    #define cuComplexFloatToDouble hipComplexFloatToDouble
    #define cuComplexDoubleToFloat hipComplexDoubleToFloat

    #define curandState           hiprandState
    #define curand_init           hiprand_init
    #define curand_uniform_double hiprand_uniform_double
    #define curand_normal_double  hiprand_normal_double

    #define nvmlShutdown amdsmi_shut_down
    #define NVML_SUCCESS AMDSMI_STATUS_SUCCESS
    #define nvmlReturn_t amdsmi_status_t
    #define nvmlInit()   amdsmi_init(AMDSMI_INIT_AMD_GPUS)
    #define nvmlDevice_t amdsmi_processor_handle

namespace getWatt {
const char *nvmlErrorString(amdsmi_status_t result) {
    const char *error_string = nullptr;
    amdsmi_status_code_to_string(result, &error_string);
    return error_string;
}

amdsmi_status_t nvmlDeviceGetHandleByIndex_v2(unsigned gpu_id, amdsmi_processor_handle *device) {
    unsigned socket_count  = 0;
    amdsmi_status_t result = amdsmi_get_socket_handles(&socket_count, nullptr);
    if (result != AMDSMI_STATUS_SUCCESS) return result;

    std::vector<amdsmi_socket_handle> sockets(socket_count);
    result = amdsmi_get_socket_handles(&socket_count, &sockets[0]);
    if (result != AMDSMI_STATUS_SUCCESS) return result;

    unsigned device_count = 0;
    result                = amdsmi_get_processor_handles(sockets[0], &device_count, nullptr);

    std::vector<amdsmi_processor_handle> proc_handles(device_count);
    result = amdsmi_get_processor_handles(sockets[0], &device_count, &proc_handles[0]);
    if (result != AMDSMI_STATUS_SUCCESS) return result;

    *device = proc_handles[gpu_id];
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t nvmlDeviceGetPowerUsage(amdsmi_processor_handle device, unsigned *mw) {
    amdsmi_power_info_t info{};
    amdsmi_status_t result = amdsmi_get_power_info(device, &info);
    if (result != AMDSMI_STATUS_SUCCESS) return result;
    *mw = (info.average_socket_power >= 10000)
              ? static_cast<unsigned>(info.current_socket_power * 1000.0)
              : static_cast<unsigned>(info.average_socket_power * 1000.0);
    return AMDSMI_STATUS_SUCCESS;
}
} // namespace getWatt

#elif defined(__NVCC__)
    #include <cuComplex.h>
    #include <cublas_v2.h>
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
    #include <nvml.h>

#endif
