#include <cstdint>
#include <cstring>
#include <string_view>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "xla/ffi/api/c_api.h"

#include "../include/gemmul8.hpp"

extern "C" {

// Opaque descriptor passed from Python.
struct Gemmul8Descriptor {
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  uint32_t num_moduli;
  uint32_t fastmode;
};

static_assert(sizeof(Gemmul8Descriptor) == 56, "Gemmul8Descriptor size mismatch");

static inline cublasHandle_t get_handle(cudaStream_t stream) {
  static thread_local cublasHandle_t handle = nullptr;
  if (!handle) {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      return nullptr;
    }
  }
  cublasSetStream(handle, stream);
  return handle;
}

static inline XLA_FFI_Error* make_error(const XLA_FFI_Api* api,
                                        XLA_FFI_Error_Code code,
                                        const char* msg) {
  XLA_FFI_Error_Create_Args args;
  args.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.message = msg;
  args.errc = code;
  return api->XLA_FFI_Error_Create(&args);
}

static inline bool bytespan_eq(const XLA_FFI_ByteSpan* span, const char* name) {
  if (!span || !name) return false;
  size_t len = std::strlen(name);
  return span->len == len && std::memcmp(span->ptr, name, len) == 0;
}

static bool get_opaque(const XLA_FFI_CallFrame* call_frame,
                       const char** data,
                       size_t* len) {
  if (call_frame->attrs.size <= 0) return false;

  for (int64_t i = 0; i < call_frame->attrs.size; ++i) {
    const XLA_FFI_ByteSpan* name = call_frame->attrs.names[i];
    const XLA_FFI_AttrType type = call_frame->attrs.types[i];
    void* attr = call_frame->attrs.attrs[i];

    if (!(bytespan_eq(name, "opaque") || bytespan_eq(name, "backend_config"))) {
      continue;
    }

    if (type == XLA_FFI_AttrType_STRING) {
      auto* span = reinterpret_cast<XLA_FFI_ByteSpan*>(attr);
      *data = span->ptr;
      *len = span->len;
      return true;
    }
    if (type == XLA_FFI_AttrType_ARRAY) {
      auto* arr = reinterpret_cast<XLA_FFI_Array*>(attr);
      if (arr->dtype != XLA_FFI_DataType_U8) return false;
      *data = reinterpret_cast<const char*>(arr->data);
      *len = arr->size;
      return true;
    }
  }

  // Fallback: if there is exactly one attribute, use it regardless of name.
  if (call_frame->attrs.size == 1) {
    const XLA_FFI_AttrType type = call_frame->attrs.types[0];
    void* attr = call_frame->attrs.attrs[0];
    if (type == XLA_FFI_AttrType_STRING) {
      auto* span = reinterpret_cast<XLA_FFI_ByteSpan*>(attr);
      *data = span->ptr;
      *len = span->len;
      return true;
    }
    if (type == XLA_FFI_AttrType_ARRAY) {
      auto* arr = reinterpret_cast<XLA_FFI_Array*>(attr);
      if (arr->dtype != XLA_FFI_DataType_U8) return false;
      *data = reinterpret_cast<const char*>(arr->data);
      *len = arr->size;
      return true;
    }
  }

  return false;
}

static XLA_FFI_Error* gemmul8_ffi_impl(XLA_FFI_CallFrame* call_frame,
                                       XLA_FFI_DataType dtype) {
  const XLA_FFI_Api* api = call_frame->api;
  if (call_frame->stage != XLA_FFI_ExecutionStage_EXECUTE) {
    return nullptr;
  }

  if (call_frame->args.size != 2 || call_frame->rets.size != 1) {
    return make_error(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "gemmul8 expects 2 args and 1 result");
  }

  if (call_frame->args.types[0] != XLA_FFI_ArgType_BUFFER ||
      call_frame->args.types[1] != XLA_FFI_ArgType_BUFFER ||
      call_frame->rets.types[0] != XLA_FFI_RetType_BUFFER) {
    return make_error(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "gemmul8 expects buffer args/results");
  }

  auto* A_buf = reinterpret_cast<XLA_FFI_Buffer*>(call_frame->args.args[0]);
  auto* B_buf = reinterpret_cast<XLA_FFI_Buffer*>(call_frame->args.args[1]);
  auto* C_buf = reinterpret_cast<XLA_FFI_Buffer*>(call_frame->rets.rets[0]);

  if (A_buf->dtype != dtype || B_buf->dtype != dtype || C_buf->dtype != dtype) {
    return make_error(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "gemmul8 dtype mismatch");
  }
  if (A_buf->rank != 2 || B_buf->rank != 2 || C_buf->rank != 2) {
    return make_error(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "gemmul8 expects rank-2 buffers");
  }

  const char* opaque_data = nullptr;
  size_t opaque_len = 0;
  if (!get_opaque(call_frame, &opaque_data, &opaque_len) ||
      opaque_len != sizeof(Gemmul8Descriptor)) {
    return make_error(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "gemmul8 missing/invalid opaque descriptor");
  }

  const auto* desc = reinterpret_cast<const Gemmul8Descriptor*>(opaque_data);

  XLA_FFI_Stream_Get_Args sargs;
  sargs.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  sargs.extension_start = nullptr;
  sargs.ctx = call_frame->ctx;
  sargs.stream = nullptr;
  if (XLA_FFI_Error* err = api->XLA_FFI_Stream_Get(&sargs)) {
    return err;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(sargs.stream);

  cublasHandle_t handle = get_handle(stream);
  if (!handle) {
    return make_error(api, XLA_FFI_Error_Code_INTERNAL,
                      "failed to create cublas handle");
  }

  const unsigned num_moduli = desc->num_moduli < 2 ? 2u : desc->num_moduli;
  const bool fastmode = desc->fastmode != 0;

  // JAX buffers are row-major. GEMMul8 expects column-major.
  // Compute C^T = B^T * A^T using column-major to match row-major C.
  const int64_t m = desc->m;
  const int64_t n = desc->n;
  const int64_t k = desc->k;

  const int64_t m_col = n;
  const int64_t n_col = m;
  const int64_t k_col = k;
  const size_t lda = static_cast<size_t>(n);  // B has shape (k, n) row-major
  const size_t ldb = static_cast<size_t>(k);  // A has shape (m, k) row-major
  const size_t ldc = static_cast<size_t>(n);  // C has shape (m, n) row-major

  size_t worksize = gemmul8::workSize<false>(
      static_cast<size_t>(m_col),
      static_cast<size_t>(n_col),
      static_cast<size_t>(k_col),
      num_moduli);

  void* work = nullptr;
  if (worksize > 0) {
    XLA_FFI_DeviceMemory_Allocate_Args aargs;
    aargs.struct_size = XLA_FFI_DeviceMemory_Allocate_Args_STRUCT_SIZE;
    aargs.extension_start = nullptr;
    aargs.ctx = call_frame->ctx;
    aargs.size = worksize;
    aargs.alignment = 256;
    aargs.data = nullptr;
    if (XLA_FFI_Error* err = api->XLA_FFI_DeviceMemory_Allocate(&aargs)) {
      return err;
    }
    work = aargs.data;
  }

  if (dtype == XLA_FFI_DataType_F32) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float* A = reinterpret_cast<const float*>(A_buf->data);
    const float* B = reinterpret_cast<const float*>(B_buf->data);
    float* C = reinterpret_cast<float*>(C_buf->data);

    (void)gemmul8::gemm<float>(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<size_t>(m_col),
        static_cast<size_t>(n_col),
        static_cast<size_t>(k_col),
        &alpha,
        B,
        lda,
        A,
        ldb,
        &beta,
        C,
        ldc,
        num_moduli,
        fastmode,
        work,
        nullptr,
        nullptr,
        false,
        false,
        false,
        false);
  } else if (dtype == XLA_FFI_DataType_F64) {
    const double alpha = 1.0;
    const double beta = 0.0;
    const double* A = reinterpret_cast<const double*>(A_buf->data);
    const double* B = reinterpret_cast<const double*>(B_buf->data);
    double* C = reinterpret_cast<double*>(C_buf->data);

    (void)gemmul8::gemm<double>(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<size_t>(m_col),
        static_cast<size_t>(n_col),
        static_cast<size_t>(k_col),
        &alpha,
        B,
        lda,
        A,
        ldb,
        &beta,
        C,
        ldc,
        num_moduli,
        fastmode,
        work,
        nullptr,
        nullptr,
        false,
        false,
        false,
        false);
  } else {
    if (work) {
      XLA_FFI_DeviceMemory_Free_Args fargs;
      fargs.struct_size = XLA_FFI_DeviceMemory_Free_Args_STRUCT_SIZE;
      fargs.extension_start = nullptr;
      fargs.ctx = call_frame->ctx;
      fargs.size = worksize;
      fargs.data = work;
      api->XLA_FFI_DeviceMemory_Free(&fargs);
    }
    return make_error(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "unsupported dtype");
  }

  if (work) {
    XLA_FFI_DeviceMemory_Free_Args fargs;
    fargs.struct_size = XLA_FFI_DeviceMemory_Free_Args_STRUCT_SIZE;
    fargs.extension_start = nullptr;
    fargs.ctx = call_frame->ctx;
    fargs.size = worksize;
    fargs.data = work;
    api->XLA_FFI_DeviceMemory_Free(&fargs);
  }

  return nullptr;
}

XLA_FFI_Error* gemmul8_f32_ffi(XLA_FFI_CallFrame* call_frame) {
  return gemmul8_ffi_impl(call_frame, XLA_FFI_DataType_F32);
}

XLA_FFI_Error* gemmul8_f64_ffi(XLA_FFI_CallFrame* call_frame) {
  return gemmul8_ffi_impl(call_frame, XLA_FFI_DataType_F64);
}

} // extern "C"
