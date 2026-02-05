import ctypes
import os
import struct
from typing import Optional

import jax
import jax.numpy as jnp
from jax import core
from jax import ffi
from jax.interpreters import mlir
from jaxlib import xla_client
try:
    from jaxlib.hlo_helpers import custom_call
except Exception:
    try:
        from jax._src.interpreters.mlir import custom_call  # type: ignore
    except Exception as exc:
        raise ImportError(
            "custom_call helper not found. Try upgrading jax/jaxlib or report this error."
        ) from exc


# ---- Custom call registration ----
_registered = False


def register_gemmul8_custom_call(lib_path: Optional[str] = None) -> None:
    global _registered
    if _registered:
        return

    if lib_path is None:
        lib_path = os.environ.get(
            "GEMMUL8_JAX_LIB",
            os.path.join(os.path.dirname(__file__), "libgemmul8_jax.so"),
        )

    lib = ctypes.CDLL(lib_path)

    f32 = lib.gemmul8_f32
    f64 = lib.gemmul8_f64

    xla_client.register_custom_call_target(
        b"gemmul8_f32", ffi.pycapsule(f32), platform="gpu"
    )
    xla_client.register_custom_call_target(
        b"gemmul8_f64", ffi.pycapsule(f64), platform="gpu"
    )

    _registered = True


# ---- Primitive definition ----

gemmul8_p = core.Primitive("gemmul8")


def _gemmul8_abstract_eval(a, b, *, num_moduli, fastmode):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("gemmul8 expects 2D matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError("gemmul8 shape mismatch: A.shape[1] must equal B.shape[0]")
    if a.dtype != b.dtype:
        raise ValueError("gemmul8 expects A and B to have the same dtype")
    if a.dtype not in (jnp.float32, jnp.float64):
        raise ValueError("gemmul8 supports float32 and float64 only")
    out_shape = (a.shape[0], b.shape[1])
    return core.ShapedArray(out_shape, a.dtype)


gemmul8_p.def_abstract_eval(_gemmul8_abstract_eval)


def _pack_descriptor(a_shape, b_shape, num_moduli, fastmode):
    m, k = a_shape
    k2, n = b_shape
    if k != k2:
        raise ValueError("gemmul8 shape mismatch")

    # Column-major layout: lda=m, ldb=k, ldc=m
    lda = m
    ldb = k
    ldc = m

    return struct.pack(
        "<qqqqqqII",
        int(m),
        int(n),
        int(k),
        int(lda),
        int(ldb),
        int(ldc),
        int(num_moduli),
        int(fastmode),
    )


def _gemmul8_lowering(ctx, a, b, *, num_moduli, fastmode):
    if not _registered:
        raise RuntimeError("gemmul8 custom call not registered; call register_gemmul8_custom_call() first")

    aval_out = ctx.avals_out[0]
    out_type = mlir.aval_to_ir_type(aval_out)

    opaque = _pack_descriptor(ctx.avals_in[0].shape, ctx.avals_in[1].shape, num_moduli, fastmode)

    if aval_out.dtype == jnp.float32:
        target = "gemmul8_f32"
    elif aval_out.dtype == jnp.float64:
        target = "gemmul8_f64"
    else:
        raise ValueError("Unsupported dtype for gemmul8")

    # Request column-major layout to match cuBLAS/GEMMul8 expectations.
    operand_layouts = [(0, 1), (0, 1)]
    result_layouts = [(0, 1)]

    call = custom_call(
        call_target_name=target,
        result_types=[out_type],
        operands=[a, b],
        backend_config=opaque,
        api_version=1,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
    )

    return call.results


def _gemmul8_cpu_lowering(ctx, a, b, *, num_moduli, fastmode):
    # CPU fallback for testing without GPU.
    def _ref(x, y):
        return x @ y

    return mlir.lower_fun(_ref, multiple_results=False)(ctx, a, b)


mlir.register_lowering(gemmul8_p, _gemmul8_lowering, platform="gpu")
mlir.register_lowering(gemmul8_p, _gemmul8_cpu_lowering, platform="cpu")


def gemmul8(a, b, *, num_moduli=8, fastmode=False):
    return gemmul8_p.bind(a, b, num_moduli=num_moduli, fastmode=fastmode)


def gemmul8_jit(a, b, *, num_moduli=8, fastmode=False):
    return jax.jit(lambda x, y: gemmul8(x, y, num_moduli=num_moduli, fastmode=fastmode))(a, b)
