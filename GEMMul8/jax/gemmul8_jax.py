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
    import jax.ffi as jffi  # JAX >= 0.4.33
    _HAS_JAX_FFI = hasattr(jffi, "ffi_call")
    _HAS_JAX_FFI_REGISTER = hasattr(jffi, "register_ffi_target")
except Exception:
    jffi = None
    _HAS_JAX_FFI = False
    _HAS_JAX_FFI_REGISTER = False

try:
    from jaxlib.hlo_helpers import custom_call
except Exception:
    try:
        from jax._src.interpreters.mlir import custom_call  # type: ignore
    except Exception:
        custom_call = None


# ---- Custom call registration ----
_registered = False
_registered_ffi = False


def register_gemmul8_custom_call(lib_path: Optional[str] = None) -> None:
    global _registered, _registered_ffi
    if _registered:
        return

    if lib_path is None:
        lib_path = os.environ.get(
            "GEMMUL8_JAX_LIB",
            os.path.join(os.path.dirname(__file__), "libgemmul8_jax.so"),
        )

    lib = ctypes.CDLL(lib_path)

    debug = os.environ.get("GEMMUL8_JAX_DEBUG") == "1"
    errors = []

    if _HAS_JAX_FFI and _HAS_JAX_FFI_REGISTER:
        if hasattr(lib, "gemmul8_f32_ffi") and hasattr(lib, "gemmul8_f64_ffi"):
            f32 = lib.gemmul8_f32_ffi
            f64 = lib.gemmul8_f64_ffi
            # Register on multiple platform aliases to cover JAX/JAXLIB variants.
            # Prefer a backend-matching registration if possible.
            backend = None
            backend_platform = None
            try:
                backend = jax.default_backend()
            except Exception:
                backend = None
            try:
                from jax.lib import xla_bridge as xb

                backend_platform = xb.get_backend().platform
            except Exception:
                backend_platform = None

            platforms = [None]
            if backend:
                platforms.append(backend)
                platforms.append(backend.upper())
            if backend_platform and backend_platform not in platforms:
                platforms.append(backend_platform)
                platforms.append(backend_platform.upper())
            platforms.extend(["CUDA", "cuda", "gpu"])

            success_backend = False
            for platform in platforms:
                try:
                    import inspect

                    sig = inspect.signature(jffi.register_ffi_target)
                    kwargs = {}
                    if platform is not None:
                        if "platform" in sig.parameters:
                            kwargs["platform"] = platform
                        elif "backend" in sig.parameters:
                            kwargs["backend"] = platform
                    jffi.register_ffi_target(
                        "gemmul8_f32", ffi.pycapsule(f32), **kwargs
                    )
                    jffi.register_ffi_target(
                        "gemmul8_f64", ffi.pycapsule(f64), **kwargs
                    )
                    if platform is None:
                        success_backend = True
                    if backend and platform in (backend, backend.upper()):
                        success_backend = True
                    if backend_platform and platform in (
                        backend_platform,
                        backend_platform.upper(),
                    ):
                        success_backend = True
                except Exception as exc:
                    errors.append(f"{platform}: {exc}")

            _registered_ffi = success_backend
        else:
            _registered_ffi = False
    else:
        f32 = lib.gemmul8_f32
        f64 = lib.gemmul8_f64
        xla_client.register_custom_call_target(
            b"gemmul8_f32", ffi.pycapsule(f32), platform="gpu"
        )
        xla_client.register_custom_call_target(
            b"gemmul8_f64", ffi.pycapsule(f64), platform="gpu"
        )

    _registered = True

    if _HAS_JAX_FFI and _HAS_JAX_FFI_REGISTER and not _registered_ffi:
        if debug and errors:
            print("GEMMul8 FFI registration failed:")
            for err in errors:
                print("  -", err)
        raise RuntimeError(
            "GEMMul8 FFI registration failed for the active backend. "
            "Set GEMMUL8_JAX_DEBUG=1 for details."
        )


# ---- Primitive definition ----

gemmul8_p = core.CallPrimitive("gemmul8")


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

    if custom_call is None:
        raise RuntimeError("custom_call helper not available in this JAX version")

    # Different JAX/JAXLIB versions expose slightly different custom_call helpers.
    # Try the most recent signature first, then fall back to simpler forms.
    try:
        call = custom_call(
            call_target_name=target,
            result_types=[out_type],
            operands=[a, b],
            backend_config=opaque,
            api_version=1,
            operand_layouts=operand_layouts,
            result_layouts=result_layouts,
        )
    except Exception:
        try:
            call = custom_call(
                call_target_name=target,
                result_types=[out_type],
                operands=[a, b],
                backend_config=opaque,
            )
        except Exception:
            # Oldest fallback: positional helper.
            call = custom_call(target, [out_type], [a, b], opaque)

    return call.results


def _gemmul8_cpu_lowering(ctx, a, b, *, num_moduli, fastmode):
    # CPU fallback for testing without GPU.
    def _ref(x, y):
        return x @ y

    return mlir.lower_fun(_ref, multiple_results=False)(ctx, a, b)


mlir.register_lowering(gemmul8_p, _gemmul8_lowering, platform="gpu")
mlir.register_lowering(gemmul8_p, _gemmul8_cpu_lowering, platform="cpu")


def gemmul8(a, b, *, num_moduli=8, fastmode=False):
    if _HAS_JAX_FFI and _registered_ffi:
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("gemmul8 expects 2D matrices")
        if a.shape[1] != b.shape[0]:
            raise ValueError("gemmul8 shape mismatch: A.shape[1] must equal B.shape[0]")
        if a.dtype != b.dtype:
            raise ValueError("gemmul8 expects A and B to have the same dtype")
        if a.dtype not in (jnp.float32, jnp.float64):
            raise ValueError("gemmul8 supports float32 and float64 only")

        target = "gemmul8_f32" if a.dtype == jnp.float32 else "gemmul8_f64"
        opaque = _pack_descriptor(a.shape, b.shape, num_moduli, fastmode)
        out_shape = (a.shape[0], b.shape[1])

        # JAX 0.9+ ffi_call signature changed; detect supported kwargs.
        import inspect

        sig = inspect.signature(jffi.ffi_call)
        kwargs = {}
        if "backend_config" in sig.parameters:
            kwargs["backend_config"] = opaque
        elif "attrs" in sig.parameters:
            kwargs["attrs"] = {"opaque": opaque}
        else:
            # Best-effort: pass opaque as a named attr if **kwargs accepted.
            if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                kwargs["opaque"] = opaque

        result = jffi.ffi_call(
            target,
            result_shape_dtypes=[jax.ShapeDtypeStruct(out_shape, a.dtype)],
            **kwargs,
        )(a, b)
        return result

    return gemmul8_p.bind(a, b, num_moduli=num_moduli, fastmode=fastmode)


def gemmul8_jit(a, b, *, num_moduli=8, fastmode=False):
    return jax.jit(lambda x, y: gemmul8(x, y, num_moduli=num_moduli, fastmode=fastmode))(a, b)
