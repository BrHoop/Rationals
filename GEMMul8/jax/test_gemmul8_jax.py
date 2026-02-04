import os

import jax
import jax.numpy as jnp

from gemmul8_jax import gemmul8, gemmul8_jit, register_gemmul8_custom_call


def main():
    lib_path = os.environ.get("GEMMUL8_JAX_LIB")
    if lib_path:
        print(f"Using GEMMul8 custom call lib: {lib_path}")
    else:
        print("Using default GEMMul8 custom call lib path")

    register_gemmul8_custom_call(lib_path)

    key = jax.random.PRNGKey(0)
    m, k, n = 128, 128, 128

    a = jax.random.normal(key, (m, k), dtype=jnp.float32)
    b = jax.random.normal(key, (k, n), dtype=jnp.float32)

    # JIT to force custom call execution on GPU.
    c = gemmul8_jit(a, b, num_moduli=8, fastmode=False)
    c_ref = a @ b

    # Transfer to host for comparison.
    c_host = jnp.array(c)
    c_ref_host = jnp.array(c_ref)

    rel_err = jnp.linalg.norm(c_host - c_ref_host) / jnp.linalg.norm(c_ref_host)
    print("relative error:", float(rel_err))

    # Simple sanity threshold (tune as needed based on num_moduli/fastmode).
    if rel_err > 1e-2:
        raise SystemExit(f"relative error too large: {rel_err}")


if __name__ == "__main__":
    main()
