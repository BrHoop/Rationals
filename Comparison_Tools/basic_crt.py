import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

MODS = [127, 125, 123, 121, 119, 113]
CAPACITY = 127 * 125 * 123 * 121 * 119 * 113
HALF_CAPACITY = CAPACITY // 2

SIZES = [128, 256, 512, 1024, 2048, 4096]
BLOCK_SIZE = 16

def random_array(size):
    key = jax.random.PRNGKey(42)
    shape = (size, size)
    random_array = jax.random.uniform(key, shape, minval=-1.0, maxval=1.0, dtype=jnp.float64)
    return random_array

@jax.jit(static_argnums=(1,))
def fp64_to_crt(arr, s):
    nx = s #Assumes symetric
    bx = BLOCK_SIZE
    nblocks_x = nx // bx

    grid_reshaped = arr.reshape((nblocks_x, bx, nblocks_x, bx)).transpose((0, 2, 1, 3))

    max_per_block = jnp.max(jnp.abs(grid_reshaped), axis=(2, 3), keepdims=True)
    safe_max = jnp.where(max_per_block == 0, 1.0, max_per_block)
    exponents = jnp.floor(jnp.log2(HALF_CAPACITY / safe_max)).astype(jnp.int8)
    exponents = jnp.where(max_per_block == 0, 0, exponents)

    shifted_grid = jnp.ldexp(grid_reshaped, exponents.astype(jnp.int32))
    virtual_ints = jnp.round(shifted_grid).astype(jnp.int64)

    crt_arrays = []
    for m in MODS:
        rem = jnp.mod(virtual_ints, m).astype(jnp.int8)
        crt_arrays.append(rem)

    return crt_arrays, exponents

@jax.jit(static_argnums=(2,))
def crt_to_fp64(crt_arrays, exponents, s):
    virtual_int = jnp.zeros_like(crt_arrays[0], dtype=jnp.int64)
    M = CAPACITY

    for i in range(6):
        m = MODS[i]
        rem = crt_arrays[i].astype(jnp.int64)

        Mi = M // m

        yi = pow(int(Mi), -1, int(m))

        term = rem * (Mi * yi)
        virtual_int = (virtual_int + term) % M

    virtual_int = jnp.where(virtual_int > HALF_CAPACITY, 
                            virtual_int - M, 
                            virtual_int)
        
    float_significand = virtual_int.astype(jnp.float64)
    grid_blocked = jnp.ldexp(float_significand, -exponents.astype(jnp.int32))

    nb = s // BLOCK_SIZE
    bx = BLOCK_SIZE
    
    grid_flat = grid_blocked.transpose(0, 2, 1, 3).reshape(s, s)
    
    return grid_flat

def main():
    print(f"{'SIZE':<10} | {'COMPRESSION':<15} | {'MAX ERROR':<15} | {'STATUS'}")
    print("-" * 60)

    for i, s in enumerate(SIZES):
        arr = random_array(s)

        crt_arrays, exponents = fp64_to_crt(arr, s)
        recon_arr = crt_to_fp64(crt_arrays, exponents, s)

        # Size in Bytes
        size_orig = s * s * 8
        # Compressed: (6 bytes per pixel) + (1 byte per 16x16 block)
        n_blocks = (s // BLOCK_SIZE) ** 2
        size_comp = (s * s * 6) + n_blocks
        ratio = size_orig / size_comp
        
        max_err = jnp.max(jnp.abs(arr - recon_arr))
        
        status = "✅ PASS" if max_err < 1e-10 else "❌ FAIL"
        print(f"{s:<10} | {ratio:.2f}x            | {max_err:.2e}        | {status}")

if __name__ == "__main__":
    main()