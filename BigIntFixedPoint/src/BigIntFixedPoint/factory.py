import jax
import jax.numpy as jnp
import numpy as np

from .model import BigIntTensor

def limbs(data, num_limbs=8, dtype=jnp.uint32) -> BigIntTensor:
    """
    Factory function to create BigIntTensor from python integers.
    Args:
        data: A single integer, a list/array of integers, or a nested list/array.
        num_limbs: Number of limbs to represent each integer.
        dtype: JAX dtype for the limbs (uint8, uint16, uint32).
    """
    # Determine bits per limb
    dt_np = np.dtype(dtype)
    if dt_np == np.uint32:
        bits = 32
    elif dt_np == np.uint16:
        bits = 16
    elif dt_np == np.uint8:
        bits = 8
    else:
        raise ValueError(f"Unsupported dtype {dtype}. Use uint8, uint16, or uint32.")

    mask = (1 << bits) - 1

    def int_to_limbs(x):
        val = int(x)
        res = []
        for _ in range(num_limbs):
            res.append(val & mask)
            val >>= bits
        return res

    # Handle scalar case
    if isinstance(data, (int, np.integer)):
        arr = np.array(int_to_limbs(data), dtype=dt_np) # Shape (num_limbs,)
        return BigIntTensor(jnp.array(arr))

    # Handle array/list case
    # Use object dtype to handle potentially large python integers without truncation
    data_np = np.array(data, dtype=object)

    # Flatten, convert, then restore shape
    flat_data = data_np.ravel()
    converted_flat = [int_to_limbs(x) for x in flat_data]

    # New shape: original_shape + (num_limbs,)
    new_shape = data_np.shape + (num_limbs,)

    # Convert list of lists to contiguous numpy array
    arr_out = np.array(converted_flat, dtype=dt_np).reshape(new_shape)

    return BigIntTensor(jnp.array(arr_out))