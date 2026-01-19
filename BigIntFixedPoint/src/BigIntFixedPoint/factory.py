import jax
import jax.numpy as jnp
import numpy as np
import math

from .model import BigIntTensor

def limbs(data, num_limbs=None, dtype=jnp.uint32, ubound=None, lbound=None) -> BigIntTensor:
    """
    Factory function to create BigIntTensor from python integers or floats.
    Args:
        data: A single number, a list/array of numbers, or a nested list/array.
        num_limbs: Number of limbs to represent each integer. If None, inferred from bounds or defaults to 8.
        dtype: JAX dtype for the limbs (uint8, uint16, uint32).
        ubound: Upper bound for the values (used to determine integer bits).
        lbound: Lower bound (precision) for the values (used to determine fractional bits).
    """
    # Determine bits per limb
    dt_np = np.dtype(dtype)
    if dt_np == np.uint32:
        limb_bits = 32
    elif dt_np == np.uint16:
        limb_bits = 16
    elif dt_np == np.uint8:
        limb_bits = 8
    else:
        raise ValueError(f"Unsupported dtype {dtype}. Use uint8, uint16, or uint32.")

    frac_bits = 0
    
    # Calculate requirements from bounds
    if ubound is not None or lbound is not None:
        if ubound is None or lbound is None:
             raise ValueError("Both ubound and lbound must be provided if one is provided.")
        
        # Integer bits: needs to cover range [-ubound, ubound] approximately
        # log2(ubound) gives bits for magnitude. +1 for sign is handled in total.
        # If ubound=10, log2(10)=3.32 -> 4 bits. 
        i_bits = math.ceil(math.log2(float(ubound)))
        
        # Fractional bits: resolution = lbound. 
        # precision = 2^-f <= lbound  => -f <= log2(lbound) => f >= -log2(lbound)
        if lbound <= 0: raise ValueError("lbound must be positive (representing precision)")
        f_bits = math.ceil(-math.log2(float(lbound)))
        
        # Ensure non-negative
        i_bits = max(0, i_bits)
        f_bits = max(0, f_bits)
        frac_bits = f_bits
        
        total_bits = i_bits + f_bits + 1 # +1 for sign bit
        required_limbs = math.ceil(total_bits / limb_bits)
        
        if num_limbs is None:
            num_limbs = required_limbs
            
    if num_limbs is None:
        num_limbs = 8

    mask = (1 << limb_bits) - 1
    
    # Pre-calculate scale as python integer (arbitrary precision)
    scale_factor = 1 << frac_bits

    def int_to_limbs(x):
        # Scale and convert
        # Handle float input by scaling
        # Python's int(x) truncates towards zero. round(x) is better for fixed point.
        val = int(round(x * scale_factor))
        res = []
        for _ in range(num_limbs):
            res.append(val & mask)
            val >>= limb_bits
        return res

    # Handle scalar case
    if isinstance(data, (int, float, np.number)):
        arr = np.array(int_to_limbs(data), dtype=dt_np) # Shape (num_limbs,)
        return BigIntTensor(jnp.array(arr), frac_bits=frac_bits)

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

    return BigIntTensor(jnp.array(arr_out), frac_bits=frac_bits)