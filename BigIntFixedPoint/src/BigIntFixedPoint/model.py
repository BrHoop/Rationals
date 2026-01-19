import builtins
from typing import Tuple
import numpy as np
import jax.numpy as jnp
import jax.tree_util
from .kernels import add_jit, sub_jit, mul_jit, divmod_jit
# Ensure 'divmod' is the python builtin, not the JAX function from previous steps.
divmod = builtins.divmod


class BigIntTensor:
    def __init__(self, tensor: jnp.ndarray, frac_bits: int = 0):
        """
        Initialize with a JAX array of shape [..., L].
        Dimension -1 is treated as the limbs (LSB at index 0).
        frac_bits: Number of bits representing the fractional part.
        """
        self.tensor = tensor
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        self.L = tensor.shape[-1]
        self.frac_bits = frac_bits

    def __repr__(self):
        return f"BigIntTensor(shape={self.shape}, limbs={self.L}, dtype={self.dtype}, frac_bits={self.frac_bits})"

    def _pad_to_match(self, other: 'BigIntTensor') -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Aligns L dimension by zero-padding the shorter tensor.
        JAX broadcasting handles the batch dimensions.
        """
        if self.dtype != other.dtype:
            raise TypeError(f"Dtype mismatch: {self.dtype} vs {other.dtype}")

        L_self = self.L
        L_other = other.L

        if L_self == L_other:
            return self.tensor, other.tensor

        target_L = max(L_self, L_other)

        def pad_tensor(t, curr_L, tgt_L):
            # Pad along the last axis (limbs).
            # We assume index 0 is LSB, so we append zeros to the end (MSB side).
            padding = [(0, 0)] * (t.ndim - 1) + [(0, tgt_L - curr_L)]
            return jnp.pad(t, padding, mode='constant')

        t_self = pad_tensor(self.tensor, L_self, target_L)
        t_other = pad_tensor(other.tensor, L_other, target_L)
        return t_self, t_other

    def __add__(self, other):
        if not isinstance(other, BigIntTensor): return NotImplemented
        if self.frac_bits != other.frac_bits:
            raise ValueError(f"frac_bits mismatch: {self.frac_bits} vs {other.frac_bits}")
        a, b = self._pad_to_match(other)
        return BigIntTensor(add_jit(a, b), frac_bits=self.frac_bits)

    def __sub__(self, other):
        if not isinstance(other, BigIntTensor): return NotImplemented
        if self.frac_bits != other.frac_bits:
            raise ValueError(f"frac_bits mismatch: {self.frac_bits} vs {other.frac_bits}")
        a, b = self._pad_to_match(other)
        return BigIntTensor(sub_jit(a, b), frac_bits=self.frac_bits)

    def __mul__(self, other):
        if not isinstance(other, BigIntTensor): return NotImplemented
        a, b = self._pad_to_match(other)
        
        # Raw multiplication (returns 2*L limbs)
        res = mul_jit(a, b)
        
        # Calculate new frac_bits
        # Standard fixed point: A(f1) * B(f2) -> C(f1+f2)
        total_frac = self.frac_bits + other.frac_bits
        
        # Determine target frac bits. 
        # If both are fixed point (frac_bits > 0), we likely want to maintain the specific precision
        # implied by the factory settings (e.g. all tensors sharing same scale).
        # We assume the target is `max(self.frac_bits, other.frac_bits)`.
        target_frac = max(self.frac_bits, other.frac_bits)
        
        if total_frac == 0:
            return BigIntTensor(res, frac_bits=0)
            
        # If we need to shift back
        shift_amount = total_frac - target_frac
        
        if shift_amount > 0:
            # Shift right by shift_amount.
            # We assume shift_amount is small enough to generally fit (it definitely fits in the expanded 2L).
            # We implementation shift via division: res_shifted = res // (1 << shift_amount)
            
            # Construct divisor (1 << shift_amount) as a BigIntTensor
            # We need to construct it compatible with 'res' which has shape [..., 2L]
            # but divmod expects [..., K] matching or broadcastable.
            
            # Create a scalar integer tensor representing the divisor
            # We use a helper utility. Since we can't import factory, we do it manually or use jax keys
            # Actually, we can use the `kernels.convert_int_vec` if we import it, or just implement specific logic.
            # But simpler: create a numpy array of limbs and convert to jax.
            
            # Find number of limbs needed for the divisor. 2L should be enough.
            current_L = res.shape[-1]
            limb_bits = self.dtype.itemsize * 8
            
            # Create divisor limbs
            divisor_val = 1 << shift_amount
            
            # Convert divisor_val to limbs
            # We can use a small local helper
            def val_to_limbs(v, n_limbs, dtype):
                mask = (1 << limb_bits) - 1
                out = []
                for _ in range(n_limbs):
                    out.append(v & mask)
                    v >>= limb_bits
                return np.array(out, dtype=dtype)
                
            div_arr = val_to_limbs(divisor_val, current_L, self.dtype)
            div_tensor = jnp.array(div_arr) # Shape (2L,)
            
            # Expand dims to match batch if necessary? 
            # divmod_jit broadcasting should handle (2L,) vs (N, 2L)
            
            # Perform division
            # res is tensor, div_tensor is tensor
            # divmod_jit expects arrays
            q, _ = divmod_jit(res, div_tensor)
            
            # Truncate back to L?
            # Usually fixed point keeps L same as inputs if intended.
            # mul_jit expanded to 2L. 
            # If we want to return a result compatible with inputs, we should slice to L.
            # But if inputs had different L, we padded to max(L).
            output_L = max(self.L, other.L)
            
            # Take the lower output_L limbs of the quotient
            q_trunc = q[..., :output_L]
            
            return BigIntTensor(q_trunc, frac_bits=target_frac)
        else:
            # No shift needed (e.g. target is sum of fracs? Unlikely for fixed point).
            # If shift_amount == 0, we just return res (2L) or truncate?
            # If we don't truncate, the tensor grows with every mul. 
            # For fixed point, we usually truncate.
            output_L = max(self.L, other.L)
            return BigIntTensor(res[..., :output_L], frac_bits=target_frac)

    def __divmod__(self, other):
        if not isinstance(other, BigIntTensor): return NotImplemented
        a, b = self._pad_to_match(other)
        # TODO: Fixed point division logic if needed. 
        # For now, treat as integer division on the underlying representation.
        # Note: Fixed point division (A/B) usually requires pre-shifting A: (A << f) / B.
        
        if self.frac_bits > 0 or other.frac_bits > 0:
             # If fixed point, we assume (A * 2^f) / (B * 2^f) = A/B (scalar). 
             # Result is integer? No, result should be fixed point implies we want A/B * 2^f.
             # So we must compute (A * 2^{2f}) / (B * 2^f) = (A/B) * 2^f.
             # So we need to shift A left by 'frac_bits'.
             pass # Leaving as simple integer div for now unless requested.
             
        q, r = divmod_jit(a, b)
        return BigIntTensor(q, frac_bits=0), BigIntTensor(r, frac_bits=0) 
        # Result of integer div is integer (frac=0)? Or does it preserve scale?
        # Let's keep it simple: Raw divmod returns integer result.

    def __floordiv__(self, other):
        q, _ = divmod(self, other)
        return q

    def __mod__(self, other):
        _, r = divmod(self, other)
        return r

    def __neg__(self):
        # 0 - self
        return BigIntTensor(sub_jit(jnp.zeros_like(self.tensor), self.tensor), frac_bits=self.frac_bits)

    # --- Indexing and Splicing ---

    def __getitem__(self, key):
        """
        Standard indexing/slicing.
        Returns a new BigIntTensor viewing the sliced data.
        """
        ret = self.tensor[key]

        # Safety check: ensure we didn't slice away the limb dimension.
        # If ndim is 0, it means we accessed a single limb, breaking the abstraction.
        # (A scalar BigInt still has ndim=1 due to the limb axis)
        if ret.ndim == 0:
            raise IndexError("Slicing reduced rank to 0. Cannot access internal limbs directly.")

        return BigIntTensor(ret, frac_bits=self.frac_bits)

    @property
    def at(self):
        """JAX-style .at[...] accessor for functional updates."""
        return _BigIntAtAccessor(self)

    def numpy(self):
        """Convert back to Python integers (host side) for inspection. Interprets as Signed."""
        arr = jax.device_get(self.tensor)
        dtype = self.dtype
        limb_bits = dtype.itemsize * 8
        total_bits = self.L * limb_bits

        # Helper to convert a single vector
        def vec_to_int(v):
            x = 0
            for limb in reversed(v):
                x = (x << limb_bits) | int(limb)

            # Check MSB for sign (Two's complement)
            if (x >> (total_bits - 1)) & 1:
                x -= (1 << total_bits)
            return x

        # Flatten batch dims to iterate
        flat = arr.reshape(-1, arr.shape[-1])
        ints = [vec_to_int(row) for row in flat]
        
        # Apply Fixed Point scaling
        if self.frac_bits > 0:
            scale = 2.0 ** self.frac_bits
            ints = [x / scale for x in ints]

        # Restore batch shape (excluding limb dim)
        if len(self.shape) > 1:
            return np.array(ints, dtype=object).reshape(self.shape[:-1]).tolist()
        return ints[0]

# --- JAX Pytree Registration ---
# This allows BigIntTensor to be passed into jax.jit, jax.vmap, etc.

def _flatten_bigint(obj):
    # Return (children, aux_data)
    # Children must be JAX arrays or other Pytrees.
    return (obj.tensor,), None

def _unflatten_bigint(aux, children):
    return BigIntTensor(children[0])

jax.tree_util.register_pytree_node(BigIntTensor, _flatten_bigint, _unflatten_bigint)


# --- Helper Classes for .at[...] ---

class _BigIntAtAccessor:
    def __init__(self, parent):
        self.parent = parent
    def __getitem__(self, key):
        return _BigIntAtItem(self.parent, key)

class _BigIntAtItem:
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key

    def set(self, value):
        # 1. Convert value to BigIntTensor if it's a raw integer/list
        if isinstance(value, (int, list, np.ndarray, jnp.ndarray)):
            # Heuristic: convert using parent's config
            value = limbs(value, num_limbs=self.parent.L, dtype=self.parent.dtype)

        if not isinstance(value, BigIntTensor):
            raise TypeError(f"Cannot set value of type {type(value)}")

        # 2. Align value's limbs to parent's limbs
        # Since 'set' implies fitting into existing storage shape, we pad/truncate value.
        tgt_L = self.parent.L
        v_t = value.tensor
        curr_L = v_t.shape[-1]

        if curr_L < tgt_L:
            # Pad MSB side (index end)
            pad_w = [(0,0)] * (v_t.ndim - 1) + [(0, tgt_L - curr_L)]
            v_t = jnp.pad(v_t, pad_width=pad_w)
        elif curr_L > tgt_L:
            # Truncate MSB side (keep LSBs at index 0)
            v_t = v_t[..., :tgt_L]

        # 3. Apply Update via JAX
        new_tensor = self.parent.tensor.at[self.key].set(v_t)
        return BigIntTensor(new_tensor)