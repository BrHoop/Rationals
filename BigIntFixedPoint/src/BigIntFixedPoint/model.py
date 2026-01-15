import builtins
from typing import Tuple
import numpy as np
import jax.numpy as jnp
import jax.tree_util
from .kernels import add_jit, sub_jit, mul_jit, divmod_jit
# Ensure 'divmod' is the python builtin, not the JAX function from previous steps.
divmod = builtins.divmod


class BigIntTensor:
    def __init__(self, tensor: jnp.ndarray):
        """
        Initialize with a JAX array of shape [..., L].
        Dimension -1 is treated as the limbs (LSB at index 0).
        """
        self.tensor = tensor
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        self.L = tensor.shape[-1]

    def __repr__(self):
        return f"BigIntTensor(shape={self.shape}, limbs={self.L}, dtype={self.dtype})"

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
        a, b = self._pad_to_match(other)
        return BigIntTensor(add_jit(a, b))

    def __sub__(self, other):
        if not isinstance(other, BigIntTensor): return NotImplemented
        a, b = self._pad_to_match(other)
        return BigIntTensor(sub_jit(a, b))

    def __mul__(self, other):
        if not isinstance(other, BigIntTensor): return NotImplemented
        a, b = self._pad_to_match(other)
        return BigIntTensor(mul_jit(a, b))

    def __divmod__(self, other):
        if not isinstance(other, BigIntTensor): return NotImplemented
        a, b = self._pad_to_match(other)
        q, r = divmod_jit(a, b)
        return BigIntTensor(q), BigIntTensor(r)

    def __floordiv__(self, other):
        q, _ = divmod(self, other)
        return q

    def __mod__(self, other):
        _, r = divmod(self, other)
        return r

    def __neg__(self):
        # 0 - self
        return BigIntTensor(sub_jit(jnp.zeros_like(self.tensor), self.tensor))

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

        return BigIntTensor(ret)

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