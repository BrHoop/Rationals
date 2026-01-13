import jax
import jax.numpy as jnp
import math
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class BFPGrid:
    def __init__(self, height, width, num_fields=2, block_size=(8, 8), 
                 mantissa_bits=56, nghost=0, mantissas=None, exponents=None):
        self.phys_H, self.phys_W = height, width
        self.C, self.ng = num_fields, nghost
        self.bh, self.bw = block_size
        self.mantissa_bits = mantissa_bits
        self.target_bit = mantissa_bits - 1
        
        # Memory Alignment for Pallas tiling
        self.mem_H = math.ceil(height / self.bh) * self.bh
        self.mem_W = math.ceil(width / self.bw) * self.bw
        self.exp_h, self.exp_w = self.mem_H // self.bh, self.mem_W // self.bw

        if mantissas is None:
            self.mantissas = jnp.zeros((self.C, self.mem_H, self.mem_W), dtype=jnp.int64)
            self.exponents = jnp.zeros((self.C, self.exp_h, self.exp_w), dtype=jnp.int64)
        else:
            self.mantissas, self.exponents = mantissas, exponents

    def tree_flatten(self):
        return (self.mantissas, self.exponents), (self.phys_H, self.phys_W, self.C, 
                (self.bh, self.bw), self.mantissa_bits, self.ng)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data[:3], block_size=aux_data[3], mantissa_bits=aux_data[4], 
                   nghost=aux_data[5], mantissas=children[0], exponents=children[1])

    def replace(self, **kwargs):
        m = kwargs.get('mantissas', self.mantissas)
        e = kwargs.get('exponents', self.exponents)
        return self.tree_unflatten(self.tree_flatten()[1], (m, e))

    def to_device(self, float_data):
        """Quantizes Float64 input. Mirrors original I/O dimensions."""
        pad_h, pad_w = self.mem_H - self.phys_H, self.mem_W - self.phys_W
        padded = jnp.pad(float_data, ((0,0), (0, pad_h), (0, pad_w)))
        reshaped = padded.reshape(self.C, self.exp_h, self.bh, self.exp_w, self.bw)
        kings = jnp.maximum(jnp.max(jnp.abs(reshaped), axis=(2, 4)), 1e-40)
        exponents = jnp.floor(jnp.log2(kings)).astype(jnp.int64) - self.target_bit
        scales = jnp.power(2.0, -exponents[:, :, None, :, None].astype(jnp.float64))
        mant = (reshaped * scales).astype(jnp.int64).reshape(self.C, self.mem_H, self.mem_W)
        return self.replace(mantissas=mant, exponents=exponents)

    def from_device(self):
        """Dequantizes and trims padding. Result is bit-compatible with original float code."""
        reshaped = self.mantissas.reshape(self.C, self.exp_h, self.bh, self.exp_w, self.bw)
        scales = jnp.power(2.0, self.exponents[:, :, None, :, None].astype(jnp.float64))
        full = (reshaped.astype(jnp.float64) * scales).reshape(self.C, self.mem_H, self.mem_W)
        return full[:, :self.phys_H, :self.phys_W]

    @staticmethod
    def make_scalar(val, target=60):
        if val == 0: return (0, 0)
        e = math.floor(math.log2(abs(val))) - target
        return (int(val * (2**-e)), e)