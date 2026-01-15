# BigIntFixedPoint

`BigIntFixedPoint` is a high-performance JAX library for arbitrary-precision integer and fixed-point arithmetic, accelerated by custom Triton kernels. It allows you to perform computations on integers much larger than 64 bits while maintaining full compatibility with JAX's transformation ecosystem (`jit`, `vmap`, etc.).

## 🚀 Key Features

- **Arbitrary Precision**: Define integers with any number of "limbs" (e.g., 256-bit, 512-bit, or more).
- **Triton Acceleration**: Custom kernels for Addition, Subtraction, Multiplication, and Division designed for GPU performance.
- **JAX Integration**: Fully compatible with `jax.jit`, `jax.vmap`, and JAX Pytrees.
- **Pythonic Operators**: Supports `+`, `-`, `*`, `//`, `%`, and `divmod()`.
- **Signed Arithmetic**: Built-in support for Two's complement signed integers.
- **Functional Updates**: Support for JAX-style `.at[...]` indexing and updates.

## 📦 Installation

You can install the package in editable mode:

```bash
pip install -e ./BigIntFixedPoint
```

*Note: Requires `jax`, `triton`, and `jax-triton` to be installed in your environment.*

## 🛠 Usage

### Basic Arithmetic

```python
import jax
import jax.numpy as jnp
from BigIntFixedPoint import limbs

# Create BigIntTensors with 8 limbs of 32 bits (256 bits total)
a = limbs(10**50, num_limbs=8)
b = limbs(10**40, num_limbs=8)

# Standard operators
c = a + b
d = a * b
q, r = divmod(a, b)

# Convert back to Python integers for inspection
print(f"Result: {c.numpy()}")
```

### Vectorization and JIT

```python
@jax.jit
def square_and_add(x, y):
    return (x * x) + y

# Vectorized creation
x = limbs([1, 2, 3], num_limbs=4)
y = limbs([10, 20, 30], num_limbs=4)

result = square_and_add(x, y)
print(result.numpy()) # [11, 24, 39]
```

### Functional Updates

```python
a = limbs([10, 20, 30], num_limbs=4)

# Update the second element (jax-style)
b = a.at[1].set(99)
print(b.numpy()) # [10, 99, 30]
```

### Fixed-Point Arithmetic

While `BigIntFixedPoint` primarily provides the integer backbone, you can easily perform fixed-point arithmetic by manually scaling your values. 

```python
SCALE = 10**18  # 18 decimal places

def to_fp(x): return limbs(int(x * SCALE), num_limbs=16)
def from_fp(x): return x.numpy() / SCALE

a = to_fp(1.23456789)
b = to_fp(2.0)

# Multiplication in fixed-point requires re-scaling
c_raw = a * b
c = c_raw // limbs(SCALE, num_limbs=16)

print(from_fp(c)) # 2.46913578
```

## 📐 How it Works

The library represents large integers as tensors of "limbs" (unsigned integers) stored in a JAX array. For example, a 256-bit integer is stored with a trailing dimension of 8 `uint32` values.

- **Addition**: Uses a parallel prefix-scan (Kogge-Stone style) carry propagation implemented in Triton.
- **Multiplication**: Implements full hardware-accelerated precision multiplication with correction for signed integers.
- **Division**: Implements an efficient bit-by-bit division kernel in Triton, yielding both quotient and remainder.

## ⚙️ Configuration

When creating tensors with `limbs()`, you can specify:
- `num_limbs`: The number of limbs (controls the maximum bit-width).
- `dtype`: The data type of each limb (e.g., `jnp.uint8`, `jnp.uint16`, or `jnp.uint32`). 

*Note: For performance on modern GPUs, `uint32` is generally recommended.*

## 📄 License

This project is licensed under the MIT License.
