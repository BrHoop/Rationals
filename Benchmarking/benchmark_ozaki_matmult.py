import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import time

sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
times = []

for size in sizes:
    A = jnp.random.randn(size, size).astype(jnp.float32)
    B = jnp.random.randn(size, size).astype(jnp.float32)
    t0 = time.time()
    C = A @ B 
    t1 = time.time()
    times.append(t1 - t0)


plt.plot(sizes, times)
plt.xlabel("Matrix Size")
plt.ylabel("Time (s)")
plt.title("Matrix Multiplication Benchmark")
plt.show()

