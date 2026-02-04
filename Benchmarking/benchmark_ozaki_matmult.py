import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import time

sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
reps = 10
times = []

for size in sizes:
    t = 0
    for rep in range(reps): 
            A = jax.random.normal(jax.random.PRNGKey(0), (size, size)).astype(jnp.float32)
            B = jax.random.normal(jax.random.PRNGKey(1), (size, size)).astype(jnp.float32)
            t0 = time.time()
            C = A @ B 
            t1 = time.time()
            temp = t1 - t0 
            t+=temp
    times.append(t/reps)


plt.plot(sizes, times)
plt.xlabel("Matrix Size")
plt.ylabel("Time (s)")
plt.title("Matrix Multiplication Benchmark")
plt.show()

