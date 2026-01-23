# Standard imports
import jax 
import jax.numpy as jnp
from jax import profiler 
import numpy as np
import matplotlib.pyplot as plt

# Scheme imports
from BigIntTensor import limbs 


def addition(x, y):
    return x + y

def subtraction(x, y):
    return x - y

def multiplication(x, y):
    return x * y

def division(x, y):
    return x / y


    
def main():
    
    
    # Warm-up everything
    addition(limbs(jnp.array([1, 2, 3])), limbs(jnp.array([4, 5, 6])))
    subtraction(limbs(jnp.array([1, 2, 3])), limbs(jnp.array([4, 5, 6])))
    multiplication(limbs(jnp.array([1, 2, 3])), limbs(jnp.array([4, 5, 6])))
    division(limbs(jnp.array([1, 2, 3])), limbs(jnp.array([4, 5, 6])))

    # Actual benchmarking
    print("Starting benchmarking")
    print("Vector Addition")
    times = []
    time = []
    profiler.start_trace('trace')
    for i in range(2,1000):
        for j in range(100):
            time.append(jax.timeit(lambda: addition(limbs(jnp.random.randint(0, 1000, i),dtype=jnp.uint32,limbs=8), limbs(jnp.random.randint(0, 1000, i),dtype=jnp.uint32,limbs=8))))
        times.append(np.mean(time))
        time = []
    profiler.stop_trace()
    plt.plot(range(2,1000), times)
    plt.savefig('vec_addition.png')
    print("Vector Addition Done")
