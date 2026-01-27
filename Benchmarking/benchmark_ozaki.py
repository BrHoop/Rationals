
import jax
import jax.numpy as jnp
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Ozaki_Scheme_1.ozaki_v1 import ozaki_matmul_v1
from Ozaki_Scheme_2.ozaki_v2 import ozaki_scheme_2_solve

# Enable x64
jax.config.update("jax_enable_x64", True)


def _fd_weights(x0, x, deriv):
    n = len(x)
    m = deriv
    c = np.zeros((n, m + 1), dtype=np.float64)
    c1 = 1.0
    c4 = x[0] - x0
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - x0
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = (c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k])) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c[:, m]

def generate_fd_stencil_coeffs(order: int):
    """
    Generates centered finite difference coefficients for the 1st derivative.
    For a centered stencil of size 2p+1, the accuracy order is 2p.
    """
    if order % 2 != 0 or order <= 0:
        raise ValueError("Order must be a positive even integer.")
    p = order // 2
    x = np.arange(-p, p + 1, dtype=np.float64)
    coeffs = _fd_weights(0.0, x, 1)
    return jnp.array(coeffs)

def construct_fd_matrix(N, coeffs):
    """
    Constructs the Finite Difference Matrix D.
    D is sparse/banded, but for Ozaki MatMul test we treat it as dense or just pass it as matrix.
    """
    # Create valid dense matrix for D @ u
    # D is (N, N)
    # This is inefficient for storage but required to test Matrix Multiplication Schemes.
    
    # Use jax.scipy.linalg.toeplitz or just diag construction
    # We'll construct manually
    
    # Central band of coeffs
    # We need to handle boundaries. Periodicity is easiest.
    center = len(coeffs) // 2
    
    # Build a circulant matrix for periodic boundaries to keep it simple and dense-like structure
    expanded = np.zeros(N, dtype=np.float64)
    for i, c in enumerate(coeffs):
        offset = i - center
        expanded[offset % N] = float(c)
    
    # Construct circulant matrix without SciPy
    D_dense = np.stack([np.roll(expanded, i) for i in range(N)], axis=0)
    return jnp.array(D_dense)

def convolve_periodic(u, coeffs):
    n = u.shape[0]
    center = len(coeffs) // 2
    kernel = jnp.zeros(n, dtype=u.dtype)
    for i, c in enumerate(coeffs):
        kernel = kernel.at[(i - center) % n].set(c)
    return jnp.fft.ifft(jnp.fft.fft(u) * jnp.fft.fft(kernel)).real

def run_benchmark():
    orders = [2, 4, 8, 16] # Testing specific orders
    sizes = [256, 512, 1024, 2048, 4096, 8192] # Increasing sizes (bounded for dense matmul)
    
    results = {
        order: {
            'sizes': [],
            'jax_native': [],
            'ozaki_1': [],
            'ozaki_2': [],
        }
        for order in orders
    }
    accuracy = {
        order: {
            'sizes': [],
            'jax_native': [],
            'ozaki_1': [],
            'ozaki_2': [],
        }
        for order in orders
    }
    
    print("Starting Benchmark...")
    
    for N in sizes:
        print(f"\nSize N={N}")
        
        # Data
        x = jnp.linspace(0, 2*jnp.pi, N)
        u_true = jnp.sin(x)
        du_true = jnp.cos(x) # Analytical derivative
        
        # Loop orders
        for order in orders:
            print(f"  Order {order}...")
            coeffs = generate_fd_stencil_coeffs(order)
            dx = float(x[1] - x[0])
            coeffs = coeffs / dx
            # Guard against huge dense matrices
            if N * N > 200_000_000:
                print("  Skipping: dense matrix would be too large for this N.")
                continue
            D = construct_fd_matrix(N, coeffs)
            
            # 1. Baseline JAX (Fastest method - Conv/Slicing not Matmul)
            # We implemented D as a matrix for Ozaki, but Baseline should be fast.
            # Fast baseline: jnp.convolve
            t0 = time.time()
            # conv mode='same' handles boundaries differently than circulant mod, 
            # but for speed baseline let's use the matrix multiply baseline to be 'fair' to the operation tasks 
            # OR use actual conv.
            # User said: "tests the speed difference between a normal JAX float 64-bit finite difference... and Ozaki"
            # And "The float 64 method doesn't need to use matrix multiplication"
            # So I will use jnp.convolve for baseline speed.
            
            # Baseline: Convolution
            _ = convolve_periodic(u_true, coeffs)
            # Warmup
            
            t0 = time.time()
            res_jax = convolve_periodic(u_true, coeffs)
            jax.block_until_ready(res_jax)
            t_jax = time.time() - t0
            
            # 2. Ozaki 1 (Matmul)
            # D @ u
            # Warmup
            # Note: My ozaki impl expects matrix-matrix usually, but works for vector if shaped (N,1)
            u_col = u_true.reshape(-1, 1)
            _ = ozaki_matmul_v1(D, u_col)
            
            t0 = time.time()
            res_o1 = ozaki_matmul_v1(D, u_col)
            # jax.block_until_ready(res_o1) # ozaki returns jax array
            res_o1.block_until_ready()
            t_o1 = time.time() - t0
            
            # 3. Ozaki 2 (Matmul)
            # Warmup
            _ = ozaki_scheme_2_solve(D, u_col)
            
            t0 = time.time()
            res_o2 = ozaki_scheme_2_solve(D, u_col)
            res_o2.block_until_ready()
            t_o2 = time.time() - t0
            
            # Record
            results[order]['jax_native'].append(t_jax)
            results[order]['ozaki_1'].append(t_o1)
            results[order]['ozaki_2'].append(t_o2)
            results[order]['sizes'].append(N)
            
            # Accuracy
            err_jax = jnp.linalg.norm(res_jax - du_true) / jnp.linalg.norm(du_true)
            err_o1 = jnp.linalg.norm(res_o1.flatten() - du_true) / jnp.linalg.norm(du_true)
            err_o2 = jnp.linalg.norm(res_o2.flatten() - du_true) / jnp.linalg.norm(du_true)
            accuracy[order]['jax_native'].append(float(err_jax))
            accuracy[order]['ozaki_1'].append(float(err_o1))
            accuracy[order]['ozaki_2'].append(float(err_o2))
            accuracy[order]['sizes'].append(N)
            
    # Modify data to be JSON serializable or just plot directly
    plot_results(results)
    plot_accuracy(accuracy)

def plot_results(results):
    for order, data in results.items():
        if not data['sizes']:
            print(f"No data collected for order {order}; skipping plot.")
            continue
        plt.figure(figsize=(10, 6))
        plt.plot(data['sizes'], data['jax_native'], label='JAX Baseline (Conv)', marker='o')
        plt.plot(data['sizes'], data['ozaki_1'], label='Ozaki Scheme 1 (MatMul)', marker='x')
        plt.plot(data['sizes'], data['ozaki_2'], label='Ozaki Scheme 2 (Modular)', marker='s')
        
        plt.xlabel('Data Size N')
        plt.ylabel('Time (s)')
        plt.title(f'Finite Difference Efficiency (Order {order})')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        out_path = f'efficiency_benchmark_order_{order}.png'
        plt.savefig(out_path)
        plt.close()
        print(f"Benchmark saved to {out_path}")

def plot_accuracy(accuracy):
    for order, data in accuracy.items():
        if not data['sizes']:
            print(f"No accuracy data collected for order {order}; skipping plot.")
            continue
        plt.figure(figsize=(10, 6))
        plt.plot(data['sizes'], data['jax_native'], label='JAX Baseline (Conv)', marker='o')
        plt.plot(data['sizes'], data['ozaki_1'], label='Ozaki Scheme 1 (MatMul)', marker='x')
        plt.plot(data['sizes'], data['ozaki_2'], label='Ozaki Scheme 2 (Modular)', marker='s')
        
        plt.xlabel('Data Size N')
        plt.ylabel('Relative L2 Error')
        plt.title(f'Finite Difference Accuracy (Order {order})')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        out_path = f'accuracy_benchmark_order_{order}.png'
        plt.savefig(out_path)
        plt.close()
        print(f"Accuracy plot saved to {out_path}")

if __name__ == "__main__":
    run_benchmark()
