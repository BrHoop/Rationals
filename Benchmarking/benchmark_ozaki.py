
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

#This is not working on H200 machine 
def _has_tensor_cores():
    try:
        dev = jax.devices()[0]
        platform = dev.platform
        kind = getattr(dev, "device_kind", "")
        if platform != "gpu":
            return False
        if "NVIDIA" not in kind.upper():
            return False
        from Ozaki_Scheme_1 import triton_kernel_v1 as tk1
        from Ozaki_Scheme_2 import triton_mod_matmul as tk2
        return tk1.triton_call is not None and tk2.triton_call is not None
    except Exception:
        return False


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

def apply_stencil_periodic(u, coeffs):
    center = len(coeffs) // 2
    out = jnp.zeros_like(u)
    for i, c in enumerate(coeffs):
        shift = center - i
        out = out + c * jnp.roll(u, shift)
    return out

def run_benchmark():
    orders = [2, 4, 8, 16] # Testing specific orders
    sizes = [256, 512, 1024, 2048, 4096, 8192] # Increasing sizes (bounded for dense matmul)
    num_moduli = int(os.environ.get("OZAKI_NUM_MODULI", "8"))
    
    results = {
        order: {
            'sizes': [],
            'jax_matmul': [],
            'jax_stencil': [],
            'ozaki_1': [],
            'ozaki_2': [],
        }
        for order in orders
    }
    accuracy = {
        order: {
            'sizes': [],
            'jax_matmul': [],
            'jax_stencil': [],
            'ozaki_1': [],
            'ozaki_2': [],
        }
        for order in orders
    }
    use_ozaki = True
    if not use_ozaki:
        print("Ozaki schemes disabled: Tensor Core-capable NVIDIA GPU + Triton required.")
    
    print("Starting Benchmark...")
    
    for N in sizes:
        print(f"\nSize N={N}")
        
        # Data
        x = jnp.linspace(0, 2*jnp.pi, N, endpoint=False)
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
            
            # 1a. Baseline JAX FP64 matmul (like-for-like with Ozaki matmul)
            u_col = u_true.reshape(-1, 1)
            _ = jnp.matmul(D, u_col)
            t0 = time.perf_counter()
            res_jax = jnp.matmul(D, u_col)
            res_jax.block_until_ready()
            t_jax = time.perf_counter() - t0

            # 1b. Baseline stencil application (algorithmic reference)
            _ = apply_stencil_periodic(u_true, coeffs)
            t0 = time.perf_counter()
            res_stencil = apply_stencil_periodic(u_true, coeffs)
            res_stencil.block_until_ready()
            t_stencil = time.perf_counter() - t0
            
            # 2. Ozaki 1 (Matmul)
            # D @ u
            # Warmup
            # Note: Ozaki impl expects matrix-matrix, so use (N,1) vector
            if use_ozaki:
                _ = ozaki_matmul_v1(D, u_col)
                
                t0 = time.perf_counter()
                res_o1 = ozaki_matmul_v1(D, u_col)
                # jax.block_until_ready(res_o1) # ozaki returns jax array
                res_o1.block_until_ready()
                t_o1 = time.perf_counter() - t0
            else:
                res_o1 = None
                t_o1 = float("nan")
            
            # 3. Ozaki 2 (Matmul)
            # Warmup
            if use_ozaki:
                _ = ozaki_scheme_2_solve(D, u_col, num_moduli=num_moduli)
                
                t0 = time.perf_counter()
                res_o2 = ozaki_scheme_2_solve(D, u_col, num_moduli=num_moduli)
                res_o2.block_until_ready()
                t_o2 = time.perf_counter() - t0
            else:
                res_o2 = None
                t_o2 = float("nan")
            
            # Record
            results[order]['jax_matmul'].append(t_jax)
            results[order]['jax_stencil'].append(t_stencil)
            results[order]['ozaki_1'].append(t_o1)
            results[order]['ozaki_2'].append(t_o2)
            results[order]['sizes'].append(N)
            
            # Accuracy
            err_jax = jnp.linalg.norm(res_jax.flatten() - du_true) / jnp.linalg.norm(du_true)
            err_stencil = jnp.linalg.norm(res_stencil - du_true) / jnp.linalg.norm(du_true)
            accuracy[order]['jax_matmul'].append(float(err_jax))
            accuracy[order]['jax_stencil'].append(float(err_stencil))
            if use_ozaki:
                err_o1 = jnp.linalg.norm(res_o1.flatten() - du_true) / jnp.linalg.norm(du_true)
                err_o2 = jnp.linalg.norm(res_o2.flatten() - du_true) / jnp.linalg.norm(du_true)
                accuracy[order]['ozaki_1'].append(float(err_o1))
                accuracy[order]['ozaki_2'].append(float(err_o2))
            else:
                accuracy[order]['ozaki_1'].append(float("nan"))
                accuracy[order]['ozaki_2'].append(float("nan"))
            accuracy[order]['sizes'].append(N)
            
    # Modify data to be JSON serializable or just plot directly
    plot_results(results)
    plot_accuracy(accuracy)

def plot_results(results):
    for order, data in results.items():
        if not data['sizes']:
            print(f"No data collected for order {order}; skipping plot.")
            continue
        sizes = np.array(data['sizes'])
        def _plot_series(values, label, marker):
            vals = np.array(values, dtype=np.float64)
            mask = np.isfinite(vals)
            if np.any(mask):
                plt.plot(sizes[mask], vals[mask], label=label, marker=marker)
        plt.figure(figsize=(10, 6))
        _plot_series(data['jax_matmul'], 'JAX Baseline (MatMul FP64)', 'o')
        _plot_series(data['jax_stencil'], 'JAX Baseline (Stencil)', '^')
        _plot_series(data['ozaki_1'], 'Ozaki Scheme 1 (MatMul)', 'x')
        _plot_series(data['ozaki_2'], 'Ozaki Scheme 2 (Modular)', 's')
        
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
        sizes = np.array(data['sizes'])
        def _plot_series(values, label, marker):
            vals = np.array(values, dtype=np.float64)
            mask = np.isfinite(vals)
            if np.any(mask):
                plt.plot(sizes[mask], vals[mask], label=label, marker=marker)
        plt.figure(figsize=(10, 6))
        _plot_series(data['jax_matmul'], 'JAX Baseline (MatMul FP64)', 'o')
        _plot_series(data['jax_stencil'], 'JAX Baseline (Stencil)', '^')
        _plot_series(data['ozaki_1'], 'Ozaki Scheme 1 (MatMul)', 'x')
        _plot_series(data['ozaki_2'], 'Ozaki Scheme 2 (Modular)', 's')
        
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
