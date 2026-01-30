import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad


# The output C = AB = D-1DABEE-1 = D-1A'B'E-1 equation (10)
# Where D and E are diagonal matrices of powers of two 
# A' = DA and B' = BE
# We first set s, the number of matrix multiplications. 

def get_power_of_two_scalings(matrix):
    """
    Calculates the minimal power of 2 k such that matrix * 2^k contains only integers.
    The method analyzes the mantissa and exponent of the float64 representation.
    """
    # Decompose into mantissa and exponent
    # matrix = mantissa * 2^exponent
    # mantissa is in [0.5, 1) or 0
    mantissa, exponent = jnp.frexp(matrix)
    
    # Treat the 53 bits of mantissa.
    # We scale the mantissa by 2^53 to get an integer representation of the significant bits.
    # Taking abs() because mantissa can be negative.
    m_int = (jnp.abs(mantissa) * (2.0**53)).astype(jnp.int64)
    
    # Isolate the Least Significant Bit (LSB) to find trailing zeros
    # lsb will be a power of 2: 1, 2, 4, ...
    lsb = m_int & (-m_int)
    
    # Find the exponent of the LSB using frexp
    # frexp(lsb) returns (0.5, e_lsb) where lsb = 0.5 * 2^e_lsb = 2^(e_lsb-1)
    # The number of trailing zeros z = e_lsb - 1
    # Example: lsb = 4 (100). frexp(4) -> (0.5, 3). z = 2.
    _, e_lsb = jnp.frexp(lsb.astype(jnp.float64))
    z = (e_lsb - 1).astype(jnp.int64)
    
    # Handle the case where input is 0. m_int=0, lsb=0, frexp(0)=(0,0), z=-1.
    # If number is 0, any shift works, but we treat it as having max trailing zeros (53).
    z = jnp.where(m_int == 0, 53, z)
    
    # To convert x to integer, we need to multiply by 2^k
    # x = m_int * 2^{exponent - 53}
    # m_int = odd_part * 2^z
    # x = odd_part * 2^{exponent - 53 + z}
    # We need exponent - 53 + z + k >= 0 => k >= 53 - z - exponent
    k = 53 - z - exponent
    return k

def get_ozaki_diagonal_matrices(A, B):
    """
    Computes two diagonal matrices D and E of powers of two such that:
    A' = D @ A  is an integer matrix.
    B' = B @ E  is an integer matrix.
    
    Args:
        A: Matrix of shape (N, K)
        B: Matrix of shape (K, M)
    
    Returns:
        D: Scaling matrix for A with shape (N, N)
        E: Scaling matrix for B with shape (M, M)
    """
    # --- Compute D for A ---
    # D scales rows of A. D_ii multiplies row i.
    # We need a shift s_i for row i such that A_ij * 2^{s_i} is integer for all j.
    needed_shifts_A = get_power_of_two_scalings(A)
    # Take the maximum required shift across the row
    row_shifts = jnp.max(needed_shifts_A, axis=1) # Shape (N,)
    d_diag = jnp.exp2(row_shifts)
    D = jnp.diag(d_diag)
    
    # --- Compute E for B ---
    # E scales columns of B. E_jj multiplies col j.
    # We need a shift t_j for col j such that B_ij * 2^{t_j} is integer for all i.
    needed_shifts_B = get_power_of_two_scalings(B)
    # Take the maximum required shift across the column
    col_shifts = jnp.max(needed_shifts_B, axis=0) # Shape (M,)
    e_diag = jnp.exp2(col_shifts)
    E = jnp.diag(e_diag)
    
    return D, E

def convert_to_integer_matrix(matrix, scaling_matrix):
    """
    Multiplies a matrix by a diagonal scaling matrix to convert it to an integer matrix.
    
    Args:
        matrix: Matrix of shape (N, M)
        scaling_matrix: Diagonal matrix of shape (N, N) or (M, M)
    
    Returns:
        integer_matrix: Matrix of shape (N, M) with integer entries
    """
    # Check if scaling matrix is for rows (N, N) or columns (M, M)
    if scaling_matrix.shape[0] == matrix.shape[0]:
        # Row scaling: D @ A
        return (scaling_matrix @ matrix).astype(jnp.int64)
    elif scaling_matrix.shape[0] == matrix.shape[1]:
        # Column scaling: A @ E
        return (matrix @ scaling_matrix).astype(jnp.int64)
    else:
        raise ValueError("Scaling matrix dimensions do not match matrix dimensions.")

def ozaki_matmul_v2(A, B, s):
    D, E = get_ozaki_diagonal_matrices(A, B)
    A_prime = convert_to_integer_matrix(A, D)
    B_prime = convert_to_integer_matrix(B, E)

    m = [256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191]
    # Do modulation arithmetic
    for t in range(s):
        mt = m[t]
        
        A_t = A_prime % mt #Not sure if this is correct
        B_t = B_prime % mt
        C_t = (A_t @ B_t)
        



    
    

