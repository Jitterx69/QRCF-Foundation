"""
qiskit_simulation.py
---------------------

Quantum–Reflexive Control Engine (QRCE) simulator.

This file implements:
  * 2-qubit density matrices for X ⊗ V
  * Prophecy POVM M with softness parameter α
  * Regulator channel ρ_rel with depolarization λ
  * Agent response π via controlled unitaries
  * World CPTP evolution �� (amplitude damping by default)
  * Φ^Q(ρ) = �� ∘ π ∘ ρ_rel ∘ M (ρ)
  * Fixed-point iteration
  * Trace-distance metrics
"""

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator, Kraus, partial_trace

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def tensor(a, b):
    """Tensor product."""
    return np.kron(a, b)

def trace_distance(rho, sigma):
    """Trace distance: 0.5 * ||ρ - σ||_1."""
    mat = rho - sigma
    evals = np.linalg.eigvals(mat @ mat.conj().T)
    return 0.5 * np.sum(np.sqrt(np.real(evals) + 1e-15))


# ---------------------------------------------------------
# Pauli + Identity
# ---------------------------------------------------------
I2 = np.eye(2)
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])


# ---------------------------------------------------------
# Prophecy Measurement POVM: M_i = sqrt(E_i)
# ---------------------------------------------------------

def prophecy_kraus(alpha):
    """
    Return Kraus operators for soft binary measurement POVM:
      E0 = (1 - α)I/2 + α|0><0|
      E1 = (1 - α)I/2 + α|1><1|
    """
    a = (1 + alpha) / 2
    b = (1 - alpha) / 2

    M0 = np.diag([np.sqrt(a), np.sqrt(b)])
    M1 = np.diag([np.sqrt(b), np.sqrt(a)])

    # Extend to X ⊗ V : measurement acts only on V
    return [
        tensor(I2, M0),
        tensor(I2, M1)
    ]


# ---------------------------------------------------------
# Regulator: Depolarizing channel on V only
# ---------------------------------------------------------

def regulator_kraus(lambda_):
    """
    Depolarizing channel:
      ρ -> (1-λ) ρ + λ (I/2 ⊗ Tr_V ρ)
    Implemented via standard 1-qubit depolarizing Kraus,
    then extended to X ⊗ V.
    """
    p = lambda_
    d0 = np.sqrt(1 - p)
    d1 = np.sqrt(p / 3)

    K0 = d0 * I2
    K1 = d1 * X
    K2 = d1 * Y
    K3 = d1 * Z

    # Regulator acts only on V
    return [
        tensor(I2, K0),
        tensor(I2, K1),
        tensor(I2, K2),
        tensor(I2, K3)
    ]


# ---------------------------------------------------------
# Agent Response π: controlled unitaries on X based on POVM outcome
# ---------------------------------------------------------

def agent_unitaries():
    """
    Agent chooses:
      outcome 0 → do nothing
      outcome 1 → apply X on subsystem X
    """
    U0 = tensor(I2, I2)  # identity on X⊗V
    U1 = tensor(X, I2)   # X applied to subsystem X only
    return [U0, U1]


# ---------------------------------------------------------
# World Evolution: CPTP Amplitude Damping
# ---------------------------------------------------------

def world_kraus(gamma=0.1):
    """
    Amplitude damping on X subsystem.
    Kraus:
      K0 = |0><0| + sqrt(1-gamma)|1><1|
      K1 = sqrt(gamma)|0><1|
    Extended to 2 qubits.
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    
    return [
        tensor(K0, I2),
        tensor(K1, I2)
    ]


# ---------------------------------------------------------
# Apply a CPTP map given a list of Kraus operators
# ---------------------------------------------------------

def apply_kraus(kraus_list, rho):
    """Return Σ_i A_i ρ A_i†."""
    out = np.zeros_like(rho, dtype=complex)
    for K in kraus_list:
        out += K @ rho @ K.conj().T
    return out


# ---------------------------------------------------------
# Φ^Q(ρ) Implementation
# ---------------------------------------------------------

def Phi_Q(rho, alpha, lambda_, gamma=0.1):
    """
    Compute:
      Φ^Q(ρ) = �� ∘ π ∘ ρ_rel ∘ M (ρ)
    """
    # --- Prophecy measurement ---
    M_ops = prophecy_kraus(alpha)
    rhoM = apply_kraus(M_ops, rho)

    # --- Regulator ---
    R_ops = regulator_kraus(lambda_)
    rhoR = apply_kraus(R_ops, rhoM)

    # --- Agent response (post-measurement branching) ---
    U_ops = agent_unitaries()
    # Expand measurement outcomes: sum_i U_i M_i ρ M_i† U_i†
    rhoA = np.zeros_like(rho, dtype=complex)
    for i, Mi in enumerate(M_ops):
        branch = Mi @ rho @ Mi.conj().T
        rhoA += U_ops[i] @ branch @ U_ops[i].conj().T

    # --- World evolution ---
    W_ops = world_kraus(gamma)
    rhoW = apply_kraus(W_ops, rhoA)

    return rhoW


# ---------------------------------------------------------
# Fixed Point Iteration
# ---------------------------------------------------------

def fixed_point(rho0, alpha, lambda_, gamma=0.1, tol=1e-10, max_iter=500):
    """
    Iteratively compute ρ_{t+1} = Φ^Q(ρ_t).
    Returns (ρ*, iterations, converged?).
    """
    rho = rho0.copy()
    for t in range(max_iter):
        new_rho = Phi_Q(rho, alpha, lambda_, gamma)
        dist = trace_distance(new_rho, rho)
        if dist < tol:
            return new_rho, t+1, True
        rho = new_rho
    return rho, max_iter, False


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

def initial_state():
    """
    Return initial 2-qubit state ρ0 = |01⟩⟨01|.
    """
    ket = np.zeros(4)
    ket[1] = 1.0
    rho = np.outer(ket, ket.conj())
    return rho


# ---------------------------------------------------------
# Example usage when run directly
# ---------------------------------------------------------

if __name__ == "__main__":

    rho0 = initial_state()
    alpha = 0.3   # measurement sharpness
    lambda_ = 0.2 # regulator strength
    gamma = 0.1   # amplitude damping

    fp, iters, ok = fixed_point(rho0, alpha, lambda_, gamma)

    print("Converged:", ok)
    print("Iterations:", iters)
    print("Fixed point ρ*:")
    np.set_printoptions(precision=4, suppress=True)
    print(fp)
