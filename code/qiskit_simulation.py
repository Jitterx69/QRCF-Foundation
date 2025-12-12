#!/usr/bin/env python3
"""
qiskit_simulation.py (advanced)

Quantum–Reflexive Control Engine (QRCE) simulation toolkit.

Requirements:
  - numpy
  - scipy
  - (optional) matplotlib for plotting

Design goals:
  - Numerically robust implementations of trace norms, entropy, mutual info.
  - Configurable pipeline: measurement instrument, regulator, agent, world.
  - Fixed-point iteration with diagnostics and contraction tests.
  - Grid search and Pareto reporting for ethical trade-offs.
"""

from typing import List, Tuple, Callable, Dict, Any
import numpy as np
import scipy.linalg as la

# Optional plotting
try:
    import matplotlib.pyplot as plt  # type: ignore
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# -----------------------
# Basic linear algebra helpers
# -----------------------

def tensor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kronecker product."""
    return np.kron(a, b)

def dagger(A: np.ndarray) -> np.ndarray:
    return A.conj().T

def is_hermitian(A: np.ndarray, tol: float = 1e-10) -> bool:
    return np.max(np.abs(A - A.conj().T)) < tol

def symmetrize(rho: np.ndarray) -> np.ndarray:
    """Force Hermiticity numerically."""
    return 0.5 * (rho + rho.conj().T)

# -----------------------
# Partial trace utility
# -----------------------

def partial_trace(rho: np.ndarray, keep: int, dims: Tuple[int, int]) -> np.ndarray:
    """
    Partial trace over a bipartite system of dimensions dims=(dA, dB).
    keep=0 returns Tr_B(rho) (i.e., keep subsystem A)
    keep=1 returns Tr_A(rho) (i.e., keep subsystem B)
    """
    dA, dB = dims
    if rho.shape != (dA * dB, dA * dB):
        raise ValueError("rho shape mismatch for given dims")
    if keep == 0:
        # Trace out B
        res = np.zeros((dA, dA), dtype=complex)
        for i in range(dB):
            idx = slice(i * dA, (i + 1) * dA)
            res += rho[idx, idx]
        return res
    else:
        # Trace out A
        res = np.zeros((dB, dB), dtype=complex)
        for i in range(dA):
            idx = slice(i * dB, (i + 1) * dB)
            # need to extract blocks at step dB across
            res += rho[i::dA, i::dA].reshape(dB, dB)
        # Simpler/faster: use reshape/transposes
        rho_reshaped = rho.reshape(dA, dB, dA, dB)
        res = np.einsum('i a j a -> i j', rho_reshaped, optimize=True)
        return res

# -----------------------
# Trace norm / trace distance
# -----------------------

def trace_norm(A: np.ndarray) -> float:
    """
    Trace norm ||A||_1 = sum singular values.
    For Hermitian A: sum absolute eigenvalues.
    Numeric stability: use SVD on general matrices.
    """
    # Use svd for numerical stability
    s = la.svd(A, compute_uv=False)
    return float(np.sum(s))

def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance 0.5 * ||rho - sigma||_1."""
    diff = rho - sigma
    return 0.5 * trace_norm(diff)

# -----------------------
# Entropy and mutual information
# -----------------------

def _safe_eigvals(rho: np.ndarray) -> np.ndarray:
    """Return eigenvalues clipped to [0,1] and real parts (numerical)."""
    # Ensure Hermitian
    rho = symmetrize(rho)
    vals = la.eigvalsh(rho)
    vals = np.clip(np.real(vals), 0.0, 1.0)
    return vals

def von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    """S(ρ) = -Tr[ρ log ρ]. Use base=2 for bits."""
    vals = _safe_eigvals(rho)
    # avoid log(0)
    eps = 1e-15
    vals_nonzero = vals[vals > eps]
    return float(-np.sum(vals_nonzero * np.log(vals_nonzero) / np.log(base)))

def mutual_information(rho_AB: np.ndarray, dims: Tuple[int, int], base: float = 2.0) -> float:
    """I(A;B) = S(A) + S(B) - S(AB). dims=(dA,dB)."""
    dA, dB = dims
    rhoA = partial_trace(rho_AB, keep=0, dims=(dA, dB))
    rhoB = partial_trace(rho_AB, keep=1, dims=(dA, dB))
    return von_neumann_entropy(rhoA, base) + von_neumann_entropy(rhoB, base) - von_neumann_entropy(rho_AB, base)

# -----------------------
# Basic operators (single-qubit)
# -----------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
proj0 = np.array([[1,0],[0,0]], dtype=complex)
proj1 = np.array([[0,0],[0,1]], dtype=complex)

# -----------------------
# Measurement (POVM + Instrument)
# -----------------------

def binary_soft_povm(alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binary soft POVM elements on 1 qubit:
      E0 = (1-α) I/2 + α |0><0|
      E1 = (1-α) I/2 + α |1><1|
    Returns E0, E1 (2x2 Hermitian PSD).
    """
    E0 = (1 - alpha) * (I2 / 2.0) + alpha * proj0
    E1 = (1 - alpha) * (I2 / 2.0) + alpha * proj1
    return E0, E1

def pow2sqrt(mat: np.ndarray) -> np.ndarray:
    """Matrix square root for positive semidefinite matrix."""
    vals, vecs = la.eigh(symmetrize(mat))
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T

def instrument_kraus_from_povm(E_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Convert POVM elements {E_i} to Kraus operators M_i = sqrt(E_i) (canonical instrument).
    Returns list of M_i (2x2).
    """
    return [pow2sqrt(E) for E in E_list]

# -----------------------
# Regulator channel on V (depolarizing) implemented as convex mix
# -----------------------

def depolarizing_kraus_on_V(lambda_: float) -> List[np.ndarray]:
    """
    Kraus operators for depolarizing channel (1-qubit):
    K0 = sqrt(1 - p) I, K1 = sqrt(p/3) X, K2 = sqrt(p/3) Y, K3 = sqrt(p/3) Z
    These act on V; to extend to X⊗V, use tensor(I_X, K).
    """
    p = float(lambda_)
    k0 = np.sqrt(max(0.0, 1 - p)) * I2
    k_rest = np.sqrt(max(0.0, p / 3.0))
    return [k0, k_rest * X, k_rest * Y, k_rest * Z]

# -----------------------
# Agent response: generic policy interface
# -----------------------

def default_agent_policy(num_outcomes: int) -> List[np.ndarray]:
    """
    Default deterministic agent: outcome 0 -> Identity on X; outcome 1 -> X on X.
    Returns list of unitaries acting on X (2x2). Extended to X⊗V with tensor(u, I2).
    """
    U0 = I2.copy()
    if num_outcomes > 1:
        U1 = X.copy()
    else:
        U1 = I2.copy()
    # If more outcomes, repeat identity for extras
    Us = [U0]
    if num_outcomes > 1:
        Us.append(U1)
    for _ in range(2, num_outcomes):
        Us.append(I2.copy())
    return Us

# -----------------------
# World CPTP: amplitude damping on X extended to X⊗V
# -----------------------

def amplitude_damping_kraus_on_X(gamma: float) -> List[np.ndarray]:
    """
    Single-qubit amplitude damping Kraus on X:
      A0 = [[1,0], [0, sqrt(1-gamma)]], A1 = [[0, sqrt(gamma)], [0,0]]
    These are then extended as tensor(Ak, I_V).
    """
    a0 = np.array([[1.0, 0.0], [0.0, np.sqrt(max(0.0, 1.0 - gamma))]], dtype=complex)
    a1 = np.array([[0.0, np.sqrt(max(0.0, gamma))], [0.0, 0.0]], dtype=complex)
    return [a0, a1]

# -----------------------
# Helper: extend single-qubit ops to 2-qubit X⊗V
# -----------------------

def extend_to_XV(op: np.ndarray, on: str = 'V') -> np.ndarray:
    """
    Extend a single-qubit operator to the full 2-qubit space.
    on='V' -> I_X ⊗ op; on='X' -> op ⊗ I_V
    """
    if on == 'V':
        return tensor(I2, op)
    else:
        return tensor(op, I2)

# -----------------------
# Apply Kraus list on full 2-qubit rho
# -----------------------

def apply_kraus_full(kraus_full: List[np.ndarray], rho: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rho, dtype=complex)
    for K in kraus_full:
        out += K @ rho @ dagger(K)
    # numerical symmetrize and ensure trace ~ 1
    out = symmetrize(out)
    tr = np.real(np.trace(out))
    if tr <= 0:
        raise RuntimeError("Non-positive trace after channel application")
    return out / tr

# -----------------------
# Phi^Q pipeline
# -----------------------

def PhiQ(
    rho: np.ndarray,
    alpha: float,
    lambda_: float,
    gamma: float = 0.1,
    agent_policy: Callable[[int], List[np.ndarray]] = default_agent_policy
) -> np.ndarray:
    """
    Full composed operator:
       1) M: measurement instrument on V (canonical sqrt(E) Kraus)
       2) rho_rel: regulator (depolarizing on V)
       3) pi: agent response (unitary on X conditioned on outcomes)
       4) E: world evolution (amplitude damping on X)
    All maps are CPTP and implemented via Kraus sums on the full space.
    """
    # dims
    dX, dV = 2, 2
    # --- measurement instrument ---
    E0, E1 = binary_soft_povm(alpha)
    Ms = instrument_kraus_from_povm([E0, E1])  # each M is 2x2 on V
    M_ops_full = [extend_to_XV(Mi, on='V') for Mi in Ms]  # list of 4x4

    # apply measurement instrument (non-projective)
    rho_after_M = apply_kraus_full(M_ops_full, rho)

    # --- regulator: depolarizing on V ---
    K_vs = depolarizing_kraus_on_V(lambda_)
    K_vs_full = [extend_to_XV(K, on='V') for K in K_vs]
    rho_after_reg = apply_kraus_full(K_vs_full, rho_after_M)

    # --- agent response: branch-by-branch application using instrument branches ---
    Us = agent_policy(len(Ms))  # returns unitaries on X (2x2)
    # extend to full: Ui_full = tensor(Ui, I_V)
    U_full = [extend_to_XV(Ui, on='X') for Ui in Us]

    rho_after_agent = np.zeros_like(rho, dtype=complex)
    # proper instrument branch: sum_i (U_i ⊗ I) M_i rho M_i^\dagger (U_i^\dagger ⊗ I)
    for i, Mi in enumerate(M_ops_full):
        branch = Mi @ rho @ dagger(Mi)
        rho_after_agent += U_full[i] @ branch @ dagger(U_full[i])
    rho_after_agent = symmetrize(rho_after_agent)
    rho_after_agent = rho_after_agent / np.real(np.trace(rho_after_agent))

    # --- world evolution ---
    A_ks = amplitude_damping_kraus_on_X(gamma)
    A_full = [extend_to_XV(Ak, on='X') for Ak in A_ks]
    rho_out = apply_kraus_full(A_full, rho_after_agent)

    return rho_out

# -----------------------
# Crash projector + metrics
# -----------------------

def crash_probability(rho: np.ndarray) -> float:
    """
    Crash projector defined as |1><1| on X tensor I_V:
    P_crash = proj1_X ⊗ I_V
    """
    P = tensor(proj1, I2)
    return float(np.real(np.trace(P @ rho)))

def disturbance_norm(rho: np.ndarray, alpha: float) -> float:
    """|| rho - M(rho) ||_1 using canonical instrument."""
    E0, E1 = binary_soft_povm(alpha)
    Ms = instrument_kraus_from_povm([E0, E1])
    M_ops_full = [extend_to_XV(Mi, on='V') for Mi in Ms]
    rhoM = apply_kraus_full(M_ops_full, rho)
    return trace_norm(rho - rhoM)

def leakage_I(rho: np.ndarray) -> float:
    """Mutual information I(X;V) in bits for 2-qubit rho"""
    return mutual_information(rho, dims=(2,2), base=2.0)

# -----------------------
# Fixed-point iteration with diagnostics
# -----------------------

def fixed_point_iteration(
    rho0: np.ndarray,
    alpha: float,
    lambda_: float,
    gamma: float = 0.1,
    tol: float = 1e-10,
    max_iter: int = 2000,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Iterate rho_{t+1} = PhiQ(rho_t) until trace-distance < tol or max_iter.
    Returns dict with rho_star, history (trace distances), converged flag, iterations.
    """
    rho = rho0.copy()
    history = []
    for t in range(1, max_iter + 1):
        rho_next = PhiQ(rho, alpha, lambda_, gamma)
        d = trace_distance(rho_next, rho)
        history.append(d)
        rho = rho_next
        if verbose and t % 50 == 0:
            print(f"[iter {t}] trace-dist change = {d:.3e}")
        if d < tol:
            return dict(rho_star=rho, history=np.array(history), converged=True, iters=t)
    return dict(rho_star=rho, history=np.array(history), converged=False, iters=max_iter)

# -----------------------
# Empirical contraction test
# -----------------------

def empirical_local_gain(
    rho: np.ndarray,
    alpha: float,
    lambda_: float,
    gamma: float = 0.1,
    delta: float = 1e-5,
    samples: int = 400
) -> float:
    """
    Sample random traceless Hermitian perturbations H with ||H||_1 = 1, compute:
      G = max_{samples} ||Phi(rho+δH) - Phi(rho)||_1 / ||δH||_1
    Returns empirical max gain.
    """
    d = rho.shape[0]
    base = PhiQ(rho, alpha, lambda_, gamma)
    max_gain = 0.0
    for _ in range(samples):
        # random Hermitian traceless matrix
        X_rand = np.random.normal(size=(d,d)) + 1j * np.random.normal(size=(d,d))
        H = X_rand + X_rand.conj().T
        # make traceless
        H = H - np.trace(H) * np.eye(d) / d
        # scale so trace-norm 1
        tn = trace_norm(H)
        if tn == 0:
            continue
        H = H / tn
        rho_pert = rho + delta * H
        # project back to nearest positive semidefinite with trace 1 heuristically:
        # small delta ensures positivity preserved; if not, use nearest PSD
        try:
            # ensure hermiticity
            rho_pert = symmetrize(rho_pert)
            # small eigen clipping
            vals, vecs = la.eigh(rho_pert)
            vals_clipped = np.clip(vals, 0.0, None)
            rho_pert = (vecs * vals_clipped) @ vecs.conj().T
            rho_pert = rho_pert / np.real(np.trace(rho_pert))
        except Exception:
            continue
        out1 = PhiQ(rho, alpha, lambda_, gamma)
        out2 = PhiQ(rho_pert, alpha, lambda_, gamma)
        numer = trace_norm(out2 - out1)
        denom = trace_norm(rho_pert - rho)
        if denom > 0:
            gain = numer / denom
            if gain > max_gain:
                max_gain = gain
    return float(max_gain)

# -----------------------
# Grid search for ethical/regulator trade-offs
# -----------------------

def grid_search_alpha_lambda(
    rho0: np.ndarray,
    alphas: List[float],
    lambdas: List[float],
    gamma: float = 0.1,
    tol: float = 1e-8,
    max_iter: int = 2000
) -> List[Dict[str, Any]]:
    """
    For each (alpha, lambda) compute fixed point (or iterate until max_iter), then compute:
      - H = c1 * P_crash + c2 * disturbance
      - L = mutual information (leakage)
    Return list of results dicts for each grid cell.
    """
    results = []
    c1, c2 = 1.0, 0.5  # cost weights (configurable)
    for alpha in alphas:
        for lam in lambdas:
            res = fixed_point_iteration(rho0, alpha, lam, gamma, tol=tol, max_iter=max_iter, verbose=False)
            rho_star = res['rho_star']
            Pcrash = crash_probability(rho_star)
            dist = disturbance_norm(rho_star, alpha)
            leak = leakage_I(rho_star)
            H = c1 * Pcrash + c2 * dist
            # empirical contraction estimate (cheap): local gain around rho_star with few samples
            gain = empirical_local_gain(rho_star, alpha, lam, gamma, delta=1e-6, samples=100)
            results.append(dict(alpha=alpha, lambda_=lam, rho_star=rho_star,
                                Pcrash=Pcrash, disturbance=dist, leakage=leak,
                                harm=H, emp_gain=gain, converged=res['converged'], iters=res['iters']))
    return results

# -----------------------
# Utilities: pretty-print results and basic plotting
# -----------------------

def summarize_grid_results(results: List[Dict[str, Any]]) -> None:
    """Print Pareto frontier candidates and best harm, best leak."""
    # best harm
    best = min(results, key=lambda r: r['harm'])
    print(f"Best harm: alpha={best['alpha']:.3f}, lambda={best['lambda_']:.3f}, harm={best['harm']:.6f}, leak={best['leakage']:.6f}, Pcrash={best['Pcrash']:.6f}")
    # best leakage
    bestL = min(results, key=lambda r: r['leakage'])
    print(f"Best leakage: alpha={bestL['alpha']:.3f}, lambda={bestL['lambda_']:.3f}, leak={bestL['leakage']:.6f}, harm={bestL['harm']:.6f}")
    # Pareto (simple)
    pareto = []
    for a in results:
        dominated = False
        for b in results:
            if (b['harm'] <= a['harm'] and b['leakage'] <= a['leakage']) and (b['harm'] < a['harm'] or b['leakage'] < a['leakage']):
                dominated = True
                break
        if not dominated:
            pareto.append(a)
    print(f"Pareto frontier size: {len(pareto)}")
    # sort and show few
    pareto_sorted = sorted(pareto, key=lambda r: (r['harm'], r['leakage']))
    for p in pareto_sorted[:10]:
        print(f"Pareto: α={p['alpha']:.3f}, λ={p['lambda_']:.3f}, harm={p['harm']:.6f}, leak={p['leakage']:.6f}, emp_gain={p['emp_gain']:.3f}")

def plot_fixed_point_trajectory(history: np.ndarray) -> None:
    """Plot trace-distance history if matplotlib available."""
    if not HAS_MPL:
        print("Matplotlib not available.")
        return
    plt.figure(figsize=(6,3))
    plt.semilogy(history + 1e-20)
    plt.xlabel("Iteration")
    plt.ylabel("Trace-distance change")
    plt.title("Fixed-point convergence trace")
    plt.grid(True)
    plt.show()

# -----------------------
# Example main: reproduce numbers from paper's toy model
# -----------------------

def canonical_initial_state(p: float = 0.7) -> np.ndarray:
    """
    |ψ0> = sqrt(p)|00> + sqrt(1-p)|11> as in the worked example
    Returns rho0 = |ψ0><ψ0|
    """
    a = np.sqrt(p)
    b = np.sqrt(1 - p)
    psi = np.zeros(4, dtype=complex)
    psi[0] = a  # |00>
    psi[3] = b  # |11>
    rho0 = np.outer(psi, psi.conj())
    return rho0

if __name__ == "__main__":
    # Toy model parameters used in Section 8 of the monograph
    p = 0.7
    rho0 = canonical_initial_state(p)
    alpha = 0.6
    lambda_ = 0.3
    gamma = 0.2

    print("Initial crash prob:", crash_probability(rho0))
    print("Initial mutual information (X;V):", leakage_I(rho0))

    result = fixed_point_iteration(rho0, alpha, lambda_, gamma, tol=1e-10, max_iter=1000, verbose=True)
    rho_star = result['rho_star']
    print("Converged:", result['converged'], "iters:", result['iters'])
    print("Fixed-point crash prob:", crash_probability(rho_star))
    print("Fixed-point leakage:", leakage_I(rho_star))
    print("Fixed-point disturbance:", disturbance_norm(rho_star, alpha))
    print("Empirical contraction gain near fixed point:", empirical_local_gain(rho_star, alpha, lambda_, gamma, delta=1e-6, samples=200))

    # quick grid search (coarse) - comment out if expensive
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.5]
    print("Starting coarse grid search...")
    grid = grid_search_alpha_lambda(rho0, alphas, lambdas, gamma, tol=1e-9, max_iter=500)
    summarize_grid_results(grid)

    # optionally: show convergence trace
    if result['history'].size > 0:
        plot_fixed_point_trajectory(result['history'])
