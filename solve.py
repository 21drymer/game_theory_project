import numpy as np
from scipy.optimize import linprog

def solve_minimax(Q):
    """
    Solve zero-sum matrix game for transmitter (maximizer).
    Returns (p*, value) where p* is the transmitter's mixed strategy.
    """
    nT, nJ = Q.shape
    c = np.zeros(nT + 1)
    c[-1] = -1  # maximize v => minimize -v
    # -Q^T p + v <= 0  => A_ub x <= b_ub
    A_ub = np.hstack([-Q.T, np.ones((nJ, 1))])
    b_ub = np.zeros(nJ)
    # sum(p) = 1
    A_eq = np.zeros((1, nT + 1))
    A_eq[0, :nT] = 1
    b_eq = [1]
    bounds = [(0, 1)] * nT + [(None, None)]
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds, method="highs")
    p = res.x[:nT]
    v = res.x[-1]
    return p, v

def solve_minimizer(Q):
    """
    Solve zero-sum matrix game for jammer (minimizer).
    Returns (q*, value) where q* is jammer's mixed strategy.
    """
    nT, nJ = Q.shape
    c = np.zeros(nJ + 1)
    c[-1] = 1  # minimize v
    # Q q - v <= 0
    A_ub = np.hstack([Q, -np.ones((nT, 1))])
    b_ub = np.zeros(nT)
    # sum(q) = 1
    A_eq = np.zeros((1, nJ + 1))
    A_eq[0, :nJ] = 1
    b_eq = [1]
    bounds = [(0, 1)] * nJ + [(None, None)]
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds, method="highs")
    q = res.x[:nJ]
    v = res.x[-1]
    return q, v
