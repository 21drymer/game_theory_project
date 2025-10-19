import numpy as np
from scipy.optimize import linprog
from solve import solve_minimax, solve_minimizer

# ---------- Game Parameters ----------
gamma = float(input("Enter discount factor gamma (e.g., 0.9): "))
states = [0, 1]             # 0 = Bad channel, 1 = Good channel
AT, AJ = [0, 1], [0, 1]     # Transmitter: 0=idle,1=transmit; Jammer: 0=idle,1=jam

# ---------- Reward Definitions ----------
r_T = np.zeros((2, 2, 2))
r_T[1, 1, 0] = float(input("Reward for Good channel, transmit, no jam (e.g., 4): "))
r_T[1, 1, 1] = float(input("Reward for Good channel, transmit, jammed (e.g., 1): "))
r_T[0, 1, 0] = float(input("Reward for Bad channel, transmit, no jam (e.g., 0): "))
r_T[0, 1, 1] = float(input("Reward for Bad channel, transmit, jammed (e.g., -1): "))
# (Idle = 0 by default)

# ---------- Transition Probabilities ----------
good_prob = float(input("Probability of staying in Good channel (e.g., 0.7): "))
bad_prob = float(input("Probability of staying in Bad channel (e.g., 0.4): "))
P = np.zeros((2, 2, 2, 2))
for s in states:
    for aT in AT:
        for aJ in AJ:
            if s == 1:  # good channel
                P[s, aT, aJ, 1] = good_prob
                P[s, aT, aJ, 0] = 1-good_prob
            else:       # bad channel
                P[s, aT, aJ, 1] = bad_prob
                P[s, aT, aJ, 0] = 1-bad_prob

def main():
    # ---------- Value Iteration ----------
    
    V = np.zeros(len(states))
    for _ in range(200):
        V_new = np.zeros_like(V)
        for s in states:
            Q = np.zeros((len(AT), len(AJ)))
            for i, aT in enumerate(AT):
                for j, aJ in enumerate(AJ):
                    Q[i, j] = r_T[s, aT, aJ] + gamma * np.dot(P[s, aT, aJ], V)
            _, v = solve_minimax(Q)
            V_new[s] = v
        if np.max(np.abs(V_new - V)) < 1e-6:
            break
        V = V_new

    print("Converged state values:", np.round(V, 3))

    # ---------- Equilibrium Mixed Strategies ----------
    for s in states:
        Q = np.zeros((len(AT), len(AJ)))
        for i, aT in enumerate(AT):
            for j, aJ in enumerate(AJ):
                Q[i, j] = r_T[s, aT, aJ] + gamma * np.dot(P[s, aT, aJ], V)
        pT, v1 = solve_minimax(Q)
        pJ, v2 = solve_minimizer(Q)
        print(f"\nState {['Bad','Good'][s]}:")
        print("  Transmitter mixed [idle, transmit]:", np.round(pT, 3))
        print("  Jammer mixed [idle, jam]:           ", np.round(pJ, 3))
        print("  Value (transmitter payoff):", round(v1, 3))
        print("  Value (jammer payoff):", -1*round(v1, 3))

if __name__ == "__main__":
    main()