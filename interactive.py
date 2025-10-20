import numpy as np
from scipy.optimize import linprog
from solve import solve_minimax, solve_minimizer
import argparse

# ---------- Game Parameters ----------
states = [0, 1]             # 0 = Bad channel, 1 = Good channel
AT, AJ = [0, 1], [0, 1]     # Transmitter: 0=idle,1=transmit; Jammer: 0=idle,1=jam

# ---------- Reward Definitions ----------
r_T = np.zeros((2, 2, 2))

def main():
    # ---------- Value Iteration ----------
    parser = argparse.ArgumentParser(description="Game Theory Project Parameters")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor gamma")
    parser.add_argument("--good_prob", type=float, default=0.7, help="Probability of staying in Good channel")
    parser.add_argument("--bad_prob", type=float, default=0.4, help="Probability of staying in Bad channel")
    parser.add_argument(
        "--payoffs",
        type=float,
        nargs=8,
        default=[4, -2, 0, 1, -1, 0, 0, 1],
        metavar=("G_T_NJ", "G_T_J", "G_I_NJ", "G_I_J", "B_T_NJ", "B_T_J", "B_I_NJ", "B_I_J"),
        help="List of 8 payoffs: [G_T_NJ, G_T_J, G_I_NJ, G_I_J, B_T_NJ, B_T_J, B_I_NJ, B_I_J]"
    )

    args = parser.parse_args()

    gamma = args.gamma
    good_prob = args.good_prob
    bad_prob = args.bad_prob

    # Payoff order: [G_T_NJ, G_T_J, G_I_NJ, G_I_J, B_T_NJ, B_T_J, B_I_NJ, B_I_J]
    r_T[1, 1, 0] = args.payoffs[0]  # Good, Transmit, No Jam
    r_T[1, 1, 1] = args.payoffs[1]  # Good, Transmit, Jam
    r_T[1, 0, 0] = args.payoffs[2]  # Good, Idle, No Jam
    r_T[1, 0, 1] = args.payoffs[3]  # Good, Idle, Jam
    r_T[0, 1, 0] = args.payoffs[4]  # Bad, Transmit, No Jam
    r_T[0, 1, 1] = args.payoffs[5]  # Bad, Transmit, Jam
    r_T[0, 0, 0] = args.payoffs[6]  # Bad, Idle, No Jam
    r_T[0, 0, 1] = args.payoffs[7]  # Bad, Idle, Jam

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