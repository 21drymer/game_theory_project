import numpy as np
from scipy.optimize import linprog
from solve import solve_minimax, solve_minimizer
import matplotlib.pyplot as plt

def main():
    # ---------- Value Iteration ----------
    gamma = 0.9
    states = [0, 1]             # 0 = Bad channel, 1 = Good channel
    AT, AJ = [0, 1], [0, 1]     # Transmitter: 0=idle,1=transmit; Jammer: 0=idle,1=jam

    # ---------- Reward Definitions ----------
    print("Enter rewards for the following scenarios:")
    r_T = np.zeros((2, 2, 2))
    scenarios = [
        ("Good channel, transmit, no jam", 1, 1, 0),
        ("Good channel, transmit, jammed", 1, 1, 1),
        ("Bad channel, transmit, no jam", 0, 1, 0),
        ("Bad channel, transmit, jammed", 0, 1, 1)
    ]
    default_rewards = [4, 1, 0, -1]
    for idx, (desc, s, aT, aJ) in enumerate(scenarios):
        val = input(f"  Reward for {desc} (default {default_rewards[idx]}): ").strip()
        r_T[s, aT, aJ] = float(val) if val else default_rewards[idx]
    # (Idle = 0 by default)

    # ---------- Transition Probabilities ----------
    good_prob = 0.7
    bad_prob = 0.4

    # Ask user which parameter to sweep
    sweep_options = {
        "1": ("good_prob", good_prob),
        "2": ("bad_prob", bad_prob),
        "3": ("gamma", gamma)
    }
    print("Which parameter would you like to sweep?")
    for k, (name, val) in sweep_options.items():
        print(f"{k}: {name} (current value: {val})")
    choice = input("Enter the number of the parameter to sweep: ").strip()
    if choice not in sweep_options:
        raise ValueError("Invalid choice.")
    sweep_param, sweep_value = sweep_options[choice]
    payoffs = []
    for sweep in np.linspace(0.1, 0.9, 9):
        print(f"\n--- Sweeping {sweep_param} to {sweep:.2f} ---")
        if sweep_param == "good_prob":
            good_prob = sweep
        elif sweep_param == "bad_prob":
            bad_prob = sweep
        elif sweep_param == "gamma":
            gamma = sweep
        P = np.zeros((2, 2, 2, 2))
        for s in states:
            for aT in AT:
                for aJ in AJ:
                    if s == 1:  # good channel
                        P[s, aT, aJ, 1] = good_prob
                        P[s, aT, aJ, 0] = 1 - good_prob
                    else:       # bad channel
                        P[s, aT, aJ, 1] = bad_prob
                        P[s, aT, aJ, 0] = 1 - bad_prob
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
            '''
            print(f"\nState {['Bad','Good'][s]}:")
            print("  Transmitter mixed [idle, transmit]:", np.round(pT, 3))
            print("  Jammer mixed [idle, jam]:           ", np.round(pJ, 3))
            print("  Value (transmitter payoff):", round(v1, 3))
            print("  Value (jammer payoff):", -1*round(v1, 3))
            '''
            payoffs.append(v1)

            # Collect payoffs for plotting
    sweep_vals = np.linspace(0.1, 0.9, 9)
    # Plot results
    plt.figure()
    plt.plot(sweep_vals, payoffs[::2], marker='o', label='Transmitter Payoff (Bad Channel)')
    plt.plot(sweep_vals, payoffs[1::2], marker='s', label='Transmitter Payoff (Good Channel)')
    plt.xlabel(sweep_param)
    plt.ylabel("Payoffs")
    plt.title(f"Payoffs vs {sweep_param}")
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    main()