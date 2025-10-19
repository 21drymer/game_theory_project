# game_theory_project

Right now this project simulated a repeated stochastic game where a transmitter is trying to send a message and a jammer is trying to stop him. The channel can either be in a good or bad state, and depending on the state there is a certain probability of which state the channel will be in next. It is a zero sum game. 

## interactive.py

In this script the user can choose the parameters they want to test on and the strategies and payoffs for those parameters are calculated.

## sweep.py

This script allows the user to choose a parameter to sweep through to see how it affects the payoffs. The rest of the parameters are fixed and a plot of the variable parameter vs the expected payoff is shown.

## solve.py

This script holds the helper functions to calculate the minimax and minimizer strategies for certain payoffs.