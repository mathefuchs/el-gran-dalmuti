## Notes

* Table-based Q-Learning infeasible, around $7 \times 10^{16}$ state-action combinations
* Deep Q-Learning for Markov Games
* $S = \textrm{Already played} \times \textrm{Board}$ at the start of the round (before first player in the order of play plays)
* $A = \textrm{Actions possible to perform}$
* Q function incorporates all agents actions 
$$Q: S \times A^{4} \\ Q(s, a_1, a_2, a_3, a_4)$$
* $a_4$ is always the agent to train
* If round starts with agent 3 (action $a_3$) for example, set $a_1$ and $a_2$ to the passing action (zeros)
* Minimax to update Q-value: 
$$V(s) = \max_{\pi} \min_{a_3} \min_{a_2} \min_{a_1} \sum_{a_4} Q(s, a_1, a_2, a_3, a_4) \pi_{a_4} \\ Q(s, a_1, a_2, a_3, a_4) = R(s, a_1, a_2, a_3, a_4) + \gamma \sum_{s'} T(s, a_1, a_2, a_3, a_4, s') V(s')$$
    * Assume that agent 1, 2 and 3 perform worst possible actions ($\min_a$) for agent 4 (agent to train)
    * $T(s, a_1, a_2, a_3, a_4, s') = 1 \Leftrightarrow s'$ is the resulting state after doing actions $a_i$ (deterministic transitions)
    * Deterministic policy. Take action with highest value estimate.
    * Q-value update rule is then simplified to
$$V(s) = \max_{a_4} \min_{a_3} \min_{a_2} \min_{a_1} Q(s, a_1, a_2, a_3, a_4) \\ Q(s, a_1, a_2, a_3, a_4) = R(s, a_1, a_2, a_3, a_4) + \gamma V(s')$$
