#!/usr/bin/env python3
"""
module which contains epsilon_greedy function
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    epsilon-greedy to determine the next action:

    - Q: is a numpy.ndarray containing the q-table
    - state: is the current state
    - epsilon: is the epsilon to use for the calculation
    - You should sample p with numpy.random.uniformn to determine
      if your algorithm should explore or exploit
    - If exploring, you should pick the next action with numpy.random.randint
      from all possible actions
    Returns: the next action index
    """

    # exploration-exploitation trade-off
    exploration_rate_threshold = np.random.uniform(low=0, high=1) # [0, 1)
    if exploration_rate_threshold > epsilon:
        # choosing action via exploitation
        action = np.argmax(Q[state, :])
    else:
        # choosing action via exploration
        action = np.random.randint(low=0, high=Q.shape[1])
    return action
