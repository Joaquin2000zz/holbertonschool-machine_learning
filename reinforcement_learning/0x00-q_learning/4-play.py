#!/usr/bin/env/ python3
"""
module which contains play function
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    that has the trained agent play an episode:

    env is the FrozenLakeEnv instance
    Q is a numpy.ndarray containing the Q-table
    max_steps is the maximum number of steps in the episode
    Each state of the board should be displayed via the console
    You should always exploit the Q-table
    Returns: the total rewards for the episode
    """
    state = env.reset()
    env.render()
    for step in range(max_steps):
        action = np.argmax(Q[state]) # take decision
        new_state, reward, done, info = env.step(action)
        env.render()
        if done:
            break
        state = new_state
    env.close()
    return reward
