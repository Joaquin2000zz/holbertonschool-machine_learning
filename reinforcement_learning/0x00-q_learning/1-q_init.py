#!/usr/bin/env python3
"""
module which contains q_init function
"""
import numpy as np


def q_init(env):
    """
    - env: is the FrozenLakeEnv instance
    Returns: the Q-table as a numpy.ndarray of zeros
    """
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    return np.zeros(shape=(state_space_size, action_space_size))
