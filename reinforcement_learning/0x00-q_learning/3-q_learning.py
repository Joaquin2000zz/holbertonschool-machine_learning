#!/usr/bin/env python3
"""
module which contains train function
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=.1, gamma=.99,
          epsilon=1, min_epsilon=.1, epsilon_decay=.5):
    """
    performs Q-learning
    - env: is the FrozenLakeEnv instance
    - Q: is a numpy.ndarray containing the Q-table
    - episodes: is the total number of episodes to train over
    - max_steps: is the maximum number of steps per episode
    - alpha: is the learning rate
    - gamma: is the discount rate
    - epsilon: is the initial threshold for epsilon greedy
    - min_epsilon: is the minimum value that epsilon should decay to
    - epsilon_decay: is the decay rate for updating epsilon between episodes
    - When the agent falls in a hole, the reward should be updated to be -1
    Returns: Q, total_rewards
        - Q: is the updated Q-table
        - total_rewards: is a list containing the rewards per episode
    """
    initial_epsilon = epsilon
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        state = state
        rewards_current_episode = 0
        for step in range(max_steps):
            # exploration-exploitation
            action = epsilon_greedy(Q=Q, state=state, epsilon=epsilon)
            # new step given action choosed
            new_state, reward, done, info = env.step(action)
            if done and reward == 0:
                    reward = -1
            # updation Q-table for Q(s,a)
            # with a weighted sum of the old value and the learned value
            Q[state, action] = Q[state, action] + alpha * \
                (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            # transition to new state
            state = new_state
            rewards_current_episode += reward
            if done:
                break
        # exploration rate decay
        epsilon = min_epsilon + \
            (min_epsilon + (initial_epsilon - min_epsilon)) * np.exp(- epsilon_decay * episode)
        total_rewards.append(rewards_current_episode)
    return Q, total_rewards
