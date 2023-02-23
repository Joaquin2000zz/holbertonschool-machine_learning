#!/usr/bin/env python3
"""
function which contains td_lambtha function
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    performs the TD(λ) algorithm:

    - env: is the openAI environment instance
    - V: is a numpy.ndarray of shape (s,) containing the value estimate
    - policy: is a function that takes in a state and
              returns the next action to take
    - lambtha: is the eligibility trace factor
    - episodes: is the total number of episodes to train over
    - max_steps: is the maximum number of steps per episode
    - alpha: is the learning rate
    - gamma: is the discount rate
    Returns: V, the updated value estimate
    """
    elegibility_trace = np.zeros(shape=V.shape[0]) # actions
    for _ in range(episodes):  # experience
        s = env.reset()

        for _ in range(max_steps):  # episode
            action = policy(s)
            s_new, reward, done, info = env.step(action)

            # δ = R(t+1) + γV(St+1) - V(St)
            delta_t = reward + (gamma * V[s_new]) - V[s]
            # Et(S) = γλEt-1(s) + 1(St=s)
            elegibility_trace *= (gamma * lambtha)
            elegibility_trace[s] += 1
            # V(St) = V(St) + αδEt(St)
            V += alpha * delta_t * elegibility_trace

            if done:
                break
            s = s_new
    return V
