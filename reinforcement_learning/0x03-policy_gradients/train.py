#!/usr/bin/env python3
"""
module which contains train function
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=.000045, gamma=.98,
          show_result=False):
    """
    implements a full training.

    - env: initial environment
    - nb_episodes: number of episodes used for training
    - alpha: the learning rate
    - gamma: the discount factor
    Return: all values of the score
            (sum of all rewards during one episode loop)
    """
    n, m = env.observation_space.shape[0], env.action_space.n
    w = np.random.rand(n, m)
    scores = []
    for episode in range(nb_episodes + 1):
        state = env.reset()[np.newaxis, :]

        grads = []
        rewards = []
        score = 0
        T = 0
        while 1 + 1 == 2:

            if show_result and episode % 1000 == 0 and episode > 0:
                env.render()

            action, grad = policy_gradient(state, w)
            next_state, reward, done, info  = env.step(action)
            
            grads.append(grad)
            rewards.append(reward)

            score += reward

            state = next_state[np.newaxis, :]
            T += 1

            if done:
                break
        scores.append(score)
        print('Episode:', episode + 1, 'Score:', score, end='\r', flush=False)
        for t in range(T): # weight update
            w += alpha * grads[t] * np.sum(
                [r * (gamma ** r) for r in rewards[t:]]
                )
    return scores
