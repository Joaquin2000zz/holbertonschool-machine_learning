#!/usr/bin/env python3
"""
module which contains sarsa_lambtha function
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

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    performs SARSA(λ):

    - env: is the openAI environment instance
    - Q: is a numpy.ndarray of shape (s,a) containing the Q table
    - lambtha: is the eligibility trace factor
    - episodes: is the total number of episodes to train over
    - max_steps: is the maximum number of steps per episode
    - alpha: is the learning rate
    - gamma: is the discount rate
    - epsilon: is the initial threshold for epsilon greedy
    - min_epsilon: is the minimum value that epsilon should decay to
    - epsilon_decay: is the decay rate for updating epsilon between episodes
    Returns: Q, the updated Q table
    """
    initial_epsilon = epsilon
    s, a = Q.shape
    elegibility_trace = np.zeros(shape=(s, a))
    for _ in range(episodes):
        s = env.reset()
        action = epsilon_greedy(Q, s, epsilon)
        for _ in range(max_steps):
            # exploration-exploitation
            new_action = epsilon_greedy(Q, s, epsilon)
            new_s, reward, done, info = env.step(new_action)
            # δ = R(t+1) + γQ(St+1, At+1) - Q(St, At)
            direction = Q[new_s, new_action] - Q[s, action]
            delta_t = reward + gamma * direction
            # Et = γλEt-1(s) + Q(St+1, At+1) - Q(St, At)
            elegibility_trace[s, action] += 1
            # Q(St) = Q(St) + αδEt(s)
            Q += alpha * delta_t * elegibility_trace
            elegibility_trace *= (gamma * lambtha)
            action = new_action
            if done:
                break
            # exploration rate decay
            exp = np.exp(- epsilon_decay * episodes)
            epsilon = min_epsilon + \
                (min_epsilon + (initial_epsilon - min_epsilon)) * exp
    return Q
