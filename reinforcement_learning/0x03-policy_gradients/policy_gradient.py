#!/usr/bin/env python3
"""
module which contains policy, softmax_grad, and policy_gradient functions
"""
import numpy as np


def policy(matrix, weight):
    """
    function that computes to policy with a weight of a matrix
    matrix: the observation
    weight: parameter to update
    returns the action's probabilities
    """
    # softmax function
    z = np.exp(matrix.dot(weight))
    return z / z.sum()

def softmax_grad(softmax):
    """
    gradient of the softmax function
    """
    s = softmax.reshape(-1, 1)

    return np.diagflat(s) - s.dot(s.T)

def policy_gradient(state, weight):
    """
    computes the Monte-Carlo policy gradient based on a state
    and a weight matrix.

    - state: matrix representing the current observation of the environment
    - weight: matrix of random weight
    Return: the action and the gradient (in this order)
    """
    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])

    d_soft = softmax_grad(probs)

    d_log = d_soft[action, :] / probs[0, action]

    grad = state.T.dot(d_log[np.newaxis, :])

    return action, grad
