import numpy as np

import syft as sy
from syft import Variable as Var
from syft import nn
from syft import optim

# this is our hook
hook = sy.TorchHook()
me = hook.local_worker

bob = sy.VirtualWorker(id="bob", hook=hook)
alice = sy.VirtualWorker(id="alice", hook=hook)

me.add_workers([bob, alice])

# Create our MDP
# A forest is managed by two actions: 'Wait' and 'Cut'.
# An action is decided each year with first the objective
# to maintain an old forest for wildlife and second to
# make money selling cut wood. Each year there is a
# probability p that a fire burns the forest.

# A state is Markov if and only if:
# Transition[S_{t+1} | S{t}] = Transition[S_{t+1} | S_{1}, \dots, S{t}]

# States:
# {0, 1, ..., s-1} are states of the forest where s-1 oldest

# Actions:
# 'wait' action 0, 'cut' action 1

# after a fire the forest is in youngest state

# probability transition matrix
# shape: A x S x S'
# each actions transition matrix is indexable by action
# transitions[a] returns the S x S transition matrix
# Defines the transition probabilities from all states (rows)
# to all successor states (columns), each row sums to 1

# TODO(korymath): must make sure that it is a valid MDP (square, stochastic, nonnegative)
transitions = sy.FloatTensor(
                      [[[0.5, 0.5],
                        [0.8, 0.2]],
                       [[0. , 1.  ],
                        [0.1, 0.9]]])

# reward matrix
# shape: A x S x S'
rewards = sy.FloatTensor(
                      [[[ 5,  5],
                        [-1, -1]],
                       [[10, 10],
                        [ 2,  2]]])

def gridworld():
    """4x4 gridworld example.
    Example 4.1 of `Reinforcement Learning: An Introduction
    <http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html>`_,
    by Richard S. Sutton and Andrew G. Barto.
    Returns
    -------
    out : tuple
        ``out[0]`` contains the transition probability matrix P,
        and ``out[1]`` contains the reward matrix R. The non-terminal
        states correspond to the indices 0-13 in both matrices,
        and the terminal state to the index 14.
    """
    # States: labelled 1, 2, ..., 14 in the figure, plus the terminal
    # state associated with the two terminal positions.
    S = 16  # number of states

    # Actions: up, down, right, left.
    A = 4  # number of actions
    up, down, right, left = range(A)  # indices of the actions

    # Transitions.
    P = sy.zeros((A, S, S))

    # Grid transitions.
    grid_transitions = {
        # from_state: ((action, to_state), ...)
        0: ((up, 0), (down, 0), (right, 0), (left, 0)),
        1: ((up, 1), (down, 5), (right, 2), (left, 0)),
        2: ((up, 2), (down, 6), (right, 3), (left, 1)),
        3: ((up, 3), (down, 7), (right, 3), (left, 2)),
        4: ((up, 0), (down, 8), (right, 5), (left, 4)),
        5: ((up, 1), (down, 9), (right, 6), (left, 4)),
        6: ((up, 2), (down, 10), (right, 7), (left, 5)),
        7: ((up, 3), (down, 11), (right, 7), (left, 6)),
        8: ((up, 4), (down, 12), (right, 9), (left, 8)),
        9: ((up, 5), (down, 13), (right, 10), (left, 8)),
        10: ((up, 6), (down, 14), (right, 11), (left, 9)),
        11: ((up, 7), (down, 15), (right, 11), (left, 10)),
        12: ((up, 8), (down, 12), (right, 13), (left, 12)),
        13: ((up, 9), (down, 13), (right, 14), (left, 12)),
        14: ((up, 10), (down, 14), (right, 15), (left, 13)),
        15: ((up, 15), (down, 15), (right, 15), (left, 15))
    }
    for i, moves in grid_transitions.items():
        for a, j in moves:
            P[a, i, j] = 1.0

    # Rewards.
    R = -1 * sy.ones((A, S, S))
    R[:, 0, :] = 0
    R[:, 15, :] = 0

    return P, R

transitions, rewards = gridworld()

# print('transitions: {}'.format(transitions))
print('transtions shape: {}'.format(transitions.shape))
# print('rewards: {}'.format(rewards))
print('rewards shape: {}'.format(rewards.shape))
# Tensors now live on the remote workers
transitions.fix_precision().share(bob, alice)
rewards.fix_precision().share(bob, alice)

num_actions = rewards.shape[0]
num_states = rewards.shape[1]
print('Number of actions: {}'.format(num_actions))
print('Number of states: {}'.format(num_states))

# Initialize a policy to hold the optimal policy
policy = sy.zeros(num_states)
# Initialize a value function to hold the long-term value of state, s
values = sy.zeros(num_states)
policy.fix_precision().share(bob, alice)
values.fix_precision().share(bob, alice)


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

# Value Iteration
def value_iteration(values, policy, transitions, rewards, gamma, max_iter=1000, theta=0.001):
    """Solving the MDP using value iteration."""
    # http://www.incompleteideas.net/book/ebook/node44.html
    # http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MDP.pdf

    # Bellman Optimality Equation is non-linear
    # No closed form solution, need an iterative solution: Value iteration

    # check theta stopping condition
    if theta is not None:
        theta = float(theta)
        assert theta > 0, "Theta must be greater than 0."

    iteration = 0
    print('t: {}, delta: {}, V(s):\n {}'.format(iteration, None, np.reshape(list(values.cpu().numpy()), (4, 4))))

    while True:
        if iteration > max_iter:
            break
        iteration += 1

        # Stopping condition
        delta = 0

        # Update each state
        for s in range(num_states):
            # store the old state value
            old_values = values.clone()

            # Calulate the action values
            action_values = sy.zeros(num_actions)
            for a in range(num_actions):
                for s_next in range(num_states):
                    action_values[a] += transitions[a, s, s_next] * (rewards[a, s, s_next] + (gamma * values[s_next]))
            values[s] = action_values.max()

            # Calculate the delta across all seen states
            delta = max(delta, abs(old_values[s] - values[s]))
            # delta = max(delta, values.max() - old_values.min())

        # Print stats
        print('t: {}, delta: {}, ep: {}, gamma: {}, V(s):\n {}'.format(
            iteration, delta, theta, gamma, np.reshape(list(values.cpu().numpy()), (4, 4))))

        # Check if we can stop
        if delta <= (theta * (1 - gamma) / gamma):
             break

    # Create a deterministic policy using the optimal value function
    # A policy is a distribution over actions given states
    # Fully defines behaviour of an agent and is stationary
    print('\n************************')
    print('BUILD DETERMINISTIC POLICY')
    for s in range(num_states):
        # Calulate the action values
        action_values = sy.zeros(num_actions)
        for a in range(num_actions):
            for s_next in range(num_states):
                action_values[a] += transitions[a, s, s_next] * (rewards[a, s, s_next] + (gamma * values[s_next]))
        policy[s], values[s] = max(enumerate(action_values), key=lambda x: x[1])

    return values, policy


print('\n************************')
# discount factor
gamma = 1.0
values, policy = value_iteration(values, policy, transitions, rewards, gamma, max_iter=1000)
print('Optimized Values:\n {}'.format(np.reshape(list(values), (4, 4))))
print('Optimized Policy:\n {}'.format(np.reshape(list(policy), (4, 4))))