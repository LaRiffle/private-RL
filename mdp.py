import argparse
import numpy as np
import syft as sy
from syft import Variable as Var
from syft import nn
from syft import optim


# Value Iteration
def value_iteration(values, policy, transitions, rewards, gamma, max_iter, theta):
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
    num_actions = rewards.shape[0]
    num_states = rewards.shape[1]
    d_state = (int(np.sqrt(num_states)), int(np.sqrt(num_states)))
    print('t: {}, delta: {}, V(s):\n {}'.format(
        iteration, None, np.reshape(list(values.cpu().numpy()), d_state)))

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

            # Sum over next states, and max over actions
            values[s] = (transitions * (rewards + (gamma * values))).sum(2)[:, s].max()

            # Calculate the delta across all seen states
            delta = max(delta, abs(old_values[s] - values[s]))

        # Print stats
        print('t: {}, d: {}, g: {}, V:\n {}'.format(
            iteration, delta,
            gamma[0], np.reshape(list(values.cpu().numpy()), d_state)))

        # Check if we can stop
        if delta <= theta:
             break

    # Create a deterministic policy using the optimal value function
    # A policy is a distribution over actions given states
    # Fully defines behaviour of an agent and is stationary
    print('\n************************')
    print('BUILD DETERMINISTIC POLICY')
    for s in range(num_states):
        # Sum over next states, and max over actions
        v, i = (transitions * (rewards + (gamma * values))).sum(2)[:, s].max(0)
        values[s] = v[0]
        policy[s] = i[0]

    return values, policy

def gridworld():
    """4x4 gridworld example."""
    # number of states
    S = 16

    # number of actions
    A = 4
    # indices of the actions
    up, down, right, left = range(A)

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

def main(args):
    # this is our hook
    hook = sy.TorchHook()
    me = hook.local_worker

    bob = sy.VirtualWorker(id="bob", hook=hook)
    alice = sy.VirtualWorker(id="alice", hook=hook)

    me.add_workers([bob, alice])

    transitions, rewards = gridworld()

    # print('transitions: {}'.format(transitions))
    print('transtions shape: {}'.format(transitions.shape))
    # print('rewards: {}'.format(rewards))
    print('rewards shape: {}'.format(rewards.shape))
    # Tensors now live on the remote workers
    transitions.fix_precision().share(bob, alice)
    rewards.fix_precision().share(bob, alice)

    # assert that transitions and rewards are shaped the same
    assert rewards.shape == transitions.shape

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

    # Share theta and gamma for learning
    gamma = args.gamma * sy.ones(1)
    gamma.fix_precision().share(bob, alice)
    theta = args.theta * sy.ones(1)
    theta.fix_precision().share(bob, alice)

    # run value iteration
    values, policy = value_iteration(
        values=values,
        policy=policy,
        transitions=transitions,
        rewards=rewards,
        gamma=gamma,
        theta=theta,
        max_iter=args.max_iter)

    # print results
    print('\n************************')
    d_state = (int(np.sqrt(num_states)), int(np.sqrt(num_states)))
    print('Optimized Values:\n {}'.format(np.reshape(list(values), d_state)))
    print('Optimized Policy:\n {}'.format(np.reshape(list(policy), d_state)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'PySyft MDP Gridworld')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount factor')
    parser.add_argument('--theta', type=float, default=0.0001, help='Learning threshold')
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum number of iterations')
    args = parser.parse_args()

    main(args)