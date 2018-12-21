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

# discount factor
gamma = 0.96

print('transitions: {}'.format(transitions))
print('transtions shape: {}'.format(transitions.shape))
print('rewards: {}'.format(rewards))
print('rewards shape: {}'.format(rewards.shape))

# Tensors now live on the remote workers
# transitions.fix_precision().share(bob, alice)
# rewards.fix_precision().share(bob, alice)


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

# Value Iteration
def value_iteration(transitions, rewards, gamma, max_iter=1000, theta=0.01):
    """Solving the MDP using value iteration."""
    # http://www.incompleteideas.net/book/ebook/node44.html
    # http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MDP.pdf

    # Bellman Optimality Equation is non-linear
    # No closed form solution, need an iterative solution: Value iteration

    # check theta stopping condition
    if theta is not None:
        theta = float(theta)
        assert theta > 0, "Theta must be greater than 0."

    num_actions = rewards.shape[0]
    num_states = rewards.shape[1]
    print('Number of actions: {}'.format(num_actions))
    print('Number of states: {}'.format(num_states))

    # Initialize a value function to hold the long-term value of state s
    values = sy.zeros(num_states)

    iteration = 0
    print('t: {}, delta: {}, V(s): {}'.format(
            iteration, None, list(values)))

    while True:
        if iteration > max_iter:
            break
        iteration += 1

        # Stopping condition
        delta = 0

        # Update each state
        for s in range(num_states):
            # store the old state value
            old_value = values[s]

            # Calulate the action values
            Q = [sum(transitions[a][s, :] * (rewards[a][s, :] + gamma * values[:])) for a in range(num_actions)]
            values[s] = max(Q)

            # Calculate the delta across all seen states
            delta = max(delta, abs(old_value - values[s]))

        # Print stats
        print('t: {}, delta: {}, ep: {}, gamma: {}, V(s): {}'.format(
            iteration, delta, theta, gamma, values.cpu().numpy()))

        # Check if we can stop
        if delta < (theta * (1 - gamma) / gamma):
             break

    # Create a deterministic policy using the optimal value function
    # A policy is a distribution over actions given states
    # Fully defines behaviour of an agent and is stationary
    policy = sy.zeros(num_states)

    print('\n************************')
    print('BUILD DETERMINISTIC POLICY')
    for s in range(num_states):
        # Calulate the action values
        Q = [sum(transitions[a][s, :] * (rewards[a][s, :] + gamma * values[:])) for a in range(num_actions)]
        values[s] = max(Q)
        policy[s] = argmax(Q)

    return values, policy


print('\n************************')
values, policy = value_iteration(transitions, rewards, gamma, max_iter=100)
print('Optimized Values: {}'.format(values))
print('Optimized Policy: {}'.format(policy))