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
# shape: A x S x S
# each actions transition matrix is indexable by action
# transitions[a] returns the S x S transition matrix
# Defines the transition probabilities from all states (rows)
# to all successor states (columns), each row sums to 1
transitions = sy.FloatTensor(
                      [[[0.5, 0.5],
                        [0.8, 0.2]],

                       [[0. , 1.  ],
                        [0.1, 0.9]]])

# reward matrix
# shape: S x A (rows x columns)
rewards = sy.FloatTensor([[ 5, 10],
                          [-1,  2]])

# discount factor
gamma = 0.96

print('transitions: {}'.format(transitions))
print('transtions shape: {}'.format(transitions.shape))
print('rewards: {}'.format(rewards))
print('rewards shape: {}'.format(rewards.shape))

# Tensors now live on the remote workers
# transitions.fix_precision().share(bob, alice)
# rewards.fix_precision().share(bob, alice)

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
    print('Initial V(s): {}'.format(values))

    iteration = 0
    while True:
        if iteration >= max_iter:
            break

        print('Iteration: {}'.format(iteration))
        print('V(s): {}'.format(values))
        iteration += 1
        # Stopping condition
        delta = 0

        # BELLMAN OPERATION ON THE VALUE FUNCTION
        # action_values = sy.zeros(num_actions, num_states)
        # for a in range(num_actions):
        #     action_values[a] = rewards[] + gamma * transitions[a].dot(values)
        # print('bellman Q {}'.format(action_values))

        old_values = values.clone()

        # Update each state
        for s in range(num_states):
            # Find the best action
            action_values = sy.zeros(num_actions, 1)
            Q = [rewards[a][s] + gamma * transitions[a][s, :].dot(values) for a in range(num_actions)]
            print('Q: {}'.format(Q))
            values[s] = max(Q)

            # for a in range(num_actions):
            #     action_value = 0
            #     for s_next in range(num_states):
            #         action_value += transitions[a, s, s_next] * (rewards[s_next, a] + gamma * values[s_next])
            #     action_values[a] = action_value
            # print('s: {}, action values: {}'.format(s, action_values))
            # best_action_value = action_values.max()

            # Calculate the delta across all seen states
            delta = max(delta, (values.max() - old_values.min()))
            # update the value function
            # values[s] = best_action_value
        print('t: {}, delta: {}, ep: {}, gamma: {}'.format(iteration, delta, theta, gamma))
        # Check if we can stop
        if delta < (theta * (1 - gamma) / gamma):
             break

    print('\n************************')
    print('BUILD DETERMINISTIC POLICY')
    # Create a deterministic policy using the optimal value function
    # A policy is a distribution over actions given states
    # Fully defines behaviour of an agent and is stationary
    policy = []
    for s in range(num_states):
        # Find the best action
        Q = sy.zeros(num_states, 1)
        for a in range(num_actions):
            Q[a] = (rewards[a][s] + gamma * transitions[a][s, :].dot(values))

        values[s] = Q.max()
        policy.append(Q.argmax())

        # action_values = sy.zeros(num_actions, 1)
        # for a in range(num_actions):
        #     action_value = 0
        #     for s_next in range(num_states):
        #         action_value += transitions[a, s, s_next] * (rewards[s_next, a] + gamma * values[s_next])
        #     action_values[a] = action_value
        # # print('action values: {}'.format(action_values))
        # # Argmax to get the maximizing action
        # best_action = action_values.max(0)[1]
        # # Always take the best action
        # policy[s] = best_action

    return values, policy


print('\n************************')
values, policy = value_iteration(transitions, rewards, gamma, max_iter=10)
print('Optimized value function:')
print('Values: {}'.format(values))
print('Optimized policy:')
print('Policy: {}'.format(policy))


# Must make sure that it is a valid MDP