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

# create our MDP
# A forest is managed by two actions: ‘Wait’ and ‘Cut’.
# An action is decided each year with first the objective
# to maintain an old forest for wildlife and second to
# make money selling cut wood. Each year there is a
# probability p that a fire burns the forest.

# States: 
# {0, 1, ..., s-1} are states of the forest where s-1 oldest

# Actions:
# 'wait' action 0, 'cut' action 1

# after a fire the forest is in youngest state

# probability transition matrix
# shape: A x S x S
# each actions transition matrix is indexable by action
# transitions[a] returns the S x S transition matrix
transitions = Var(sy.FloatTensor(
                      [[[0.5, 0.5],
                        [0.8, 0.2]],

                       [[0. , 1.  ],
                        [0.1, 0.9]]]))

# reward matrix
# shape: S x A
rewards = Var(sy.FloatTensor(
                       [[ 5, 10],
                        [-1,  2]]))

# discount factor
gamma = sy.FloatTensor([0.96])

print('transitions: {}'.format(transitions))
print('transtions shape: {}'.format(transitions.shape))
print('rewards: {}'.format(rewards))
print('rewards shape: {}'.format(rewards.shape))

# Tensors now live on the remote workers
transitions.fix_precision().share(bob, alice)
rewards.fix_precision().share(bob, alice)

print('transitions: {}'.format(transitions))
print('rewards: {}'.format(rewards))

# Value Iteration
def value_iteration(transitions, rewards, gamma, epsilon=0.001):
    """Solving the MDP using value iteration."""
    # http://www.incompleteideas.net/book/ebook/node44.html

    num_actions = rewards.shape[0]
    num_states = rewards.shape[1]
    print('Number of actions: {}'.format(num_actions))
    print('Number of states: {}'.format(num_states))

    values = sy.zeros(num_states, 1)
    print('Initial V(s): {}'.format(values))

    iterations = 0
    while True:
        print('Iteration: {}'.format(iterations))
        iterations += 1
        values_prev = values.clone()

        delta = 0
        print('V(s): {}'.format(values_prev))
        for s in range(num_states):
            print('s: {}'.format(s))
            action_values = []
            for a in range(num_actions):
                print('a: {}'.format(a))
                total_action_value = 0
                for s_next in range(num_states):
                    total_action_value += transitions[a, s, s_next] * (rewards[a, s_next] + gamma * values[s_next])
                action_values.append(total_action_value)
                print(action_values)
                values[s] = max(action_values)
            value_diff = (values[s] - values_prev[s]).abs()[0]
            delta = max(delta, value_diff)

        if delta < epsilon * (1 - gamma) / gamma:
             return values

value_iteration(transitions, rewards, gamma)