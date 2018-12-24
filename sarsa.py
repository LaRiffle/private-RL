import sys
import gym
import time
import random
import argparse
import syft as sy
import numpy as np
from gym import wrappers, logger, spaces

# Import environment classes
import secret_breakout
sys.path.append('secret_breakout')

# Import agent classes
from agents import RandomAgent, ReinforceAgent, ActorCriticAgent

class WindyGridworldEnv(gym.Env):
    def __init__(self):
        self.height = 7
        self.width = 10
        self.num_actions = 4
        self.num_states = self.height * self.width
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.height),
            spaces.Discrete(self.width)
        ))
        self.moves = {
            # up
            0: (-1, 0),
            # right
            1: (0, 1),
            # down
            2: (1, 0),
            # left
            3: (0, -1),
        }
        # Begin in the start state
        self.reset()

    def __coords_to_idx(self, coords):
        return (self.width * coords[0]) + coords[1]

    def step(self, action):
        """Take a step in the windy gridworld."""

        # Update state due to wind
        if self.state[1] in (3, 4, 5, 8):
            self.state = self.state[0] - 1, self.state[1]
        elif self.state[1] in (6, 7):
            self.state = self.state[0] - 2, self.state[1]

        # Update the state due to action
        x, y = self.moves[action]
        self.state = self.state[0] + x, self.state[1] + y

        # Enforce the boundaries
        self.state = max(0, self.state[0]), max(0, self.state[1])
        self.state = (min(self.state[0], self.height - 1),
                      min(self.state[1], self.width - 1))

        # if reached the goal state
        if self.state == (3, 7):
            return self.__coords_to_idx(self.state), 0, True, {}

        return self.__coords_to_idx(self.state), -1, False, {}

    def reset(self):
        """Reset the state of the windy gridworld."""
        self.state = (3, 0)
        return self.__coords_to_idx(self.state)

class SarsaAgent(object):
    def __init__(self, num_states, num_actions,
        epsilon=0.1, alpha=0.5, gamma=1.0):

        self.Q = sy.zeros(num_states, num_actions)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = range(num_actions)

    def act(self, state, reward, done):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            # TODO(korymath): should break ties randomly
            value, action = self.Q[state, :].max(0)
            action = action[0]
        return action

    def update_agent(self, state, action, reward, next_state, next_action):
        # Calculate the td-error
        td_error = (reward + (self.gamma * self.Q[next_state, next_action] - self.Q[state, action]))
        # Update the action-value function
        self.Q[state, action] += self.alpha * td_error
        # Update the epsilon value to decay exploration
        self.epsilon = 0.99 * self.epsilon


def main(args):
    # Make the environment.
    # env = gym.make(args.env_id)
    env = WindyGridworldEnv()

    # logging
    outdir = 'logs/window_gridworld'

    if args.monitoring:
        env = wrappers.Monitor(env,
            directory=outdir,
            video_callable=False,
            force=True)

    env.seed(args.seed)
    state = env.reset()

    # Get the action and observation space from the environment.
    logger.debug('Action space vector length: {}'.format(env.action_space.n))
    # TODO: fix this so that it can be directly read from env.observation_space
    logger.debug('Observation space vector length: {}'.format(1))
    # logger.debug('Max episode steps: {}'.format(env.spec.max_episode_steps))
    max_episode_steps = int(1e9)
    logger.debug('Max episode steps: {}'.format(max_episode_steps))

    # Build the agent
    if args.agent_id == 'sarsa':
        agent = SarsaAgent(num_states=env.num_states, num_actions=env.num_actions)

    reward = 0
    done = False

    ep_rewards = []
    ep_start_time = time.time()

    for i_episode in range(args.max_episodes + 1):
        # Don't loop forever, add one to the env_max_steps
        # to make sure to take the final step
        state = env.reset()
        # get the action from the agent
        action = agent.act(state, reward, done)
        # keep track of the performance over the episode
        single_ep_cumulative_reward = 0
        for step in range(max_episode_steps):
            # perform the action in the environment
            next_state, reward, done, info = env.step(action)
            ####
            # sarsa specific updates
            next_action = agent.act(next_state, reward, done)
            if done:
                break
            agent.update_agent(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            #####
            # track the episode performance
            single_ep_cumulative_reward += reward
            if done:
                break

        # add the accumulated reward to list of episode returns
        ep_rewards.append(single_ep_cumulative_reward)

        # update reporting times
        ep_report_time = round(time.time() - ep_start_time, 2)
        ep_start_time = time.time()

        if i_episode % args.log_interval == 0:
            logger.info('t(s): {}, ep: {}, R: {:.2f}, R_av_5: {:.2f}, i: {}'.format(
                ep_report_time, i_episode, ep_rewards[-1], np.mean(ep_rewards[-5:]), info))
    env.close()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for OpenAI Gym Runs')
    parser.add_argument('--env_id',
                        default='SecretBreakout-v0',
                        help='Environment: (SecretBreakout-v0, CartPole-v0)')
    parser.add_argument('--agent_id',
                        default='sarsa',
                        help='Agent: (random, reinforce)')
    parser.add_argument('--log_interval',
                        type=int, default=10, metavar='N',
                        help='interval between status logs (default: 10)')
    parser.add_argument('--max_episodes',
                        type=int, default=1000,
                        help='maximum number of episodes to run')
    parser.add_argument('--verbose', action='store_true',
                        help='output verbose logging for steps')
    parser.add_argument('--monitoring',
                        action='store_true',
                        help='monitor and output to log file')
    parser.add_argument('--random_action',
                        action='store_true',
                        help='Random policy for comparison')
    parser.add_argument('--random_ball_start_vel',
                        action='store_true',
                        help='Random ball starting velocity.')
    parser.add_argument('--env_width',
                        help='Environment width.',
                        default=300, type=int)
    parser.add_argument('--env_height',
                        help='Environment height.', default=400, type=int)
    parser.add_argument('--gamma',
                        type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--learning_rate',
                        type=float, default=1e-2,
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--hidden_size',
                        help='Number of units in hidden layer.',
                        default=32, type=int)
    parser.add_argument('--seed',
                        type=int, metavar='N',
                        help='random seed')
    args = parser.parse_args()
    logger.set_level(logger.INFO)

    if args.verbose:
        logger.set_level(logger.DEBUG)

    # Set the random seed if defined
    if args.seed:
        random.seed(args.seed)

    # Run the training
    main(args)