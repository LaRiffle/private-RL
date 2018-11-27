import sys
import gym
import time
import argparse
import numpy as np
from gym import wrappers, logger

# Import environment classes
import secret_breakout
sys.path.append('secret_breakout')

# Import agent classes
from agents import RandomAgent, ReinforceAgent, ActorCriticAgent


def main(args):
    # Make the environment.
    env = gym.make(args.env_id)

    # logging
    outdir = 'logs/secret_breakout'

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
    logger.debug('Observation space vector length: {}'.format(len(state)))
    logger.debug('Max episode steps: {}'.format(env.spec.max_episode_steps))

    # Build the agent
    if args.agent_id == 'random':
        agent = RandomAgent(env.action_space)
    elif args.agent_id == 'reinforce':
        agent = ReinforceAgent(input_size=len(state),
                               hidden_size=args.hidden_size,
                               output_size=env.action_space.n,
                               learning_rate=args.learning_rate,
                               gamma=args.gamma)
    elif args.agent_id == 'ac':
        agent = ActorCriticAgent(input_size=len(state),
                            hidden_size=args.hidden_size,
                            output_size=env.action_space.n,
                            learning_rate=5e-3,
                            gamma=args.gamma)

    reward = 0
    done = False

    ep_rewards = []
    ep_start_time = time.time()

    for i_episode in range(args.max_episodes + 1):
        # Don't loop forever, add one to the env_max_steps
        # to make sure to take the final step
        state = env.reset()

        # keep track of the performance over the episode
        single_ep_cumulative_reward = 0
        for step in range(env.spec.max_episode_steps):
            # get the next action from the agent
            action = agent.act(state, reward, done)
            # perform the action in the environment
            state, reward, done, _ = env.step(action)
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
            logger.info('t(s): {}, ep: {}, R: {:.2f}, R_av_5: {:.2f}'.format(
                ep_report_time, i_episode,
                ep_rewards[-1], np.mean(ep_rewards[-5:])))
    env.close()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for OpenAI Gym Runs')
    parser.add_argument('--env_id',
                        default='SecretBreakout-v0',
                        help='Environment: (SecretBreakout-v0, CartPole-v0)')
    parser.add_argument('--agent_id',
                        default='random',
                        help='Agent: (random, reinforce)')
    parser.add_argument('--log_interval',
                        type=int, default=10, metavar='N',
                        help='interval between status logs (default: 10)')
    parser.add_argument('--max_episodes',
                        type=int, default=500,
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
        logger.set_level(logger.INFO)

    # Set the random seed if defined
    if args.seed:
        random.seed(args.seed)

    # Run the training
    main(args)