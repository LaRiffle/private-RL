import sys
import argparse

import gym
from gym import wrappers, logger


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()
    logger.set_level(logger.DEBUG)

    env = gym.make(args.env_id)

    # logging and monitoring
    outdir = 'logs/gym_random_agent_results'
    env = wrappers.Monitor(env, directory=outdir, video_callable=False, force=True)

    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    done = False
    reward = 0

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
    env.close()