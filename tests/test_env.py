import argparse
import unittest
import sys
import torch
sys.path.append('secret_breakout')
from env import TorchBreakoutEnv, TorchBox
from gym import spaces


class TestSpaces(unittest.TestCase):
    def setUp(self):
        highs = (150, 300, 400, 5, 5, 12,
                 300, 80, 300, 80, 300, 80,
                 300, 80, 300, 80, 300, 80,
                 300, 80, 300, 80, 300, 80,
                 300, 80, 300, 80, 300, 80)
        lows = tuple(0 for _ in range(len(highs)))
        self.obsFix = TorchBox(lows, highs, dtype=torch.FloatTensor)
        self.actFix = spaces.Discrete(2)

    def testActionSpace(self):
        env.action_space == self.actFix

    def testObservationSpace(self):
        env.observation_space == self.obsFix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Arguments for Simple Breakout')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='interval between status logs (default: 10)')
    parser.add_argument('--max_episodes', type=int, default=500,
                        help='maximum number of episodes to run')
    parser.add_argument('--verbose', action='store_true',
                        help='output verbose logging for steps')
    parser.add_argument('--random_action', action='store_true',
                        help='Random policy for comparison')
    parser.add_argument('--random_ball_start_vel', action='store_true',
                        help='Random ball starting velocity.')
    parser.add_argument('--env_width',
                        help='Environment width.', default=300, type=int)
    parser.add_argument('--env_height',
                        help='Environment height.', default=400, type=int)
    parser.add_argument('--env_max_steps',
                        help='Max steps each episode', default=1000)
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, metavar='N',
                        help='random seed')
    args = parser.parse_args(["--env_width", "300", "--env_height", "400"])

    env = TorchBreakoutEnv(args)

    unittest.main()
