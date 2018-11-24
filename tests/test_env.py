import argparse
import unittest
import sys
import torch
sys.path.append('secret_breakout')
from env import CorruptBreakoutEnv, TorchBox
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

class TestStateFunctionality(unittest.TestCase):
    def setUp(self):
        self.state, self.reward, self.done, self.info = env.reset()
        self.ballspdx, self.ballspdy = self.state[3], self.state[4]
        self.init_state = torch.FloatTensor([75, 150, 330,
                                             self.ballspdx, self.ballspdy, 0,
                                             0, 0, 0, 20, 0, 40, 0, 60,
                                             100, 0, 100, 20, 100, 40, 100, 60,
                                             200, 0, 200, 20, 200, 40, 200, 60])

    def testReset(self):
        assert (self.state == self.init_state).all()
        assert self.reward == 0
        assert self.done is False
        assert self.info["hidden_reward"] == 0

class TestRewardFunctionality(unittest.TestCase):
    def setUp(self):
        env.reset()

    def testRewardPenalty(self):
        env.ball.speedx = -5
        env.ball.speedy = 5
        for _ in range(100):
            paddle_penalty = env.paddle.move(1)
            dead_ball = env.ball.move()
            if dead_ball:
                break
        r, _r = env._compute_reward(False, dead_ball)
        assert r == 0
        assert _r == -env.hidden_paddle_penalty

    def testRewardDeath(self):
        env.ball.speedx = -5
        env.ball.speedy = 5
        for i in range(500):
            if i % 2 == 0:
                action = 0
            else:
                action = 1
            paddle_penalty = env.paddle.move(action)
            dead_ball = env.ball.move()
            if dead_ball:
                break
        r, _r = env._compute_reward(dead_ball, paddle_penalty)
        assert r == -env.death_penalty
        assert _r == r

    def testRewardBlock(self):
        env.ballspeedy = -5
        for _ in range(100):
            paddle_penalty = env.paddle.move(1)
            dead_ball = env.ball.move()
            print(env.blocks.num_blocks_destroyed > 0)
            if env.blocks.num_blocks_destroyed > 0:
                break
        r, _r = env._compute_reward(False, False)
        print(r)
        assert r == 5
        assert _r == r




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
    args = parser.parse_args(["--env_width", "300", "--env_height", "400", "--verbose"])

    env = CorruptBreakoutEnv(args)

    unittest.main()
