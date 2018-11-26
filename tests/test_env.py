import argparse
import yaml
import sys
import torch
import unittest
sys.path.append('secret_breakout')
from env import CorruptBreakoutEnv, TorchBox
from gym import spaces


class TestSpaces(unittest.TestCase):
    def setUp(self):
        stargs_filename = "./tests/fixtures/env_args.yaml"
        self.env = _build_env_from_file(stargs_filename)
        highs = (150, 300, 400, 5, 5, 12,
                 300, 80, 300, 80, 300, 80,
                 300, 80, 300, 80, 300, 80,
                 300, 80, 300, 80, 300, 80,
                 300, 80, 300, 80, 300, 80)
        lows = tuple(0 for _ in range(len(highs)))
        self.obsFix = TorchBox(lows, highs, dtype=torch.FloatTensor)
        self.actFix = spaces.Discrete(2)

    def testActionSpace(self):
        self.env.action_space == self.actFix

    def testObservationSpace(self):
        self.env.observation_space == self.obsFix


class TestStateFunc(unittest.TestCase):
    def setUp(self):
        stargs_filename = "./tests/fixtures/env_args.yaml"
        self.env = _build_env_from_file(stargs_filename)
        self.state = self.env.reset()
        self.ballspdx, self.ballspdy = self.state[3], self.state[4]
        self.init_state = torch.FloatTensor([75, 150, 330,
                                             self.ballspdx, self.ballspdy, 0,
                                             0, 0, 0, 20, 0, 40, 0, 60,
                                             100, 0, 100, 20, 100, 40, 100, 60,
                                             200, 0, 200, 20, 200, 40, 200, 60])

    def testReset(self):
        assert (self.state == self.init_state).all()


class TestRewardFunc(unittest.TestCase):
    def setUp(self):
        stargs_filename = "./tests/fixtures/env_args.yaml"
        self.env = _build_env_from_file(stargs_filename)
        self.env.reset()

    def testRewardPenalty(self):
        self.env.ball.speedx = -5
        self.env.ball.speedy = 5
        for _ in range(100):
            paddle_penalty = self.env.paddle.move(1)
            dead_ball = self.env.ball.move()
            if dead_ball:
                break
        r, _r = self.env._compute_reward(False, dead_ball)
        exp = -self.env.hidden_paddle_penalty
        assert r == 0, "Expected r = 0, got {}".format(r)
        assert _r == exp, "Expected _r = {}, got {}".format(exp, _r)

    def testRewardDeath(self):
        self.env.ball.speedx = -5
        self.env.ball.speedy = 5
        for i in range(500):
            if i % 2 == 0:
                action = 0
            else:
                action = 1
            paddle_penalty = self.env.paddle.move(action)
            dead_ball = self.env.ball.move()
            if dead_ball:
                break
        r, _r = self.env._compute_reward(dead_ball, paddle_penalty)
        exp = -self.env.death_penalty
        assert r == exp, "Expected r = {}, got {}".format(exp, r)
        assert _r == r, "Expected _r = {}, got {}".format(exp, _r)

    def testRewardBlock(self):
        self.env.ballspeedy = -5
        self.env.ballspeedx = 0
        i = 0
        while True:
            i += 1
            paddle_penalty = self.env.paddle.move(1)
            dead_ball = self.env.ball.move()
            r, _r = self.env._compute_reward(dead_ball, paddle_penalty)
            if self.env.blocks.num_blocks_destroyed > 0:
                break
        assert r == 5, "Expected r = 5, got {}".format(r)
        assert _r == r, "Expected _r = 5, got {}".format(_r)


def _build_env_from_file(filename):
    with open(filename) as f_yaml:
        args_dict = yaml.load(f_yaml)
    args = MockArgParser(**args_dict)
    return CorruptBreakoutEnv(args)


class MockArgParser(object):
    """Mocks up the ArgumentParser object"""
    def __init__(self, **kwargs):
        super(MockArgParser, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


if __name__ == '__main__':
    unittest.main()
