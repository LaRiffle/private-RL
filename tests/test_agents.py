import sys
import random
import unittest

sys.path.append('secret_breakout')
from gym import spaces
from agents import RandomAgent

random.seed(1017)

class TestRandomAgent(unittest.TestCase):
    def setUp(self):
        number_of_actions = 2
        action_space = spaces.Discrete(2)
        self.agent = RandomAgent(action_space)

    def testAction(self):
        action = self.agent.act(state=None, reward=None, done=None)
        assert action == 0

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
