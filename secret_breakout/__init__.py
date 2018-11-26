import yaml
from gym.envs.registration import register

class MockArgParser(object):
    """Mocks up the ArgumentParser object"""
    def __init__(self, **kwargs):
        super(MockArgParser, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

filename = "./tests/fixtures/env_args.yaml"
with open(filename) as f_yaml:
    args_dict = yaml.load(f_yaml)
args = MockArgParser(**args_dict)

register(
    id='SecretBreakout-v0',
    entry_point='secret_breakout.env:CorruptBreakoutEnv',
    kwargs={'args': args}
)