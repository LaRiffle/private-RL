import random
import torch
from breakout import Ball, Blocks, Paddle, Rect
from gym import Env, Space, spaces, logger


class TorchBreakoutEnv(Env):
    """
    Wraps the Breakout materials into the Gym environment spec.
    """
    def __init__(self, args, dtype=torch.FloatTensor):
        super(TorchBreakoutEnv, self).__init__()

        ### Build Paddle, Blocks, and Ball
        self.paddle = Paddle(args)
        self.blocks = Blocks(args)
        spd_init = (random.choice([-5, 5]), random.choice([-5, 5]))
        self.ball = Ball(args, *spd_init)

        # Calculate spec for observation space
        paddle_spec = (0, args.env_width - self.paddle.width)   # paddle location
        ballx_spec = (0, args.env_width)                        # ball location x
        bally_spec = (0, args.env_height)                       # ball location y
        speedx_spec = (0, self.ball.speed_magnitude)            # ball speed x
        speedy_spec = (0, self.ball.speed_magnitude)            # ball speed y
        dblocks_spec = (0, self.blocks.num_blocks_start)        # num_destroyed_blocks
        nonloc_spec = [paddle_spec,
                       ballx_spec,
                       bally_spec,
                       speedx_spec,
                       speedy_spec,
                       dblocks_spec]
        blocs_spec = [(0, self.blocks.args.env_width),  # block locations x
                      (0, self.blocks.rows_height)]     # block locations y
        blocs_spec *= len(self.blocks.blocks)           # extend by max number of blocks
        spec = nonloc_spec + blocs_spec
        self.observation_space = TorchBox(*zip(*spec), dtype=dtype)
        self.action_space = spaces.Discrete(2)


class TorchBox(Space):
    """
    A box in R^n.
    I.e., each coordinate is bounded.
    Backed by torch.Tensor instead of np.ndarray.
    dtype is a kind of Torch Tensor.

    Example usage:
    self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
    """
    def __init__(self, low=None, high=None, shape=None, dtype=None):
        """
        Two kinds of valid input:
            Box(low=-1.0, high=1.0, shape=(3,4)) # low and high are scalars, and shape is provided
            Box(low=torch.Tensor([-1.0,-2.0]), high=torch.Tensor([2.0,4.0])) # low and high are arrays of the same shape
        """
        if dtype is None:  # Autodetect type
            if (torch.FloatTensor(high) == 255).all():  # hack
                Tensor = torch.ByteTensor
            else:
                Tensor = torch.FloatTensor
            logger.warn("gym.spaces.Box autodetected dtype as %s. Please provide explicit dtype." % Tensor)
        else:
            Tensor = dtype
        if isinstance(low, tuple) and isinstance(high, tuple):
            low = Tensor(low)
            high = Tensor(high)
        elif shape is None:
            assert low.shape == high.shape
            shape = low.shape
        else:
            assert np.isscalar(low) and np.isscalar(high)
            low = low + torch.zeros(shape)
            high = high + torch.zeros(shape)
        self.low = low.type(Tensor)
        self.high = high.type(Tensor)
        self.shape = shape
        self.Tensor = Tensor
        if self.Tensor in ["torch.HalfTensor", "torch.FloatTensor", "torch.DoubleTensor"]:
            self.dtype = "float"
        else:
            self.dtype = "int"

    def sample(self):
        raise NotImplementedError

    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()

    def to_jsonable(self, sample_n):
        raise NotImplementedError

    def from_jsonable(self, sample_n):
        raise NotImplementedError

    def __repr__(self):
        return "TorchBox" + str(self.shape)

    def __eq__(self, other):
        return (self.low == other.low).all() and (self.high == other.high).all()
