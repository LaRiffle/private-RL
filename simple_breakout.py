import sys
import time
import random
import argparse
import itertools
import numpy as np

# PyTorch Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from collections import namedtuple


### AGENT ###
class Normalizer():
    def __init__(self, input_size):
        self.n = torch.zeros(input_size)
        self.mean = torch.zeros(input_size)
        self.mean_diff = torch.zeros(input_size)
        self.var = torch.zeros(input_size)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean)/obs_std

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size, bias=False)
        self.affine2 = nn.Linear(hidden_size, output_size, bias=False)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        x = self.affine2(x)
        action_scores = F.softmax(x, dim=0)
        return action_scores

hidden_size = 32
learning_rate = 1e-2
input_size = 30
output_size = 2
policy = Policy(input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
eps = np.finfo(np.float32).eps.item()
normalizer = Normalizer(input_size=input_size)


def select_action(state):
    probs = policy(Variable(state))
    m = Categorical(probs)
    selected_action = m.sample()
    action = torch.Tensor([0, 0])
    action[selected_action.data] = 1
    log_prob = m.log_prob(selected_action)
    policy.saved_log_probs.append(log_prob)
    return action

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    return policy_loss


### ENV #####
HEIGHT = 300
WIDTH = 400

class Rect:
    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.right = left + self.width
        self.bottom = top + self.height

    def move(self, x):
        return Rect(self.left+x, self.top, self.width, self.height)

    def destroyed(self):
        self.left = -1
        self.top = -1
        self.width = -1
        self.height = -1
        self.right = -1
        self.bottom = -1

    def __repr__(self):
        return 'Rect({}, {}, {}, {})'.format(
            self.left, self.top, self.width, self.height)

class Blocks:
    """Implements blocks as a collection  instead of
        as individual block objects """
    def __init__(self):
        self.width = 100
        self.height = 20
        self.blocks = self.make_blocks()
        self.num_blocks_start = len(self.blocks)
        self.num_blocks_destroyed = 0

    def make_blocks(self):
        rects = []
        rows = 5
        rows_height = HEIGHT//rows
        for i in range(0, rows_height, self.height):
            for j in range(0, WIDTH, self.width):
                rects.append(Rect(j, i, self.width, self.height))
        return rects

    # removes single block from blocks list when it is hit by ball
    # ball being the ball object
    def collided(self, ball_object):
        collided = False
        for block in self.blocks:
            if ball_object.collided(block, 'block'):
                # self.blocks.remove(block)
                # set the block to destroyed if collision occured
                block.destroyed()
                collided = True
                self.num_blocks_destroyed += 1
        return collided

    def block_locations(self):
        block_locs = [-1] * (2*self.num_blocks_start)
        for i,block in enumerate(self.blocks):
            block_locs[2*i] = block.left
            block_locs[(2*i)+1] = block.top
        return block_locs

class Paddle:
    def __init__(self):
        self.width = WIDTH // 2
        self.height = 20
        self.initial_x = (WIDTH//2) - (self.width//2)
        self.initial_y = HEIGHT - 50
        self.rect = Rect(self.initial_x, self.initial_y, 
                         self.width, self.height)

    def move(self, speed):
        if self.rect.right + speed > WIDTH or self.rect.left + speed < 0:
            # out of bounds, do not update the paddle position
            # print('paddle collide with side of screen')
            return
        else:
            # update the paddel position
            self.rect = self.rect.move(speed)

class Ball:
    """Ball object that takes initial speed in x direction (speedx)
        and initial speed in y direction(speedy)"""
    def __init__(self, args, speedx, speedy):
        self.radius = 10
        self.x = WIDTH//2
        self.y = HEIGHT - 70
        self.speed_magnitude = 5
        self.speedx = speedx
        self.speedy = speedy
        self.args = args

    def move(self):
        # check for collision with the right side of the game screen
        if self.x + self.radius + self.speedx >= WIDTH:
            if args.verbose:
                print('ball collide with right side of screen')
            self.speedx = -self.speed_magnitude

        # check for collision with the left hand side of the game screen
        elif self.x + self.speedx <= 0:
            if args.verbose:
                print('ball collide with left side of screen')
            self.speedx = self.speed_magnitude

        # check for collision with the bottom of the game screen
        if self.y + self.radius + self.speedy >= HEIGHT:
            if args.verbose:
                print('ball collide with bottom of screen')
            self.speedy = -self.speed_magnitude
            return False

        # check for collision with the top of the game screen
        elif self.y + self.radius + self.speedy <= 0:
            if args.verbose:
                print('ball collide with top of screen')
            self.speedy = self.speed_magnitude

        # update the ball position
        self.x += self.speedx
        self.y += self.speedy
        return True

    # checks if ball has collided with the rect
    # which may be rect of block or paddle
    def collided(self, rect, collider):
        if ((rect.left <= self.x + self.radius) and
            (self.x - self.radius <= rect.right)):
            if rect.top < self.y + self.radius < rect.bottom:
                if args.verbose:
                    print('ball collide with {}'.format(collider))
                self.speedy = -self.speedy
                # add an extra displacement to avoid double collision
                self.y += self.speedy
                return True


def main(args):
    episode_returns = []
    episode_timesteps = []

    for ep in range(args.max_episodes):
        # make the game, env.reset()
        paddle = Paddle()
        blocks = Blocks()

        # Initialize a moving ball
        # ball = Ball(args, random.choice([-5, 5]), random.choice([-5, 5]))
        ball = Ball(args, 5, 5)

        # start a timer for time checking
        last_time = time.time()
        for t in range(args.env_max_steps):
            # check the state of the env and break if necessary
            if len(blocks.blocks) == 0:
                if args.verbose:
                    print('t: {}, no more blocks, end ep'.format(t))
                    reward = 0
                    policy.rewards.append(reward)
                    break

            if t == args.env_max_steps:
                if args.verbose:
                    print('t: {}, max timesteps, end ep'.format(t))
                    reward = 0
                    policy.rewards.append(reward)
                    break

            # BUILD THE STATE
            block_locs = blocks.block_locations()
            state_temp = [
                     paddle.rect.left,
                     ball.x,
                     ball.y,
                     ball.speedx,
                     ball.speedy,
                     blocks.num_blocks_destroyed]
            state_temp.extend(block_locs)

            if args.verbose:
                print('b: {},{}, p: {}'.format(
                    ball.x, ball.y, paddle.rect.left))

            state = torch.Tensor(state_temp)

            # normalize the state
            if True:
                normalizer.observe(state)
                state = normalizer.normalize(state)


            # Agent selects the action
            ## REINFORCE ACTION SELECTION
            actions = [-5, 5]
            action_temp = select_action(state)
            action = actions[np.argmax(action_temp)]

            if args.random_action:
                ## RANDOM ACTION SELECTION
                action = random.choice([-5, 5])

            # Use the action to progress the env
            # Move the paddle according to the action selected
            paddle.move(action)

            # Move the ball according to the collision physics defined
            ball_update = ball.move()

            if not ball_update:
                reward = -100
                policy.rewards.append(reward)
                if args.verbose:
                    print('t: {}, a: {}, r: {}'.format(t, action, reward))
                break

            # Check for a collision with a block
            # collect the reward
            base_reward = 1
            if blocks.collided(ball):
                bonus_block = 5
            else:
                bonus_block = 0

            # Check for a collision with the paddle
            if ball.collided(paddle.rect, 'paddle'):
                bonus_paddle = 2
            else:
                bonus_paddle = 0

            reward = base_reward + bonus_paddle + bonus_block
            policy.rewards.append(reward)

            if args.verbose:
                print('t: {}, a: {}, r: {}'.format(t, action, reward))

            # Increment the step counter
            t += 1

        episode_returns.append(np.sum(policy.rewards))
        episode_timesteps.append(t)

        # calculate the policy loss, update the model
        # clear saved rewards and log probs
        policy_loss_temp = finish_episode()
        policy_loss = policy_loss_temp.data[0]
        # policy_loss = 0

        if ep % args.log_interval == 0:
            print('ep: {}, b: {}, t: {}, L: {}, R: {:.2f}, R_av_5: {:.2f}'.format(
                ep, blocks.num_blocks_destroyed, t, round(policy_loss, 2),
                episode_returns[-1], np.mean(episode_returns[-5:])))


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
    parser.add_argument('--env_max_steps',
                        help='Max steps each episode', default=500)
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, metavar='N',
                        help='random seed')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    main(args)