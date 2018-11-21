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

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size, bias=True)
        self.affine2 = nn.Linear(hidden_size, output_size, bias=True)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        x = self.affine2(x)
        action_scores = F.softmax(x, dim=0)
        return action_scores

hidden_size = 4
learning_rate = 1e-3
input_size = 7
output_size = 2
policy = Policy(input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
eps = np.finfo(np.float32).eps.item()


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
    # rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
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
HEIGHT = 600
WIDTH = 800

class Rect:
    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.right = left + width
        self.bottom = top + height

    def move(self, x):
        return Rect(self.left+x, self.top, self.width, self.height)

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

    def make_blocks(self):
        rects = []
        rows = 5
        rows_height = 120
        for i in range(0, rows_height, self.height):
            for j in range(0, WIDTH, self.width):
                rects.append(Rect(j, i, self.width, self.height))
        return rects

    # removes single block from blocks list when it is hit by ball
    # ball being the ball object
    def collided(self, ball_object):
        collided = 0
        for block in self.blocks:
            if ball_object.collided(block, 'block'):
                self.blocks.remove(block)
                collided = 100
        return collided

class Paddle:
    def __init__(self):
        self.width = 150
        self.height = 10
        self.initial_x = 400 # random.randint(0, 650)
        self.initial_y = HEIGHT - 50
        self.rect = Rect(self.initial_x, self.initial_y, 
                         self.width, self.height)

    def move(self, speed):
        if self.rect.right + speed > WIDTH or self.rect.left + speed < 0:
            # out of bounds, do not update the paddle position
            return
        else:
            # update the paddel position
            self.rect = self.rect.move(speed)

class Ball:
    """Ball object that takes initial speed in x direction (speedx)
        and initial speed in y direction(speedy)"""
    def __init__(self, speedx, speedy):
        self.x = 300 # random.randint(200, 400)
        self.y = 300 # HEIGHT - random.randint(200, 400)
        self.radius = 10
        self.speed_magnitude = 5
        self.speedx = speedx
        self.speedy = speedy

    def move(self):
        # check for collision with the right side of the game screen
        if self.x + self.radius + self.speedx >= WIDTH:
            # print('collision with right side of screen')
            self.speedx = -self.speedx

        # check for collision with the left hand side of the game screen
        elif self.x + self.speedx <= 0:
            # print('collision with left side of screen')
            self.speedx = self.speed_magnitude

        # check for collision with the top of the game screen
        if self.y + self.radius + self.speedy >= HEIGHT:
            # print('collision with bottom of screen')
            return False

        # check for collision with the bottom of the game screen
        elif self.y + self.radius + self.speedy <= 0:
            # print('collision with top of screen')
            self.speedy = -self.speedy

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
                # print('ball collide with {}'.format(collider))
                self.speedy = -self.speedy
                return True


def main(args):
    episode_returns = []
    episode_timesteps = []

    for ep in range(args.max_episodes):
        # make the game, env.reset()
        paddle = Paddle()
        blocks = Blocks()

        # Initialize a moving ball
        # ball = Ball(random.choice([-5, 5]), random.choice([-5, 5]))
        ball = Ball(5, 5)

        # start a timer for time checking
        last_time = time.time()
        for t in range(args.env_max_steps):
            # check the state of the env and break if necessary
            if len(blocks.blocks) == 0:
                if args.verbose:
                    print('t: {}, no more blocks, end ep'.format(t))
                    reward = 0
                    break

            if t == args.env_max_steps:
                if args.verbose:
                    print('t: {}, max timesteps, end ep'.format(t))
                    reward = 0
                    break

            # Build the state
            state = torch.Tensor([
                     paddle.rect.left/float(WIDTH),
                     paddle.rect.right/float(WIDTH),
                     ball.x/float(WIDTH),
                     ball.y/float(HEIGHT),
                     ball.speedx/float(5),
                     ball.speedy/float(5),
                     len(blocks.blocks)])

            # Agent selects the action

            ## RANDOM ACTION SELECTION
            # action = random.choice([-5, 5])

            ## REINFORCE ACTION SELECTION
            actions = [-5, 5]
            action_temp = select_action(state)
            action = actions[np.argmax(action_temp)]
            # action = random.choice(actions)

            # Use the action to progress the env
            # Move the paddle according to the action selected
            paddle.move(action)

            # Move the ball according to the collision physics defined
            ball_update = ball.move()

            # Check for a collision with the paddle
            ball.collided(paddle.rect, 'paddle')

            if not ball_update:
                reward = -10
                policy.rewards.append(reward)
                if args.verbose:
                    print('t: {}, s: {}, a: {}, r: {}'.format(
                        t, state.numpy(), action, reward))
                break

            # Check for a collision with a block
            # collect the reward
            base_reward = 1
            bonus_reward = blocks.collided(ball)
            reward = bonus_reward + base_reward
            policy.rewards.append(reward)

            if args.verbose:
                print('t: {}, s: {}, a: {}, r: {}'.format(
                    t, state.numpy(), action, reward))

            # sleep if necessary
            if args.step_delay:
                time.sleep(args.step_delay)

            # Increment the step counter
            t += 1

        episode_returns.append(np.sum(policy.rewards))
        episode_timesteps.append(t)
        # calculate the policy loss, update the model
        # clear saved rewards and log probs
        policy_loss = finish_episode()
        if ep % args.log_interval == 0:
            print('ep: {}, t: {}, L: {}, R: {:.2f}, R_av_5: {:.2f}'.format(
                ep, t, round(policy_loss.data[0], 2),
                episode_returns[-1], np.mean(episode_returns[-5:])))

    # print('Ep returns: {}'.format(episode_returns))
    # print('Mean episode return: {}'.format(np.mean(episode_returns)))
    # print('Ep time steps: {}'.format(episode_timesteps))
    # print('Mean episode time steps: {}'.format(np.mean(episode_timesteps)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Arguments for Simple Breakout')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='interval between status logs (default: 10)')
    parser.add_argument('--max_episodes', type=int, default=1,
                        help='maximum number of episodes to run')
    parser.add_argument('--verbose', action='store_true', 
        help='output verbose logging for steps')
    parser.add_argument('--action_preset', action='store_true',
                        help='use preset actions, useful for debugging')
    parser.add_argument('--step_delay',
                        help='delay after each step, useful for debugging',
                        default=0)
    parser.add_argument('--env_max_steps',
                        help='Max steps each episode', default=1000)
    parser.add_argument('--gamma', type=float, default=0.97, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=1017, metavar='N',
                        help='random seed (default: 1017)')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='number of runs to perform')
    parser.add_argument('--exp_name_prefix', type=str, 
                        default='default_exp_name_prefix',
                        help='prefix to name of experiment')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    main(args)