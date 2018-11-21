import sys
import time
import random
import argparse
import itertools
import numpy as np


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
        return 'Rect({}, {}, {}, {})'.format(self.left, self.top, self.width, self.height)


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
                collided = 1
        return collided


class Paddle:
    def __init__(self):
        self.width = 800
        self.height = 10
        self.initial_x = 0
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
        self.x = random.randint(200, 600)
        self.y = HEIGHT - 200
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
        if rect.left <= self.x + self.radius and self.x - self.radius <= rect.right:
            if rect.top < self.y + self.radius < rect.bottom:
                # print('ball collide with {}'.format(collider))
                self.speedy = -self.speedy
                return True


def main(args):
    episode_returns = []
    episode_timesteps = []

    for ep in range(args.max_eps):
        paddle = Paddle()
        blocks = Blocks()

        # Initialize a moving ball
        ball = Ball(random.choice([-5, 5]), random.choice([-5, 5]))

        # cumulative reward for epiosde
        ep_return = 0

        for t in range(args.max_steps):
            if len(blocks.blocks) == 0:
                if args.verbose:
                    print('t: {}, no more blocks, end ep'.format(t))
                    reward = 0
                    ep_return += reward
                    break

            if t == args.max_steps:
                if args.verbose:
                    print('t: {}, max timesteps, end ep'.format(t))
                    break

            # Build the state
            state = (paddle.rect.left, paddle.rect.right,
                     ball.x, ball.y,
                     ball.speedx, ball.speedy,
                     len(blocks.blocks))

            # Agent selects the action
            action = random.choice([-5, 5])

            # Use the action to progress the env
            # Move the paddle according to the action selected
            paddle.move(action)

            # Move the ball according to the collision physics defined
            ball_update = ball.move()

            # Check for a collision with the paddle
            ball.collided(paddle.rect, 'paddle')

            if not ball_update:
                if args.verbose:
                    print('t: {}, ball out, end ep'.format(t))
                reward = 0
                ep_return += reward
                break

            # Check for a collision with a block
            # collect the reward
            reward = blocks.collided(ball)
            ep_return += reward

            # Increment the step counter
            t+= 1

            if args.verbose:
                print('t: {}, s: {}, a: {}, r: {}'.format(t, state, action, reward))

            time.sleep(args.step_delay)

        episode_returns.append(ep_return)
        episode_timesteps.append(t)

    print('Ep returns: {}'.format(episode_returns))
    print('Mean episode return: {}'.format(np.mean(episode_returns)))
    print('Ep time steps: {}'.format(episode_timesteps))
    print('Mean episode time steps: {}'.format(np.mean(episode_timesteps)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Arguments for Simple Breakout')
    parser.add_argument('--verbose', help='verbose logging', default=False)
    parser.add_argument('--step_delay', help='delay after each step', default=0)
    parser.add_argument('--max_eps', help='Maximum number of episodes', default=10)
    parser.add_argument('--max_steps', help='Maximum number of steps each episode', default=1000)
    parser.add_argument('--random_seed', help='Random seed', default=1017)
    args = parser.parse_args()
    random.seed(args.random_seed)
    main(args)