import sys
import time
import random
import itertools

HEIGHT = 600
WIDTH = 800


class Rect:
    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.right = left + width
        self.bottom = top - height

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
            if ball_object.collided(block):
                self.blocks.remove(block)
                collided = 1
                sys.exit(0)
        return collided


class Paddle:
    def __init__(self):
        self.width = 150
        self.height = 10
        self.initial_x = 325
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
        self.x = 400 # WIDTH//2
        self.y = HEIGHT - 60
        self.radius = 10
        self.speed_magnitude = 5
        self.speedx = speedx
        self.speedy = speedy

    def move(self):
        # check for collision with the right side of the game screen
        if self.x + self.radius + self.speedx >= WIDTH:
            self.speedx = -self.speedx

        # check for collision with the left hand side of the game screen
        elif self.x + self.speedx <= 0:
            self.speedx = self.speed_magnitude

        # check for collision with the top of the game screen
        if self.y + self.radius + self.speedy >= HEIGHT:
            self.speedy = -self.speedy

        # check for collision with the bottom of the game screen
        elif self.y + self.radius + self.speedy <= 0:
            self.speedy = self.speed_magnitude

        # update the ball position
        self.x += self.speedx
        self.y += self.speedy

    # checks if ball has collided with the rect
    # which may be rect of block or paddle
    def collided(self, rect):
        if rect.left <= self.x + self.radius and self.x - self.radius <= rect.right:
            if rect.top < self.y + self.radius < rect.bottom:
                self.speedy = -self.speedy
                return True


if __name__ == '__main__':
    running = True
    paddle = Paddle()
    blocks = Blocks()

    # Initialize a stationary ball
    ball = Ball(5, 5)
    direction = 0

    t = 0
    max_steps = -1
    while running:
        if len(blocks.blocks) == 0 or t == max_steps:
            print("GAME OVER")
            sys.exit(0)

        # Build the state
        state = (paddle.rect.left, ball.x, ball.y)

        # Agent selects the action
        action = random.choice([-5, 5])

        # Use the action to progress the env

        # Move the paddle according to the action selected
        paddle.move(action)

        # Move the ball according to the collision physics defined
        ball.move()

        # Check for a collision with the paddle
        ball.collided(paddle.rect)

        # Check for a collision with a block
        # collect the reward
        reward = blocks.collided(ball)

        # Build the new state
        new_state = (paddle.rect.left, ball.x, ball.y)

        # Increment the step counter
        t+= 1

        # args.verbose
        if True:
            # Print the status
            print('t: {}, s: {}, a: {}, r: {}, ns: {}'.format(t, state, action, reward, new_state))

        # args.step_delay
        if True:
            time.sleep(0.5)