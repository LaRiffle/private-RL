from gym import logger

class Rect(object):
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


class Blocks(object):
    """Implements blocks as a collection instead of
        as individual block objects """
    def __init__(self, args):
        self.args = args
        self.width = 100
        self.height = 20
        self.blocks = self.make_blocks()
        self.num_blocks_start = len(self.blocks)
        self.num_blocks_destroyed = 0

    def make_blocks(self):
        rects = []
        rows = 5
        self.rows_height = self.args.env_height // rows
        for i in range(0, self.args.env_width, self.width):
            for j in range(0, self.rows_height, self.height):
                rects.append(Rect(i, j, self.width, self.height))
        return rects

    # removes single block from blocks list when it is hit by ball
    # ball being the ball object
    def collided(self, ball_object):
        collided = False
        for block in self.blocks:
            if ball_object.collided(block, 'block'):
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


class Paddle(Rect):
    def __init__(self, args):
        self.args = args
        # TODO(korymath): what is the correct size for the paddle
        self.width = self.args.env_width // 3
        self.height = 20
        self.initial_x = (self.args.env_width // 2) - (self.width // 2)
        self.initial_y = self.args.env_height - 50
        self.rect = Rect(self.initial_x, self.initial_y,
                         self.width, self.height)

    def move(self, speed):
        # check if the move would collide paddle with edge
        if ((self.rect.right + speed > self.args.env_width) or
            (self.rect.left + speed < 0)):
            # out of bounds, do not update the paddle position
            # TODO[jason] handle reward corruption
            return True
        self.rect = self.rect.move(speed)
        return False


class Ball(object):
    """Ball object that takes initial speed in x direction (speedx)
        and initial speed in y direction(speedy)"""
    def __init__(self, args, speedx, speedy):
        self.args = args
        self.radius = 10
        self.x = self.args.env_width//2
        self.y = self.args.env_height - 70
        self.speed_magnitude = 5
        self.speedx = speedx
        self.speedy = speedy

    def move(self):
        # check for collision with the right side of the game screen
        if self.x + self.radius + self.speedx >= self.args.env_width:
            logger.debug('ball collide with right side of screen')
            self.speedx = -self.speed_magnitude

        # check for collision with the left hand side of the game screen
        elif self.x + self.speedx <= 0:
            logger.debug('ball collide with left side of screen')
            self.speedx = self.speed_magnitude

        # check for collision with the bottom of the game screen
        if self.y + self.radius + self.speedy >= self.args.env_height:
            logger.debug('ball collide with bottom of screen')
            self.speedy = -self.speed_magnitude
            return True

        # check for collision with the top of the game screen
        elif self.y + self.radius + self.speedy <= 0:
            logger.debug('ball collide with top of screen')
            self.speedy = self.speed_magnitude

        # update the ball position
        self.x += self.speedx
        self.y += self.speedy
        return False

    # checks if ball has collided with the rect_obj
    # which may block or paddle
    def collided(self, rect, collider):
        if collider == 'paddle':
            left_temp = rect.rect.left
            right_temp = rect.rect.right
            bottom_temp = rect.rect.bottom
            top_temp = rect.rect.top
        else:
            left_temp = rect.left
            right_temp =rect.right
            bottom_temp = rect.bottom
            top_temp = rect.top

        if ((left_temp <= self.x + self.radius) and
            (self.x - self.radius <= right_temp)):
            if top_temp < self.y + self.radius < bottom_temp:
                logger.debug('ball collide with {}'.format(collider))
                self.speedy = -self.speedy
                # add an extra displacement to avoid double collision
                self.y += self.speedy
                return True
