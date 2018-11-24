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


class Paddle(Rect):
    def __init__(self, args):
        self.args = args
        self.width = self.args.env_width // 2
        self.height = 20
        self.initial_x = (self.args.env_width//2) - (self.width//2)
        self.initial_y = self.args.env_height - 50
        super().__init__(self.initial_x, self.initial_y,
                         self.width, self.height)

    def move(self, speed):
        if ((self.right + speed > self.args.env_width) or
            (self.left + speed < 0)):
            # out of bounds, do not update the paddle position
            # print('paddle collide with side of screen')
            # TODO[jason] handle reward corruption
            return
        else:
            # update the paddle position
            super().move(speed)


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
            if args.verbose:
                print('ball collide with right side of screen')
            self.speedx = -self.speed_magnitude

        # check for collision with the left hand side of the game screen
        elif self.x + self.speedx <= 0:
            if args.verbose:
                print('ball collide with left side of screen')
            self.speedx = self.speed_magnitude

        # check for collision with the bottom of the game screen
        if self.y + self.radius + self.speedy >= self.args.env_height:
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
