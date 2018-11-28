import sys
import pygame
from pygame.locals import *
import itertools

pygame.init()
pygame.font.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (192, 192, 192)
GREEN = (46, 139, 87)
RED = (220, 20, 60)
BLUE = (25, 25, 112)
BROWN = (244, 164, 96)
PURPLE = (178, 102, 255)
ORANGE = (255, 128, 0)
HEIGHT = 600
WIDTH = 800
FONT = pygame.font.SysFont(None, 60)

display = pygame.display.set_mode((WIDTH, HEIGHT))
display.fill(BLACK)
pygame.display.set_caption('Atari Breakout')


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
        rows_height = HEIGHT//rows
        for i in range(0, rows_height, self.height):
            for j in range(0, WIDTH, self.width):
                rects.append(pygame.Rect(j, i, self.width, self.height))
        print('rects', rects)
        print('len', len(rects))
        return rects

    def draw_blocks(self):
        colors = itertools.cycle([RED, GREEN, BLUE, PURPLE, ORANGE])
        for i in self.blocks:
            color = next(colors)
            pygame.draw.rect(display, color, i)
        return

    # removes single block from blocks list when it is hit by ball
    # ball being the ball object
    def collided(self, ball_object):
        for i in self.blocks:
            if ball_object.collided(i):
                self.blocks.remove(i)
        return


class Paddle:
    def __init__(self):
        self.width = WIDTH // 4
        self.height = 10
        self.initial_x = (WIDTH//2) - (self.width//2)
        self.initial_y = HEIGHT - 50
        self.rect = pygame.Rect(self.initial_x, self.initial_y,
                                self.width, self.height)
        print(self.rect)

    def move(self, speed):
        if self.rect.right + speed > WIDTH or self.rect.left + speed < 0:
            return
        else:
            self.rect = self.rect.move(speed, 0)

    def draw(self):
        pygame.draw.rect(display, GREY, self.rect)
        return


class Ball:
    """Ball object that takes initial speed in x direction (speedx)
        and initial speed in y direction(speedy)"""
    def __init__(self, speedx, speedy):
        self.x = WIDTH//2
        self.y = HEIGHT - 60
        self.radius = 10
        self.speedx = speedx
        self.speedy = speedy

    def draw(self):
        pygame.draw.circle(display, BROWN, (self.x, self.y), self.radius)

    def move(self):
        if self.x + self.radius + self.speedx >= WIDTH:
            self.speedx = -self.speedx
        elif self.x + self.speedx <= 0:
            self.speedx = abs(self.speedx)
        if self.y + self.radius + self.speedy >= HEIGHT:
            self.speedy = -self.speedy
        elif self.y + self.radius + self.speedy <= 0:
            self.speedy = abs(self.speedy)
        self.x += self.speedx
        self.y += self.speedy
        return

    # checks if ball has collided with the given pygame rect object
    # which may be rect of block or paddle
    def collided(self, rect):
        if rect.left <= self.x + self.radius and\
                self.x - self.radius <= rect.right:
            if rect.top < self.y + self.radius < rect.bottom:
                self.speedy = -self.speedy
                return True
        else:
            return False


def show_text(text):
    text = str(text).encode("UTF-8")
    display.fill(BLACK)
    my_text = FONT.render(text, True, WHITE)
    width, height = FONT.size(text)
    display.blit(my_text, (WIDTH//2 - width//2, HEIGHT//2 - height//2))
    return

if __name__ == '__main__':
    running = True
    paddle = Paddle()
    blocks = Blocks()
    ball = Ball(0, 0)
    direction = 0
    paused = False
    clock = pygame.time.Clock()
    while running:
        if len(blocks.blocks) == 0:
            show_text("GAME OVER")
            pygame.display.flip()
            pygame.time.wait(2000)
            pygame.display.quit()
            pygame.quit()
            sys.exit(0)
        for e in pygame.event.get():
            if e.type == QUIT:
                pygame.display.quit()
                pygame.quit()
                sys.exit(0)
            elif e.type == KEYDOWN:
                if e.key == K_RIGHT:
                    direction = 5
                elif e.key == K_LEFT:
                    direction = -5
                elif e.key == K_p:
                    paused = not paused
                elif e.key == K_SPACE:
                    if ball.speedx == 0 and ball.speedy == 0:
                        ball.speedx = 5
                        ball.speedy = 5
                continue
        if not paused:
            paddle.move(direction)
            ball.move()
            ball.collided(paddle.rect)
            blocks.collided(ball)
            display.fill(BLACK)
            blocks.draw_blocks()
            paddle.draw()
            ball.draw()
        else:
            show_text("PAUSED")
        pygame.display.flip()
        clock.tick(60)
