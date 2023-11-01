import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

from pygame.locals import DOUBLEBUF, OPENGL, QUIT
from OpenGL.GL import glBegin, glEnd, glVertex3fv, glClear, glRotatef, glTranslatef, GL_QUADS, glColor3fv, \
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_LINES
from OpenGL.GLU import gluPerspective


pygame.init()


class Direction(Enum):
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP = 4
    STRAIGHT = 5
    BACK = 6

Point = namedtuple('Point', 'x, y, z')

# rgb colors
red = (1, 0, 0)
green = (0, 1, 0)
blue = (0, 0, 1)
white = (1, 1, 1)
sky = (0, 1, 1)
yellow = (1, 1, 0)
black = (0, 0, 0)
gray = (0.5, 0.5, 0.5)
pink = (1, 0, 1)


block_size = 1
arena_size = 10 * block_size
sn_len = 3
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=600, h=600, sn_len=sn_len):
        self.w = w
        self.h = h
        self.x = 0
        self.y = 0
        self.z = 0
        self.sn_len = sn_len
        # init display
        self.display = pygame.display.set_mode((self.w, self.h), DOUBLEBUF|OPENGL)
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        # OpenGL Params
        self.gluPerspective = gluPerspective(50, (self.w / self.h), 0.1, 500.0)
        self.glTranslatef = glTranslatef(0.0, 0.0, -2 * arena_size)
        self.glRotatef = glRotatef(45, 1.0, 1.5, 0.2)
        self._move_matrix()

        self.reset()

    def _move_matrix(self):
        n = 6
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i < j:
                    matrix[i, j] = -1 * (j - i)
                elif i > j:
                    matrix[i, j] = i - j
        self.matrix = matrix

    def cube(self, coords, block_size, color = [white], fill = True):
        self.vertices = ((coords[0] + block_size / 2, coords[1] + block_size / 2, coords[2] + block_size / 2),
                        (coords[0] + block_size / 2, coords[1] - block_size / 2, coords[2] + block_size / 2),
                        (coords[0] - block_size / 2, coords[1] - block_size / 2, coords[2] + block_size / 2),
                        (coords[0] - block_size / 2, coords[1] + block_size / 2, coords[2] + block_size / 2),
                        (coords[0] - block_size / 2, coords[1] + block_size / 2, coords[2] - block_size / 2),
                        (coords[0] + block_size / 2, coords[1] + block_size / 2, coords[2] - block_size / 2),
                        (coords[0] + block_size / 2, coords[1] - block_size / 2, coords[2] - block_size / 2),
                        (coords[0] - block_size / 2, coords[1] - block_size / 2, coords[2] - block_size / 2))

        self.edges = ((0,1),
                    (1,2),
                    (2,3),
                    (3,0),
                    (5,6),
                    (6,7),
                    (7,4),
                    (4,5),
                    (0,5),
                    (1,6),
                    (2,7),
                    (3,4))

        self.surfaces = ((0, 1, 2, 3),
                        (4, 5, 6, 7),
                        (0, 1, 6, 5),
                        (2, 1, 6, 7),
                        (7, 2, 3, 4),
                        (3, 4, 5, 0))
        if not fill:
            glBegin(GL_LINES)
            glColor3fv(black)
            for edge in self.edges:
                for vertex in edge:
                    glVertex3fv(self.vertices[vertex])
            glEnd()
        else:	
            i = 0
            colors = color
            glBegin(GL_QUADS)
            for surface in self.surfaces:
                for vertex in surface:
                    glColor3fv(colors[i % len(colors)])
                    i += 1
                    glVertex3fv(self.vertices[vertex])
            glEnd()


    def reset(self):
        # init game state
        self.direction = random.choice(list(Direction))

        self.snakelen = self.sn_len
        self.head = Point(self.x, self.y, self.z)
        self.snakelist = []
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def apple(self, coords, block_size = block_size):
        self.cube(tuple(coords), block_size, color = [red, pink])

    def snake(self, snakelist, snakelen, block_size = block_size):
        if len(snakelist) > snakelen:
            del snakelist[0]

        for xyz in snakelist:
            self.cube(tuple(xyz), block_size, [sky, yellow])


    def _get_random_apple_point(self):
        val = round(arena_size // 2) - 1
        apple_x = round((random.randint(-val, val)) / block_size)
        apple_y = round((random.randint(-val, val)) / block_size)
        apple_z = round((random.randint(-val, val)) / block_size)
        return Point(apple_x, apple_y, apple_z)


    def _place_food(self):
        self.food = self._get_random_apple_point()
        if self.food in self.snakelist:
            self._place_food()
        self.apple(self.food, block_size)


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snakelist.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 50*len(self.snakelist):
            game_over = True
            reward = -20
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self.snakelen += 1
            self.snakelist.insert(0, self.food)
            reward = 20
            self._place_food()
            self._update_ui()
        elif len(self.snakelist) <= self.snakelen:
            pass
        else:
            self.snakelist.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        # if pt is None:
        pt = self.head
        # hits boundary
        if abs(pt.x) >= abs((arena_size - block_size) / 2) or abs(pt.y) >= abs((arena_size - block_size) / 2) or abs(pt.z) >= abs((arena_size - block_size) / 2):
            return True
        # hits itself
        if pt in self.snakelist[1:]:
            return True

        return False


    def _update_ui(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        self.cube(Point(0, 0, 0), arena_size, color = [gray, white])
        self.cube(Point(0, 0, 0), arena_size, fill = False)
        self.display.fill(black)
        self.apple(self.food)
        self.snake(self.snakelist, self.snakelen)
        pygame.display.flip()


    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP, Direction.STRAIGHT, Direction.BACK]
        idx = clock_wise.index(self.direction)
        action_ind = np.where(np.array(action) == 1)[0][0]
        new_dir = clock_wise[idx + self.matrix[action_ind, idx]]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        z = self.head.z
        if self.direction == Direction.RIGHT:
            x += block_size
        elif self.direction == Direction.LEFT:
            x -= block_size
        elif self.direction == Direction.UP:
            y += block_size
        elif self.direction == Direction.DOWN:
            y -= block_size
        elif self.direction == Direction.STRAIGHT:
            z += block_size
        elif self.direction == Direction.BACK:
            z -= block_size

        self.head = Point(x, y, z)