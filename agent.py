import torch
import random
import numpy as np
from collections import deque
from game3D import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(19, 256, 6)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game, first=False):
        if first:
            head = Point(0, 0, 0)
        else:
            head = game.snakelist[0]
        point_r = Point(head.x + 1, head.y, head.z)
        point_l = Point(head.x - 1, head.y, head.z)
        point_u = Point(head.x, head.y + 1, head.z)
        point_d = Point(head.x, head.y - 1, head.z)
        point_s = Point(head.x, head.y, head.z + 1)
        point_b = Point(head.x, head.y, head.z - 1)
        
        dir_r = game.direction == Direction.RIGHT
        dir_l = game.direction == Direction.LEFT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        dir_s = game.direction == Direction.STRAIGHT
        dir_b = game.direction == Direction.BACK

        state = [
                        # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)) or 
            (dir_s and game.is_collision(point_s)) or 
            (dir_b and game.is_collision(point_b)),
            # Danger right
            dir_r == game.is_collision(point_r),
            # Danger left
            dir_l == game.is_collision(point_l),
            # Danger up
            dir_u == game.is_collision(point_u),
            # Danger down
            dir_d == game.is_collision(point_d),
            # Danger straight
            dir_s == game.is_collision(point_s),
            # Danger back
            dir_b == game.is_collision(point_b),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            dir_s,
            dir_b,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food down
            game.food.y > game.head.y,  # food up
            game.food.z < game.head.z,  # food back
            game.food.z > game.head.z  # food straight
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 5)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game, first=True)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()