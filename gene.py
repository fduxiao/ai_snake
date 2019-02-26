import random
import time

import torch
import torch.nn as nn

from basic_game import Game, Direction
from network import Neuron


class Brain(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_dim = hidden_size
        self.hidden = None
        self.init_hidden()
        self.i2h = Neuron(hidden_size + 2 + 2 + 4, hidden_size)
        self.i2o = Neuron(hidden_size + 2 + 4, 4)
        self.softmax = nn.Softmax(dim=0)

    def init_hidden(self):
        self.hidden = torch.zeros(self.hidden_dim)

    def forward(self, food, obstacle, body):
        self.init_hidden()
        for x in body:
            combined = torch.cat((food, obstacle, x, self.hidden))
            self.hidden = self.i2h(combined)
        combined = torch.cat((food, obstacle, self.hidden))
        output = self.i2o(combined)
        return self.softmax(output)

    @property
    def neuron(self):
        return [self.i2h, self.i2o]

    def mutate(self, rate=0.1):
        for neuron in self.neuron:
            neuron.mutate(rate)

    def cross(self, another):
        another: Brain = another
        for n1, n2 in zip(self.neuron, another.neuron):
            n1.cross(n2)


class Snake:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.brain = Brain(int((width * height) ** 0.5) * 2)
        self.fitness = 0.

    def decide(self, game: Game):
        # noinspection PyUnresolvedReferences
        food = torch.tensor(game.food, dtype=torch.float)
        # noinspection PyUnresolvedReferences
        body = torch.tensor(game.snake, dtype=torch.float)
        head = game.snake[0]
        x, y = head
        obstacle = []
        for i in range(1, self.width+1):
            if x+i >= self.width or (x+i, y) in game.snake:
                obstacle.append(i)
                break
        for i in range(1, self.width+1):
            if x-i < 0 or (x-i, y) in game.snake:
                obstacle.append(i)
                break

        for i in range(1, self.height+1):
            if y+i >= self.height or (x, y+i) in game.snake:
                obstacle.append(i)
                break
        for i in range(1, self.height+1):
            if y-i < 0 or (x, y-i) in game.snake:
                obstacle.append(i)
                break
        # noinspection PyUnresolvedReferences
        result = self.brain(food, torch.tensor(obstacle, dtype=torch.float), body)
        direction = result.argmax().item()
        return Direction(direction)

    def mutate(self, rate=0.1):
        self.brain.mutate(rate)

    def cross(self, another):
        self.brain.cross(another.brain)

    def clone(self):
        snake = Snake(self.width, self.height)
        snake.brain.load_state_dict(self.brain.state_dict(), strict=True)
        return snake


class Population:
    def __init__(self, width, height, size, n_tops, gamma=0.1, rate=0.5):
        self.size = size
        self.n_tops = n_tops
        self.game = Game(width, height)
        self.snakes = []
        self.width = width
        self.height = height
        self.gamma = gamma
        self.rate = rate
        self.create()
        self.generation = 0

    def create(self):
        self.snakes = []
        for i in range(self.size):
            self.snakes.append(Snake(self.width, self.height))

    def sort(self):
        self.snakes.sort(key=lambda x: x.fitness, reverse=True)

    def evolve(self):
        self.generation += 1
        self.sort()
        if self.snakes[0].fitness <= 0.00001:
            self.create()

        for i in range(self.n_tops, self.size):  # losers
            if i in {self.n_tops, self.n_tops+1, self.n_tops+2, self.n_tops+3, self.n_tops+4}:
                parent_a = self.snakes[0]
                parent_b = self.snakes[1]
                if random.random() > 0.5:
                    parent_b, parent_a = parent_a, parent_b
                offspring = parent_a.clone()
                offspring.cross(parent_b)

            elif i < self.size * 0.8:
                parent_a = random.choice(self.snakes[:self.n_tops])
                parent_b = random.choice(self.snakes[:self.n_tops])
                offspring = parent_a.clone()
                offspring.cross(parent_b)
            else:
                offspring = random.choice(self.snakes[:self.n_tops]).clone()
            offspring.fitness = 0
            offspring.mutate()
            self.snakes[i] = offspring
        self.sort()

    def step(self, info=lambda *args, **kwargs: None):
        for i, snake in enumerate(self.snakes):
            game = self.game.copy()
            visited = set()
            dist = game.head2food()
            snake.fitness = 0
            for game_map in game.play(snake.decide):
                base_fitness = 0.
                new_dist = game.head2food()
                base_fitness += 1/(new_dist - dist)

                dist = new_dist
                hashed_map = hash(tuple(map(tuple, game_map)))
                info(i, game, game_map)
                if hashed_map in visited:
                    game.result = None
                    snake.fitness = 0
                    break
                else:
                    visited.add(hashed_map)
            else:
                if game.result is True:
                    snake.fitness = (self.width * self.height) ** 0.5 + len(game.snake)
                else:
                    snake.fitness = len(game.snake) + game.len_per_step + 1/dist
        self.sort()


class Timer:
    def __init__(self, seconds):
        self.seconds = seconds
        self.prev = time.time()

    def wait(self):
        delta = time.time() - self.prev
        if delta < self.seconds:
            time.sleep(self.seconds - delta)
        self.prev = time.time()


def main(stdscr):
    width = 20
    height = 10
    # n_step = 10
    timer = Timer(0.1)
    from simple_play import display_game, print_screen
    population = Population(width, height, 10, 8, rate=0.5)

    def print_game(i, game, game_map):
        timer.wait()
        # display game
        display_game(game_map, stdscr)
        # current generation
        print_screen(stdscr, f"generation: {population.generation}")
        # current solution
        print_screen(stdscr, f"direction: {game.direction}")
        if i is not None:
            print_screen(stdscr, f"{i+1}, {(i+1)/population.size*100:6.2f}%")

    def decider(game: Game) -> Direction:
        # for _ in range(n_step):
        population.step(print_game)
        return population.snakes[0].decide(game)

    def one_game():
        game = population.game
        for game_map in game.play(decider):
            population.evolve()
            print_game(None, game, game_map)
        if population.game.result is True:
            stdscr.addstr("Success!")
            display_game(game.draw_map(), stdscr)
            return True
        return False

    while not one_game():
        population.game.reset()
        population.evolve()


if __name__ == '__main__':
    cur = __import__('curses')
    cur.start_color = lambda: None
    print(cur.wrapper(main))
