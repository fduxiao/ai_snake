import math
import os
import pickle
from time import sleep
import torch
from basic_game import Game, Direction
from dqn import DQNTrainer
from network import DQN
from simple_play import display_game, print_screen
from tensor_helper import game2tensor

SLEEP = 0.1
SAVE_TIME = 10
WIDTH = 20
HEIGHT = 10
SIDE = math.sqrt(WIDTH*HEIGHT)
policy_network = DQN(width=WIDTH, height=HEIGHT)
POLICY_PATH = 'data/policy.pt'
if os.path.isfile(POLICY_PATH):
    policy_network = torch.load(POLICY_PATH)

target_network = DQN(width=WIDTH, height=HEIGHT)


# noinspection PyTypeChecker
trainer = DQNTrainer(policy_network, target_network, len(Direction), eps_decay=100000)
DECAY_PATH = 'data/decay.pkl'
if os.path.isfile(DECAY_PATH):
    with open(DECAY_PATH, 'rb') as f_decay_read:
        trainer.decay = pickle.load(f_decay_read)


class Decider:
    action = None

    def __call__(self, game):
        action = trainer.select_action(game2tensor(game))
        self.action = action
        # change to game format
        action = Direction(action.item())
        return action


def nope(*_args, **__kwargs):
    pass


def main(stdscr):
    main.counter = 0

    def save_model():
        # update counter
        main.counter += 1
        main.counter %= SAVE_TIME
        if main.counter == SAVE_TIME - 1:
            # save after each episode
            torch.save(policy_network, POLICY_PATH)
            with open(DECAY_PATH, 'wb') as file:
                pickle.dump(trainer.decay, file)

    def my_print(*args, **kwargs):
        return print_screen(stdscr, *args, **kwargs)

    nope(my_print)

    while True:
        # Initialize the environment and state
        decider = Decider()
        game = Game(width=WIDTH, height=HEIGHT)
        for game_before, game_after in game.train(decider):

            display_game(game_before.draw_map(), stdscr)
            state = game2tensor(game_before)
            next_state = game2tensor(game_after)

            if next_state is None:  # game failed
                reward = -1
            elif next_state is True:  # success
                reward = 10
            else:
                delta_length = len(game_after.snake) - len(game_before.snake)
                delta_distance = game_after.head2food() - game_before.head2food()
                reward = delta_length * 2 - delta_distance / SIDE
            # noinspection PyCallingNonCallable,PyUnresolvedReferences
            reward = torch.tensor([reward], dtype=torch.float32)

            # Store the transition in memory
            trainer.memory.push(state, decider.action, next_state, reward)

            # Perform one step of the optimization (on the target network)
            trainer.train_step()
            sleep(SLEEP)
            save_model()


if __name__ == '__main__':
    cur = __import__('curses')
    cur.start_color = lambda: None
    cur.wrapper(main)
