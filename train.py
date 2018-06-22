from contextlib import contextmanager
import math
import os
import pickle
import time
import torch
import torch.optim as optim
from basic_game import Game, Direction
from dqn import DQNTrainer, next_epsilon
from network import DQN
from simple_play import display_game, print_screen
from tensor_helper import game2tensor

SLEEP = 0.1
SAVE_TIME = 1000
WIDTH = 20
HEIGHT = 10
DECAY_STEP = 200
SIDE = math.sqrt(WIDTH*HEIGHT)
MAX_DIST = math.ceil(math.sqrt(WIDTH ** 2 + HEIGHT ** 2))
policy_network = DQN(width=WIDTH, height=HEIGHT)
optimizer = optim.SGD(policy_network.parameters(), lr=1e-3)
POLICY_PATH = 'data/policy.pt'
if os.path.isfile(POLICY_PATH):
    policy_network = torch.load(POLICY_PATH)


class SnakeTrainer(DQNTrainer):

    def __init__(self, *args, **kwargs):
        super(SnakeTrainer, self).__init__(*args, **kwargs)
        self.snake_step_info = dict()

    def decide_epsilon(self, game: Game):
        snake_len = len(game.snake)
        decay = self.snake_step_info.get(snake_len, 1)
        epsilon, self.snake_step_info[snake_len] = next_epsilon(
            self.eps_start, self.eps_end, self.decay_step, decay
        )
        return epsilon

    def save_trainer(self, dump):
        super().save_trainer(dump)
        dump(self.snake_step_info)

    def load_trainer(self, load):
        super().load_trainer(load)
        self.snake_step_info = load()


# noinspection PyTypeChecker
trainer = SnakeTrainer(policy_network, len(Direction), decay_step=DECAY_STEP)
TRAINER_PATH = 'data/trainer.pkl'
if os.path.isfile(TRAINER_PATH):
    with open(TRAINER_PATH, 'rb') as f_decay_read:
        trainer.load_trainer(lambda: pickle.load(f_decay_read))


class Decider:
    action = None

    def __call__(self, game):
        epsilon = trainer.decide_epsilon(game)
        action = trainer.select_action(game2tensor(game), epsilon)
        self.action = action
        # change to game format
        action = Direction(action.item())
        return action


def nope(*_args, **__kwargs):
    pass


@contextmanager
def ensure_time(second):
    time_before = time.time()
    yield
    time_delta = time.time() - time_before
    if time_delta < second:
        time.sleep(second - time_delta)


def main(stdscr):
    main.counter = 0

    def save_model():
        # update counter
        main.counter += 1
        main.counter %= SAVE_TIME
        if main.counter == SAVE_TIME - 1:
            # save after each episode
            torch.save(policy_network, POLICY_PATH)
            with open(TRAINER_PATH, 'wb') as file:
                trainer.save_trainer(lambda obj: pickle.dump(obj, file))

    def my_print(*args, **kwargs):
        return print_screen(stdscr, *args, **kwargs)

    trainer.print = my_print
    nope(my_print)

    loss = None
    try:  # for user interrupt
        while True:
            # Initialize the environment and state
            decider = Decider()
            game = Game(width=WIDTH, height=HEIGHT)
            for game_before, game_after in game.train(decider):
                with ensure_time(SLEEP):
                    display_game(game_before.draw_map(), stdscr)
                    state = game2tensor(game_before)
                    with torch.no_grad():
                        my_print(policy_network(state))
                    my_print('loss:', loss)

                    next_state = game2tensor(game_after)

                    if next_state is None:  # game failed
                        reward = 0
                    elif next_state is True:  # success
                        reward = 10
                        next_state = game2tensor(game_after)
                    else:
                        delta_length = len(game_after.snake) - len(game_before.snake)
                        delta_distance = game_after.head2food() - game_before.head2food()
                        if delta_length == 0:
                            reward = 1 - delta_distance / MAX_DIST
                        else:
                            reward = delta_length * 2
                    assert reward >= 0
                    # noinspection PyCallingNonCallable,PyUnresolvedReferences
                    reward = torch.tensor([reward], dtype=torch.float32)

                    # Store the transition in memory
                    trainer.memory.push(state, decider.action, next_state, reward)

                    # Perform one step of the optimization (on the target network)
                    loss = trainer.train_step()
                    save_model()
    except KeyboardInterrupt:  # no need to handler this
        pass
    finally:
        save_model()
    return "stop at {}".format(loss)


if __name__ == '__main__':
    cur = __import__('curses')
    cur.start_color = lambda: None
    result = cur.wrapper(main)
    print(result)
