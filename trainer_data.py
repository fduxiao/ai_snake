#!/usr/bin/env python3
from simple_play import display_and_play_game, interactive_decider
from tensor_helper import flatten_tensor
from basic_game import Game
import pickle


width = 20
height = 10


def decider(target_file: open, game: Game, stdscr):
    d = interactive_decider(stdscr)
    data = {
        'direction': d.value,
        'game': flatten_tensor(game.draw_map())
    }
    pickle.dump(data, target_file)
    return d


if __name__ == '__main__':
    cur = __import__("curses")
    cur.start_color = lambda: None
    with open('./data/train.pkl', 'ab') as f:
        cur.wrapper(lambda stdscr: display_and_play_game(
            stdscr,
            height=height,
            width=width,
            decider=lambda game: decider(f, game, stdscr)
        ))
