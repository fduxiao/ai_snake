#!/usr/bin/env python3
from simple_play import display_and_play_game
from bigger import get_decider
import tensorflow as tf
import time

sess = tf.Session()

width = 20
height = 10
dec = get_decider(sess, width, height)


init_op = tf.global_variables_initializer()
sess.run(init_op)


def decider(game):
    time.sleep(0.5)
    return dec(game)


if __name__ == '__main__':
    cur = __import__("curses")
    cur.start_color = lambda: None
    cur.wrapper(display_and_play_game, height=height, width=width, decider=decider)
