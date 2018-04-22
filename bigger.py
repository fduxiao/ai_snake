from network import *
from basic_game import MapStatus, Direction, Game
from tensor_helper import flatten_tensor
import numpy as np


def play_net(input_tensor, width, height, keep_prob, net_name):
    with tf.name_scope(net_name):
        in_number = width * height * len(MapStatus)

        out_number = 3000
        nn = nn_layer(input_tensor, in_number, out_number, 'nn1')
        nn = tf.nn.dropout(nn, keep_prob)
        in_number = out_number

        out_number = 3000
        nn = nn_layer(nn, in_number, out_number, 'nn1')
        nn = tf.nn.dropout(nn, keep_prob)
        in_number = out_number

        out_number = 1000
        nn = nn_layer(nn, in_number, out_number, 'nn1')
        nn = tf.nn.dropout(nn, keep_prob)
        in_number = out_number

        out_number = 100
        nn = nn_layer(nn, in_number, out_number, 'nn1')
        nn = tf.nn.dropout(nn, keep_prob)
        in_number = out_number

        nn = nn_layer(nn, in_number, len(Direction), 'nn2', act=tf.identity)
    return nn


def get_decider(sess: tf.Session, width, height):
    with tf.name_scope('decider'):
        x = tf.placeholder(tf.float32, [None, width * height * len(MapStatus)], name='x')
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            tf.summary.scalar('dropout_keep_probability', keep_prob)

        main_net = play_net(x, width, height, keep_prob, 'PlayNet')
        max_arg = tf.argmax(main_net, 1)

    def decider(game: Game):
            arr = np.array(flatten_tensor(game.draw_map()))
            k = sess.run(max_arg, feed_dict={x: arr.reshape([-1, len(arr)]), keep_prob: 1})
            return Direction(k[0])
    return decider
