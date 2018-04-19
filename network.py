import tensorflow as tf
from basic_game import MapStatus, Direction, Game
from tensor_helper import flatten_tensor
import numpy as np


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, w, name=None):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', name=name)


def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


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


def train_affair(labels, logits, name, rate=1e-4):
    with tf.name_scope(name):
        with tf.name_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    return train_step, accuracy


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
