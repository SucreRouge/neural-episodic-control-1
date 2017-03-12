import tensorflow as tf
import numpy as np


class NEC(object):
    def __init__(self, sess, input_dim, num_act, p):
        """
        Args:
            sess: tf.Session()
            input_dim: input dimension.
            num_act: number of actions
            num_mem: number of memories per action
            p: parameter for nearest neighbours
        """
        self.sess = sess
        self.input_dim = input_dim
        self.num_act = num_act
        self.p = p

        self.x = tf.placeholder(dtype=tf.float32, shape=(None, )+input_dim, name='input_img')
        self.r = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="n_step_return")

        self.nn = self.build_network()

    def build_network(self):
        with tf.variable_scope('conv1'):
            # conv = tf.layers.conv2d(self.x, )
            pass
