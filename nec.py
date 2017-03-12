import tensorflow as tf
import numpy as np
from kernel_methods import *

class NEC(object):
    def __init__(self, sess, input_dim, num_mem, num_act, p):
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

        with tf.name_scope('dnd'):
            self.key = tf.zeros((num_act, num_mem, 512), dtype=tf.float32, name='dnd_key')
            self.value = tf.zeros((num_act, num_mem, 1), dtype=tf.float32, name='dnd_value')

        self.nn = self.build_network()

    def build_network(self):
        with tf.variable_scope('conv1'):
            # conv = tf.layers.conv2d(self.x, )
            pass

    def query(self):
        """
        Query a key-value pair based on equation(1) and (2).
        """
        # calculate w
        # TODO: Kernel method
        with tf.name_scope('w'):
            w_s = []
            for i in range(self.num_act):
                w = k(self.nn, self.key[i])
                w_s.append(w)

        # calculate o
        # TODO: KNN
        with tf.name_scope('o'):
            o_s = []
            for i in range(self.num_act):
                o = w_s[i] * self.value[i]
                o_s.append(o)

    def write(self):
        """
        Write or update DND key-value pair
        """
