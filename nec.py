import tensorflow as tf
import numpy as np
from kernel_methods import *
from sklearn.neighbors import KDTree

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
        # placeholder for memory
        self.key = [tf.placeholder(dtype=tf.float32, shape=(None, 512), name='key_%s' % i) for i in range(num_act)]
        self.value = [tf.placeholder(dtype=tf.float32, shape=(None, 1), name='value_%s' % i) for i in range(num_act)]

        self.nn = self.build_network()

    def build_network(self):
        """Architecture same as DQN"""
        with tf.variable_scope('conv1'):
            # conv = tf.layers.conv2d(self.x, )
            pass


class DND(object):
    """Numpy implementation of Differentiable Neural Dictionary. Use Numpy structured array"""

    def __init__(self, mem_size, num_action):
        # construct dnd for different actions
        self.dnd = (np.empty(mem_size, dtype=[('key', '512float32'),
                                              ('value', 'float32')]), ) * num_action
        self.mem_size = mem_size
        self.num_action = num_action
        self.current_size = 0
        # kd-tree with key to perform knn search
        self.trees = [KDTree(self.dnd[i]['key']) for i in range(num_action)]
        self.visited_neighbors = (np.zeros(mem_size), ) * num_action

    def write(self, h, q):
        """
            Append to memory if no key matched, update memory if key matches, or overwrite memory if size reaches max.
        """
        # append

        # update

        # overwrite

    def lookup(self, h):
        """
            Perform knn soft-lookup.
            h is a N*D dimensional numpy array, where N is mini-batch size, D is the dimension of the last Q network
        """

        indexes = []
        for i in range(self.num_action):
            _, index = self.trees[i].query(h, k=50)
            # update the count for neighbors
            self.visited_neighbors[i][index] += 1
            indexes.append(i)

        # lookup
        o_s = []
        for i in range(self.num_action):
            # 50 * 512
            keys = self.dnd[i]['key'][indexes[i]]
            # 50 * 1
            values = self.dnd[i]['value'][indexes[i]]

            # N
            kernel_sum = np.sum(k(h, keys), axis=1)
            # N * 50
            w = k(h, keys)/kernel_sum

            # Q(s, a[i])
            o = np.dot(w, values)
            # append to o_s
            o_s.append(o)






