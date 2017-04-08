import os
import time
import sys
import random
import math
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import functools
import utils.sunnybrook as sunnybrook

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops


class LVSegmentation(object):
    def __init__(self, use_cpu=False, checkpoint_dir='../../result/segmenter/train_result/v1/'):
        random.seed(time.time())

        self.build(use_cpu=use_cpu)
        self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        self.checkpoint_dir = checkpoint_dir

        self.loss_array = []

    def restore_session(self):
        global_step = 0
        if not os.path.exists(self.checkpoint_dir):
            raise IOError(self.checkpoint_dir + ' does not exist.')
        else:
            path = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if path is None:
                raise IOError('No checkpoint to restore in ' + self.checkpoint_dir)
            else:
                self.saver.restore(self.session, path.model_checkpoint_path)
                global_step = int(path.model_checkpoint_path.split('-')[-1])

        with open(self.checkpoint_dir + 'loss.pickle', 'rb') as f:
            self.loss_array = pickle.load(f)

        return global_step

    def save_loss(self):

        if os.path.exists(self.checkpoint_dir + 'loss.pickle'):
            os.remove(self.checkpoint_dir + 'loss.pickle')

        with open(self.checkpoint_dir + 'loss.pickle', 'wb') as f:
            pickle.dump(self.loss_array, f)

    def predict(self, images):
        self.restore_session()

        return self.prediction.eval(session=self.session, feed_dict={self.x: images})

    def evaluate(self, eval_paths):
        self.restore_session()

        _, images, labels = self.read_data(eval_paths)

        accuracy = self.accuracy.eval(session=self.session, feed_dict={self.x: images, self.y: labels})

        print('Model has accuracy : {:.6f} '.format(accuracy))

    def train(self, train_paths, training_steps=1000, restore_session=False, learning_rate=1e-6):
        if restore_session:
            step_start = self.restore_session() + 1
        else:
            step_start = 1

        for i in range(step_start, step_start + training_steps):
            # pick random data
            train_path = random.sample(train_paths, 5)
            _, images, labels = self.read_data(train_path)

            if i % 10 == 0:
                print('run train step: ' + str(i))

            start = time.time()

            self.train_step.run(session=self.session,
                                feed_dict={self.x: images, self.y: labels, self.rate: learning_rate})

            if i % 100 == 0:
                loss = self.loss.eval(session=self.session, feed_dict={self.x: images, self.y: labels})

                print('Step {} finished in {:.2f} s with Loss : {:.6f}'.format(i, time.time() - start, loss))

                self.saver.save(self.session, self.checkpoint_dir + 'model', global_step=i)

                self.loss_array.append(loss)
                self.save_loss()

                print('Model {} saved'.format(i))

    def read_data(self, paths):
        images, labels = sunnybrook.export_all_contours(paths)

        crop_x = random.randint(0, 16)
        crop_y = random.randint(0, 16)

        images = images[:, crop_y:crop_y + 224, crop_x: crop_x + 224]
        labels = labels[:, crop_y:crop_y + 224, crop_x: crop_x + 224]
        images = np.float32(images)

        before_normalization = images

        images -= np.mean(images, dtype=np.float32)  # zero-centered
        images /= np.std(images, dtype=np.float32)  # normalization

        images = np.reshape(images, (-1, 224, 224, 1))

        return before_normalization, images, labels

    @ops.RegisterGradient("MaxPoolWithArgmax")
    def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
        return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
                                                     grad,
                                                     op.outputs[1],
                                                     op.get_attr("ksize"),
                                                     op.get_attr("strides"),
                                                     padding=op.get_attr("padding"))

    def build(self, use_cpu=False):

        if use_cpu:
            device = '/cpu:0'
        else:
            device = '/gpu:0'

        with tf.device(device):
            self.x = tf.placeholder(tf.float32, shape=(None, 224, 224, 1))
            self.y = tf.placeholder(tf.int64, shape=(None, 224, 224))

            expected = tf.expand_dims(self.y, -1)
            self.rate = tf.placeholder(tf.float32, shape=[])

            conv_1_1 = self.conv_layer(self.x, [3, 3, 1, 64], 64, 'conv_1_1')
            conv_1_2 = self.conv_layer(conv_1_1, [3, 3, 64, 64], 64, 'conv_1_2')

            pool_1, pool_1_argmax = self.pool_layer(conv_1_2)

            conv_2_1 = self.conv_layer(pool_1, [3, 3, 64, 128], 128, 'conv_2_1')
            conv_2_2 = self.conv_layer(conv_2_1, [3, 3, 128, 128], 128, 'conv_2_2')

            pool_2, pool_2_argmax = self.pool_layer(conv_2_2)

            conv_3_1 = self.conv_layer(pool_2, [3, 3, 128, 256], 256, 'conv_3_1')
            conv_3_2 = self.conv_layer(conv_3_1, [3, 3, 256, 256], 256, 'conv_3_2')
            conv_3_3 = self.conv_layer(conv_3_2, [3, 3, 256, 256], 256, 'conv_3_3')

            pool_3, pool_3_argmax = self.pool_layer(conv_3_3)

            conv_4_1 = self.conv_layer(pool_3, [3, 3, 256, 512], 512, 'conv_4_1')
            conv_4_2 = self.conv_layer(conv_4_1, [3, 3, 512, 512], 512, 'conv_4_2')
            conv_4_3 = self.conv_layer(conv_4_2, [3, 3, 512, 512], 512, 'conv_4_3')

            pool_4, pool_4_argmax = self.pool_layer(conv_4_3)

            conv_5_1 = self.conv_layer(pool_4, [3, 3, 512, 512], 512, 'conv_5_1')
            conv_5_2 = self.conv_layer(conv_5_1, [3, 3, 512, 512], 512, 'conv_5_2')
            conv_5_3 = self.conv_layer(conv_5_2, [3, 3, 512, 512], 512, 'conv_5_3')

            pool_5, pool_5_argmax = self.pool_layer(conv_5_3)

            fc_6 = self.conv_layer(pool_5, [7, 7, 512, 4096], 4096, 'fc_6')
            fc_7 = self.conv_layer(fc_6, [1, 1, 4096, 4096], 4096, 'fc_7')

            deconv_fc_6 = self.deconv_layer(fc_7, [7, 7, 512, 4096], 512, 'fc6_deconv')

            unpool_5 = self.unpool_layer2x2(deconv_fc_6, pool_5_argmax)

            deconv_5_3 = self.deconv_layer(unpool_5, [3, 3, 512, 512], 512, 'deconv_5_3')
            deconv_5_2 = self.deconv_layer(deconv_5_3, [3, 3, 512, 512], 512, 'deconv_5_2')
            deconv_5_1 = self.deconv_layer(deconv_5_2, [3, 3, 512, 512], 512, 'deconv_5_1')

            unpool_4 = self.unpool_layer2x2(deconv_5_1, pool_4_argmax)

            deconv_4_3 = self.deconv_layer(unpool_4, [3, 3, 512, 512], 512, 'deconv_4_3')
            deconv_4_2 = self.deconv_layer(deconv_4_3, [3, 3, 512, 512], 512, 'deconv_4_2')
            deconv_4_1 = self.deconv_layer(deconv_4_2, [3, 3, 256, 512], 256, 'deconv_4_1')

            unpool_3 = self.unpool_layer2x2(deconv_4_1, pool_3_argmax)

            deconv_3_3 = self.deconv_layer(unpool_3, [3, 3, 256, 256], 256, 'deconv_3_3')
            deconv_3_2 = self.deconv_layer(deconv_3_3, [3, 3, 256, 256], 256, 'deconv_3_2')
            deconv_3_1 = self.deconv_layer(deconv_3_2, [3, 3, 128, 256], 128, 'deconv_3_1')

            unpool_2 = self.unpool_layer2x2(deconv_3_1, pool_2_argmax)

            deconv_2_2 = self.deconv_layer(unpool_2, [3, 3, 128, 128], 128, 'deconv_2_2')
            deconv_2_1 = self.deconv_layer(deconv_2_2, [3, 3, 64, 128], 64, 'deconv_2_1')

            unpool_1 = self.unpool_layer2x2(deconv_2_1, pool_1_argmax)

            deconv_1_2 = self.deconv_layer(unpool_1, [3, 3, 64, 64], 64, 'deconv_1_2')
            deconv_1_1 = self.deconv_layer(deconv_1_2, [3, 3, 32, 64], 32, 'deconv_1_1')

            score_1 = self.deconv_layer(deconv_1_1, [1, 1, 2, 32], 2, 'score_1')

            logits = tf.reshape(score_1, (-1, 2))
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                           labels=tf.reshape(expected, [-1]),
                                                                           name='x_entropy')

            self.loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')

            self.train_step = tf.train.AdamOptimizer(self.rate).minimize(self.loss)

            self.prediction = tf.argmax(tf.reshape(tf.nn.softmax(logits), tf.shape(score_1)), dimension=3)

            self.accuracy = tf.reduce_sum(tf.pow(self.prediction - expected[:, :, :, 0], 2))

    def weight_variable(self, shape, stddev):
        initial = tf.truncated_normal(shape, stddev=stddev)

        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)

        return tf.Variable(initial)

    def conv_layer(self, x, W_shape, b_shape, name, padding='SAME'):

        nr_units = functools.reduce(lambda x, y: x * y, W_shape)
        stddev = 1.0 / math.sqrt(float(nr_units))

        weights = self.weight_variable(W_shape, stddev)
        biases = self.bias_variable([b_shape])

        hidden = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding=padding)
        hidden = tf.add(hidden, biases)
        hidden = tf.nn.relu(hidden)

        return hidden

    def pool_layer(self, x):
        with tf.device('/gpu:0'):
            return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def deconv_layer(self, x, W_shape, b_shape, name, padding='SAME'):

        nr_units = functools.reduce(lambda x, y: x * y, W_shape)
        stddev = 1.0 / math.sqrt(float(nr_units))

        weights = self.weight_variable(W_shape, stddev)
        biases = self.bias_variable([b_shape])

        x_shape = tf.shape(x)
        out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])

        hidden = tf.nn.conv2d_transpose(x, weights, out_shape, [1, 1, 1, 1], padding=padding)
        hidden = tf.add(hidden, biases)

        return hidden

    def unravel_argmax(self, argmax, shape):
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])

        return tf.stack(output_list)

    def unpool_layer2x2(self, bottom, argmax):
        bottom_shape = tf.shape(bottom)
        top_shape = [bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3]]

        batch_size = top_shape[0]
        height = top_shape[1]
        width = top_shape[2]
        channels = top_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        argmax = self.unravel_argmax(argmax, argmax_shape)

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [batch_size * (width // 2) * (height // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, batch_size, height // 2, width // 2, 1])
        t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

        t2 = tf.to_int64(tf.range(batch_size))
        t2 = tf.tile(t2, [channels * (width // 2) * (height // 2)])
        t2 = tf.reshape(t2, [-1, batch_size])
        t2 = tf.transpose(t2, perm=[1, 0])
        t2 = tf.reshape(t2, [batch_size, channels, height // 2, width // 2, 1])

        t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

        t = tf.concat([t2, t3, t1], 4)
        indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

        x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])

        return tf.scatter_nd(indices, values, tf.to_int64(top_shape))


if __name__ == '__main__':
    train, val = sunnybrook.get_all_contours()
    segmenter = LVSegmentation()

    if len(sys.argv) != 2:
        print('The program must be run as : python3.5 step2_train_segmenter_v1.py [train|predict]')
        sys.exit(2)
    else:
        if sys.argv[1] == 'train':
            print('Run Train .....')

            segmenter.train(train)

        elif sys.argv[1] == 'predict':
            print('Run Predict .....')

            images, prepoces_images, labels = segmenter.read_data(val)
            prediction = segmenter.predict(prepoces_images)

            for i in range(len(images)):
                plt.imshow(images[i], cmap='gray')
                plt.show()

                plt.imshow(labels[i])
                plt.show()

                plt.imshow(prediction[i])
                plt.show()

        else:
            print('The available options for this script are : train, evaluate and predict')
            sys.exit(2)
