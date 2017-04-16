import os
import sys
import random
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import functools
from collections import OrderedDict
import utils.sunnybrook as sunnybrook


class LVSegmentation(object):
    def __init__(self, learning_rate, checkpoint_dir='../../result/segmenter/train_result/v3/'):
        self.x = tf.placeholder(tf.float32, shape=(None, 224, 224, 1))
        self.y = tf.placeholder(tf.int64, shape=(None, 224, 224))
        self.keep_prob = tf.placeholder(tf.float32)

        logits = self.__create_conv_net()
        self.cost = self.__cost_function(logits)
        self.optimizer = self.__adam_optimizer(learning_rate)
        self.prediction = self.__predict_function(logits)

        self.saver = tf.train.Saver(max_to_keep=30, keep_checkpoint_every_n_hours=1)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.checkpoint_dir = checkpoint_dir

    def predict(self, images):
        self.__restore_session()

        return self.prediction.eval(session=self.session, feed_dict={self.x: images, self.keep_prob: 1.})

    def train(self, train_paths, train_size, batch_size, epochs=30, dropout=0.75, restore_session=False):

        if restore_session:
            self.__restore_session()

        avg_gradients = None

        for epoch in range(epochs):
            total_loss = 0

            for step in range(0, train_size, batch_size):
                current_step = train_size * epoch + step

                train_path = train_paths[step:step + batch_size]
                _, images, labels = self.read_data(train_path)

                _, loss, lr, gradients = self.session.run(
                    (self.optimizer, self.cost, self.learning_rate_node, self.gradients_node),
                    feed_dict={self.x: images,
                               self.y: labels,
                               self.keep_prob: dropout})

                if avg_gradients is None:
                    avg_gradients = [np.zeros_like(gradient) for gradient in gradients]

                for i in range(len(gradients)):
                    avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (current_step + 1)))) + (
                        gradients[i] / (current_step + 1))

                norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                self.norm_gradients_node.assign(norm_gradients).eval(session=self.session)

                total_loss += loss

            self.__output_epoch_stats(epoch, total_loss, train_size)

            self.__save(epoch)

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

    def __create_conv_net(self, layers=5, features_root=32, filter_size=3, pool_size=2):

        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()

        in_size = 1000
        size = in_size

        in_node = self.x

        # down layers
        for layer in range(0, layers):
            features = 2 ** layer * features_root

            if layer == 0:
                w1 = self.__weight_variable([filter_size, filter_size, 1, features])
            else:
                w1 = self.__weight_variable([filter_size, filter_size, features // 2, features])

            w2 = self.__weight_variable([filter_size, filter_size, features, features])

            b1 = self.__variable([features], 0.0)
            b2 = self.__variable([features], 0.0)

            conv1 = self.__conv_layer(in_node, w1, self.keep_prob)
            tmp_h_conv = tf.nn.relu(conv1 + b1)
            conv2 = self.__conv_layer(tmp_h_conv, w2, self.keep_prob)
            dw_h_convs[layer] = tf.nn.relu(conv2 + b2)

            size -= 4
            if layer < layers - 1:
                pools[layer] = self.__max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= 2

        in_node = dw_h_convs[layers - 1]

        # up layers
        for layer in range(layers - 2, -1, -1):
            features = 2 ** (layer + 1) * features_root

            wd = self.__weight_variable([pool_size, pool_size, features // 2, features])
            bd = self.__variable([features // 2], 0.0)
            h_deconv = tf.nn.relu(self.__deconv_layer(in_node, wd, pool_size) + bd)
            h_deconv_concat = self.__crop_and_concat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat

            w1 = self.__weight_variable([filter_size, filter_size, features, features // 2])
            w2 = self.__weight_variable([filter_size, filter_size, features // 2, features // 2])
            b1 = self.__variable([features // 2], 0.0)
            b2 = self.__variable([features // 2], 0.0)

            conv1 = self.__conv_layer(h_deconv_concat, w1, self.keep_prob)
            h_conv = tf.nn.relu(conv1 + b1)
            conv2 = self.__conv_layer(h_conv, w2, self.keep_prob)
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node

            size *= 2
            size -= 4

        # Output Map
        weight = self.__weight_variable([1, 1, features_root, 2])
        bias = self.__variable([2], 0.0)
        conv = self.__conv_layer(in_node, weight, tf.constant(1.0))
        output_map = tf.nn.relu(conv + bias)
        up_h_convs["out"] = output_map

        return output_map

    def __adam_optimizer(self, learning_rate):
        self.learning_rate_node = tf.Variable(learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node) \
            .minimize(self.cost)

        return optimizer

    def __predict_function(self, logits):
        flat_logits = tf.reshape(logits, (-1, 2))

        return tf.argmax(tf.reshape(tf.nn.softmax(flat_logits), tf.shape(logits)), dimension=3)

    def __cost_function(self, logits):
        flat_logits = tf.reshape(logits, (-1, 2))
        expected = tf.expand_dims(self.y, -1)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                       labels=tf.reshape(expected, [-1]))
        loss = tf.reduce_mean(cross_entropy)

        return loss

    def __weight_variable(self, shape):
        nr_units = functools.reduce(lambda x, y: x * y, shape)
        stddev = 1.0 / math.sqrt(float(nr_units))

        initial = tf.truncated_normal(shape, stddev=stddev)

        return tf.Variable(initial)

    def __variable(self, shape, constant):
        initial = tf.constant(constant, shape=shape)

        return tf.Variable(initial)

    def __conv_layer(self, x, weights, keep_prob_):
        hidden = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')

        return tf.nn.dropout(hidden, keep_prob_)

    def __deconv_layer(self, x, weights, stride):

        x_shape = tf.shape(x)
        out_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])

        return tf.nn.conv2d_transpose(x, weights, out_shape, strides=[1, stride, stride, 1], padding='SAME')

    def __max_pool(self, x, n):
        return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

    def __crop_and_concat(self, x1, x2):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

    def __output_epoch_stats(self, epoch, total_loss, training_iters):
        print("Epoch {:}, Average loss: {:.4f}".format(epoch, (total_loss / training_iters)))

    def __save(self, epoch):
        self.saver.save(self.session, self.checkpoint_dir + 'model', global_step=epoch)

    def __restore_session(self):

        if not os.path.exists(self.checkpoint_dir):
            raise IOError(self.checkpoint_dir + ' does not exist.')
        else:
            path = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if path is None:
                raise IOError('No checkpoint to restore in ' + self.checkpoint_dir)
            else:
                self.saver.restore(self.session, path.model_checkpoint_path)


if __name__ == '__main__':
    train, val = sunnybrook.get_all_contours()
    segmenter = LVSegmentation(1e-3)

    if len(sys.argv) != 2:
        print('The program must be run as : python3.5 step2_train_segmenter_v3.py [train|predict]')
        sys.exit(2)
    else:
        if sys.argv[1] == 'train':
            print('Run Train .....')

            segmenter.train(train, 800, 2, epochs=30)

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
            print('The available options for this script are : train and predict')
            sys.exit(2)
