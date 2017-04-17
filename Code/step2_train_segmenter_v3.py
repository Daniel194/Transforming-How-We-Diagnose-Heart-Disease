import os
import sys
import random
import pickle
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import functools
import utils.sunnybrook as sunnybrook


class LVSegmentation(object):
    def __init__(self, checkpoint_dir='../../result/segmenter/train_result/v3/'):
        self.__build_con_net()
        self.__saver = tf.train.Saver(max_to_keep=30, keep_checkpoint_every_n_hours=1)
        self.__session = tf.Session()
        self.__session.run(tf.global_variables_initializer())

        self.__checkpoint_dir = checkpoint_dir
        self.__loss_array = []

    def predict(self, images):
        self.__restore_session()

        return self.__prediction.eval(session=self.__session, feed_dict={self.x: images, self.keep_prob: 1.0})

    def train(self, train_paths, epochs=30, batch_size=2, restore_session=False, learning_rate=1e-6, dropout=0.75):
        if restore_session:
            self.__restore_session()

        train_size = len(train_paths)

        for epoch in range(epochs):
            total_loss = 0

            for step in range(0, train_size, batch_size):
                train_path = train_paths[step:step + batch_size]
                _, images, labels = self.read_data(train_path)

                self.__train_step.run(session=self.__session,
                                      feed_dict={self.x: images, self.y: labels, self.rate: learning_rate,
                                                 self.keep_prob: dropout})

                loss = self.__loss.eval(session=self.__session,
                                        feed_dict={self.x: images, self.y: labels, self.keep_prob: 1.0})

                total_loss += loss

            print('Epoch {} - Loss : {:.6f}'.format(epoch, total_loss / train_size))

            self.__saver.save(self.__session, self.__checkpoint_dir + 'model', global_step=epoch)

            self.__loss_array.append(total_loss / train_size)
            self.__save_loss()

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

    def __build_con_net(self):
        self.x = tf.placeholder(tf.float32, shape=(None, 224, 224, 1))
        self.y = tf.placeholder(tf.int64, shape=(None, 224, 224))
        self.keep_prob = tf.placeholder(tf.float32)
        self.rate = tf.placeholder(tf.float32, shape=[])

        expected = tf.expand_dims(self.y, -1)

        conv_1_1 = self.__conv_layer(self.x, [3, 3, 1, 32], 32, self.keep_prob)
        conv_1_2 = self.__conv_layer(conv_1_1, [3, 3, 32, 32], 32, self.keep_prob)

        pool_1 = self.__max_pool(conv_1_2)

        conv_2_1 = self.__conv_layer(pool_1, [3, 3, 32, 64], 64, self.keep_prob)
        conv_2_2 = self.__conv_layer(conv_2_1, [3, 3, 64, 64], 64, self.keep_prob)

        pool_2 = self.__max_pool(conv_2_2)

        conv_3_1 = self.__conv_layer(pool_2, [3, 3, 64, 128], 128, self.keep_prob)
        conv_3_2 = self.__conv_layer(conv_3_1, [3, 3, 128, 128], 128, self.keep_prob)

        pool_3 = self.__max_pool(conv_3_2)

        conv_4_1 = self.__conv_layer(pool_3, [3, 3, 128, 256], 256, self.keep_prob)
        conv_4_2 = self.__conv_layer(conv_4_1, [3, 3, 256, 256], 256, self.keep_prob)

        pool_4 = self.__max_pool(conv_4_2)

        conv_5_1 = self.__conv_layer(pool_4, [3, 3, 256, 512], 512, self.keep_prob)
        conv_5_2 = self.__conv_layer(conv_5_1, [3, 3, 512, 512], 512, self.keep_prob)

        deconv_4 = self.__deconv_layer(conv_5_2, [2, 2, 256, 512], 256)

        deconv_4_concat = self.__crop_and_concat(conv_4_2, deconv_4)

        conv_6_1 = self.__conv_layer(deconv_4_concat, [3, 3, 512, 256], 256, self.keep_prob)
        conv_6_2 = self.__conv_layer(conv_6_1, [3, 3, 256, 256], 256, self.keep_prob)

        deconv_3 = self.__deconv_layer(conv_6_2, [2, 2, 128, 256], 128)

        deconv_3_concat = self.__crop_and_concat(conv_3_2, deconv_3)

        conv_7_1 = self.__conv_layer(deconv_3_concat, [3, 3, 256, 128], 128, self.keep_prob)
        conv_7_2 = self.__conv_layer(conv_7_1, [3, 3, 128, 128], 128, self.keep_prob)

        deconv_2 = self.__deconv_layer(conv_7_2, [2, 2, 64, 128], 64)

        deconv_2_concat = self.__crop_and_concat(conv_2_2, deconv_2)

        conv_8_1 = self.__conv_layer(deconv_2_concat, [3, 3, 128, 64], 64, self.keep_prob)
        conv_8_2 = self.__conv_layer(conv_8_1, [3, 3, 64, 64], 64, self.keep_prob)

        deconv_1 = self.__deconv_layer(conv_8_2, [2, 2, 32, 64], 32)

        deconv_1_concat = self.__crop_and_concat(conv_1_2, deconv_1)

        conv_9_1 = self.__conv_layer(deconv_1_concat, [3, 3, 64, 32], 32, self.keep_prob)
        conv_9_2 = self.__conv_layer(conv_9_1, [3, 3, 32, 32], 32, self.keep_prob)

        output = self.__conv_layer(conv_9_2, [1, 1, 32, 2], 2, tf.constant(1.0))

        logits = tf.reshape(output, (-1, 2))
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=tf.reshape(expected, [-1]),
                                                                       name='x_entropy')

        self.__loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')

        self.__train_step = tf.train.AdamOptimizer(self.rate).minimize(self.__loss)

        self.__prediction = tf.argmax(tf.reshape(tf.nn.softmax(logits), tf.shape(output)), dimension=3)

    def __weight_variable(self, shape):
        nr_units = functools.reduce(lambda x, y: x * y, shape)
        stddev = 1.0 / math.sqrt(float(nr_units))

        initial = tf.truncated_normal(shape, stddev=stddev)

        return tf.Variable(initial)

    def __bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)

        return tf.Variable(initial)

    def __conv_layer(self, x, w_shape, b_shape, keep_prob):

        weights = self.__weight_variable(w_shape)
        biases = self.__bias_variable([b_shape])

        hidden = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
        hidden = tf.add(hidden, biases)
        hidden = tf.nn.relu(hidden)
        hidden = tf.nn.dropout(hidden, keep_prob)

        return hidden

    def __deconv_layer(self, x, w_shape, b_shape):

        weights = self.__weight_variable(w_shape)
        biases = self.__bias_variable([b_shape])

        x_shape = tf.shape(x)
        out_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])

        hidden = tf.nn.conv2d_transpose(x, weights, out_shape, strides=[1, 2, 2, 1], padding='SAME')
        hidden = tf.add(hidden, biases)
        hidden = tf.nn.relu(hidden)

        return hidden

    def __max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def __crop_and_concat(self, x1, x2):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

    def __restore_session(self):
        if not os.path.exists(self.__checkpoint_dir):
            raise IOError(self.__checkpoint_dir + ' does not exist.')
        else:
            path = tf.train.get_checkpoint_state(self.__checkpoint_dir)
            if path is None:
                raise IOError('No checkpoint to restore in ' + self.__checkpoint_dir)
            else:
                self.__saver.restore(self.__session, path.model_checkpoint_path)

        with open(self.__checkpoint_dir + 'loss.pickle', 'rb') as f:
            self.__loss_array = pickle.load(f)

    def __save_loss(self):
        if os.path.exists(self.__checkpoint_dir + 'loss.pickle'):
            os.remove(self.__checkpoint_dir + 'loss.pickle')

        with open(self.__checkpoint_dir + 'loss.pickle', 'wb') as f:
            pickle.dump(self.__loss_array, f)


if __name__ == '__main__':
    train, val = sunnybrook.get_all_contours()
    segmenter = LVSegmentation()

    if len(sys.argv) != 2:
        print('The program must be run as : python3.5 step2_train_segmenter_v3.py [train|predict]')
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
            print('The available options for this script are : train and predict')
            sys.exit(2)
