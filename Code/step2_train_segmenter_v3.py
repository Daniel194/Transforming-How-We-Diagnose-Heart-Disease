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
from collections import OrderedDict
import utils.sunnybrook as sunnybrook


class LVSegmentation(object):
    def __init__(self, learning_rate, checkpoint_dir='../../result/segmenter/train_result/v3/'):
        random.seed(time.time())

        self.x = tf.placeholder(tf.float32, shape=(None, 224, 224, 1))
        self.y = tf.placeholder(tf.int64, shape=(None, 224, 224))
        self.keep_prob = tf.placeholder(tf.float32)

        logits, self.variables, self.offset = self.create_conv_net()
        self.cost = self.cost_function(logits)

        self.gradients_node = tf.gradients(self.cost, self.variables)

        self.cross_entropy = tf.reduce_mean(
            self.cross_entropy(tf.reshape(self.y, [-1]), tf.reshape(self.pixel_wise_softmax_2(logits), [-1, 2])))

        self.predicter = self.pixel_wise_softmax_2(logits)
        self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.global_step = tf.Variable(0)

        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.gradients_node)]))

        tf.summary.scalar('loss', self.cost)
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        tf.summary.scalar('accuracy', self.accuracy)

        self.optimizer = self.adam_optimizer(learning_rate)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.checkpoint_dir = checkpoint_dir

    def predict(self, model_path, x_test):
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], 2))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})

        return prediction

    def save(self, sess, model_path):
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

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

    def predict(self, images):
        self.restore_session()

        return self.prediction.eval(session=self.session, feed_dict={self.x: images})

    def evaluate(self, eval_paths):
        self.restore_session()

        _, images, labels = self.read_data(eval_paths)

        accuracy = self.accuracy.eval(session=self.session, feed_dict={self.x: images, self.y: labels})

        print('Model has accuracy : {:.6f} '.format(accuracy))

    def train(self, train_paths, epochs=100, dropout=0.75, restore_session=False, display_step=100):

        save_path = os.path.join(self.checkpoint_dir, "model.cpkt")

        if epochs == 0:
            return save_path

        if restore_session:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                self.restore(ckpt.model_checkpoint_path)

        summary_writer = tf.summary.FileWriter(self.checkpoint_dir, graph=self.session)

        avg_gradients = None

        for epoch in range(epochs):
            total_loss = 0

            for step in range(0, 800, 2):
                current_step = 800 * epoch + step

                train_path = train_paths[step:step + 2]
                _, images, labels = self.read_data(train_path)

                # Run optimization op (backprop)
                _, loss, lr, gradients = self.session.run(
                    (self.optimizer, self.cost, self.learning_rate_node, self.gradients_node),
                    feed_dict={self.x: images,
                               self.y: labels,
                               self.keep_prob: dropout})

                if avg_gradients is None:
                    avg_gradients = [np.zeros_like(gradient) for gradient in gradients]

                for i in range(len(gradients)):
                    avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step + 1)))) + (gradients[i] / (step + 1))

                norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                self.norm_gradients_node.assign(norm_gradients).eval()

                if current_step % display_step == 0:
                    self.output_minibatch_stats(self.session, summary_writer, step, images, labels)

                total_loss += loss

            self.output_epoch_stats(epoch, total_loss, 800, lr)

            save_path = self.save(save_path)

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

    def create_conv_net(self, layers=3, features_root=16, filter_size=3, pool_size=2):

        weights = []
        biases = []
        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()

        in_size = 1000
        size = in_size

        expected = tf.expand_dims(self.y, -1)
        self.rate = tf.placeholder(tf.float32, shape=[])

        in_node = self.x

        # down layers
        for layer in range(0, layers):
            features = 2 ** layer * features_root

            if layer == 0:
                w1 = self.weight_variable([filter_size, filter_size, 1, features])
            else:
                w1 = self.weight_variable([filter_size, filter_size, features // 2, features])

            w2 = self.weight_variable([filter_size, filter_size, features, features])

            b1 = self.variable([features], 0.0)
            b2 = self.variable([features], 0.0)

            conv1 = self.conv_layer(in_node, w1, self.keep_prob)
            tmp_h_conv = tf.nn.relu(conv1 + b1)
            conv2 = self.conv_layer(tmp_h_conv, w2, self.keep_prob)
            dw_h_convs[layer] = tf.nn.relu(conv2 + b2)

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size -= 4
            if layer < layers - 1:
                pools[layer] = self.max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= 2

        in_node = dw_h_convs[layers - 1]

        # up layers
        for layer in range(layers - 2, -1, -1):
            features = 2 ** (layer + 1) * features_root

            wd = self.weight_variable([pool_size, pool_size, features // 2, features])
            bd = self.variable([features // 2], 0.0)
            h_deconv = tf.nn.relu(self.deconv_layer(in_node, wd, pool_size) + bd)
            h_deconv_concat = self.crop_and_concat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat

            w1 = self.weight_variable([filter_size, filter_size, features, features // 2])
            w2 = self.weight_variable([filter_size, filter_size, features // 2, features // 2])
            b1 = self.variable([features // 2], 0.0)
            b2 = self.variable([features // 2], 0.0)

            conv1 = self.conv_layer(h_deconv_concat, w1, self.keep_prob)
            h_conv = tf.nn.relu(conv1 + b1)
            conv2 = self.conv_layer(h_conv, w2, self.keep_prob)
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size *= 2
            size -= 4

        # Output Map
        weight = self.weight_variable([1, 1, features_root, 2])
        bias = self.variable([2], 0.0)
        conv = self.conv_layer(in_node, weight, tf.constant(1.0))
        output_map = tf.nn.relu(conv + bias)
        up_h_convs["out"] = output_map

        variables = []
        for w1, w2 in weights:
            variables.append(w1)
            variables.append(w2)

        for b1, b2 in biases:
            variables.append(b1)
            variables.append(b2)

        return output_map, variables, int(in_size - size)

    def adam_optimizer(self, learning_rate):
        self.learning_rate_node = tf.Variable(learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node) \
            .minimize(self.cost, global_step=self.global_step)

        return optimizer

    def cost_function(self, logits):
        flat_logits = tf.reshape(logits, [-1, 2])
        flat_labels = tf.reshape(self.y, [-1])

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels))

        regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
        loss += (0.5 * regularizers)

        return loss

    def weight_variable(self, shape):
        nr_units = functools.reduce(lambda x, y: x * y, shape)
        stddev = 1.0 / math.sqrt(float(nr_units))

        initial = tf.truncated_normal(shape, stddev=stddev)

        return tf.Variable(initial)

    def variable(self, shape, constant):
        initial = tf.constant(constant, shape=shape)

        return tf.Variable(initial)

    def conv_layer(self, x, weights, keep_prob_):
        hidden = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='VALID')

        return tf.nn.dropout(hidden, keep_prob_)

    def deconv_layer(self, x, weights, stride):

        x_shape = tf.shape(x)
        out_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])

        return tf.nn.conv2d_transpose(x, weights, out_shape, strides=[1, stride, stride, 1], padding='VALID')

    def max_pool(self, x, n):
        return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

    def batch_normalization(self, hidden, v_shape):
        scale = self.variable([v_shape], 1.0)
        beta = self.variable([v_shape], 0.0)

        batch_mean, batch_var = tf.nn.moments(hidden, [0])

        return tf.nn.batch_normalization(hidden, batch_mean, batch_var, beta, scale, 1e-3)

    def crop_and_concat(self, x1, x2):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

    def pixel_wise_softmax(self, output_map):
        exponential_map = tf.exp(output_map)
        evidence = tf.add(exponential_map, tf.reverse(exponential_map, [False, False, False, True]))
        return tf.div(exponential_map, evidence, name="pixel_wise_softmax")

    def pixel_wise_softmax_2(self, output_map):
        exponential_map = tf.exp(output_map)
        sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
        tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
        return tf.div(exponential_map, tensor_sum_exp)

    def cross_entropy(self, y_, output_map):
        return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, predictions = sess.run([self.summary_op,
                                                        self.cost,
                                                        self.accuracy,
                                                        self.predicter],
                                                       feed_dict={self.x: batch_x,
                                                                  self.y: batch_y,
                                                                  self.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        print("Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))


if __name__ == '__main__':
    train, val = sunnybrook.get_all_contours()
    segmenter = LVSegmentation()

    if len(sys.argv) != 2:
        print('The program must be run as : python3.5 step2_train_segmenter_v3.py [train|predict]')
        sys.exit(2)
    else:
        if sys.argv[1] == 'train':
            print('Run Train .....')

            segmenter.train(train, training_steps=10000)

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
