from datetime import datetime
import numpy as np
import tensorflow as tf
import functools
import random
import time
import math
import os
import sys
import re
import utils.sunnybrook as sunnybrook


class LVSegmentation(object):
    def __init__(self, train, val):
        """
        Initialize the parameters for neural network
        :param train: all trains contours path
        :param val: all validates contours path
        """

        # Initialize random
        np.random.seed(time.time())
        random.seed(time.time())

        # Image constant variables
        self.IMAGE_INIT_SIZE = 252
        self.IMAGE_SIZE = 224
        self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = len(train)
        self.TRAIN = train
        self.VAL = val

        # Constants describing the training process.
        self.BATCH_SIZE = 50  # Batch size per iteration
        self.NR_EPOCHS = 2000  # Number of epoch
        self.STEPS = 6000  # Number of steps

        self.MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
        self.NUM_EPOCHS_PER_DECAY = 200.0  # Epochs after which learning rate decays.
        self.LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
        self.INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.
        self.EPSILON = 1e-3  # Hyperparamter for Batch Normalization.

        self.TOWER_NAME = 'tower'

    def train(self):
        """
        Train Segmenter for a number of steps.
        :return: Nothing
        """

        with tf.Graph().as_default():
            global_step = tf.contrib.framework.get_or_create_global_step()

            # Get images and labels for Segmenter
            images, labels = self.__distorted_inputs(self.TRAIN)

            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = self.__inference(images)

            # Calculate loss.
            loss = self.__loss(logits, labels)

            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            train_op = self.__train(loss, global_step)

            class _LoggerHook(tf.train.SessionRunHook):
                """Logs loss and runtime."""

                def begin(self):
                    self._step = -1

                def before_run(self, run_context):
                    self._step += 1
                    self._start_time = time.time()

                    return tf.train.SessionRunArgs(loss)  # Asks for loss value.

                def after_run(self, run_context, run_values):
                    duration = time.time() - self._start_time
                    loss_value = run_values.results

                    if self._step % 10 == 0:
                        sec_per_batch = float(duration)

                        format_str = '%s: step %d, loss = %.2f (%.3f  sec/batch)'
                        print(format_str % (datetime.now(), self._step, loss_value, sec_per_batch))

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=FLAGS.train_dir,
                    hooks=[tf.train.StopAtStepHook(last_step=self.STEPS), tf.train.NanTensorHook(loss),
                           _LoggerHook()],
                    config=tf.ConfigProto(log_device_placement=False)) as mon_sess:
                while not mon_sess.should_stop():
                    mon_sess.run(train_op)

    def evaluate(self):
        """
        Eval CIFAR-10 for a number of steps.
        :return: Nothing.
        """

        with tf.Graph().as_default() as g:
            # Get images and labels for CIFAR-10.
            eval_data = FLAGS.eval_data == 'test'
            images, labels = self._inputs(eval_data=eval_data)

            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = self.__inference(images)

            # Calculate predictions.
            top_k_op = tf.nn.in_top_k(logits, labels, 1)

            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()

            summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

            while True:
                self.__eval_once(saver, summary_writer, top_k_op, summary_op)

                if FLAGS.run_once:
                    break

                time.sleep(FLAGS.eval_interval_secs)

    def __eval_once(self, saver, summary_writer, top_k_op, summary_op):
        """
        Run Eval once.
        :param saver:  Saver.
        :param summary_writer: Summary writer.
        :param top_k_op: Top K op.
        :param summary_op: Summary op.
        :return: Nothing
        """

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()

            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0

                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1

                # Compute precision @ 1.
                precision = true_count / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=precision)
                summary_writer.add_summary(summary, global_step)
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

    def __inputs(self, eval_data, data_dir, batch_size):
        """
        Construct input for CIFAR evaluation using the Reader ops.
        :param eval_data: bool, indicating if one should use the train or eval data set.
        :param data_dir: Path to the CIFAR-10 data directory.
        :param batch_size: Number of images per batch.
        :return: images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
                 labels: Labels. 1D tensor of [batch_size] size.
        """

        if not eval_data:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
            num_examples_per_epoch = self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]
            num_examples_per_epoch = self.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue.
        read_input = self.__read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = self.IMAGE_SIZE
        width = self.IMAGE_SIZE

        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(resized_image)

        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

        # Generate a batch of images and labels by building up a queue of examples.
        return self.__generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size,
                                                     shuffle=False)

    def __activation_summary(self, x):
        """
        Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.
        :param x: Tensor
        :return: nothing
        """

        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training session.
        # This helps the clarity of presentation on tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % self.TOWER_NAME, '', x.op.name)
        tf.contrib.deprecated.histogram_summary(tensor_name + '/activations', x)
        tf.contrib.deprecated.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def __variable_on_cpu(self, name, shape, initializer):
        """
        Helper to create a Variable stored on CPU memory.
        :param name: name of the variable
        :param shape: list of ints
        :param initializer: initializer for Variable
        :return: Variable Tensor
        """

        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)

        return var

    def __variable_with_weight_decay(self, name, shape, stddev, wd):
        """
        Helper to create an initialized Variable with weight decay.
        A weight decay is added only if one is specified.
        :param name: name of the variable
        :param shape: list of ints
        :param stddev: standard deviation of a truncated Gaussian
        :param wd: add L2Loss weight decay multiplied by this float. If None, weight decay is not added for this Variable.
        :return: Variable Tensor
        """

        var = self.__variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))

        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        return var

    def __distorted_inputs(self, train_ctrs):
        """
        Construct distorted input for Sunnybrook training data.
        :param train_ctrs: an array of [batch_size] which contains paths to images and labels.
        :return: images: Images. 3D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE] size.
                 labels: Labels. 3D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE] size.
        """

        np.random.shuffle(train_ctrs)

        for i in range(int(np.ceil(len(train_ctrs) / float(self.BATCH_SIZE)))):
            batch = train_ctrs[(self.BATCH_SIZE * i):(self.BATCH_SIZE * (i + 1))]

            images, labels = sunnybrook.export_all_contours(batch)

            images = tf.cast(images, tf.float32)
            labels = tf.cast(images, tf.int16)

            # Randomly crop a [height, width] section of the image.
            crop_max = (self.IMAGE_INIT_SIZE - self.IMAGE_SIZE) / 2
            crop_max = int(crop_max)

            crop_x = random.randint(0, crop_max)
            crop_y = random.randint(0, crop_max)

            images = images[:, crop_y:crop_y + self.IMAGE_SIZE, crop_x: crop_x + self.IMAGE_SIZE]
            labels = labels[:, crop_y:crop_y + self.IMAGE_SIZE, crop_x: crop_x + self.IMAGE_SIZE]

            # Subtract off the mean and divide by the variance of the pixels.
            images = tf.image.per_image_standardization(images)

            yield images, labels

    def _inputs(self, eval_data):
        """
        Construct input for CIFAR evaluation using the Reader ops.
        :param eval_data: bool, indicating if one should use the train or eval data set.
        :return: images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
                 labels: Labels. 1D tensor of [batch_size] size.
        """

        if not FLAGS.data_dir:
            raise ValueError('Please supply a data_dir')

        data_dir = os.path.join(FLAGS.data_dir)
        images, labels = self.__inputs(eval_data=eval_data, data_dir=data_dir, batch_size=FLAGS.batch_size)

        if FLAGS.use_fp16:
            images = tf.cast(images, tf.float16)
            labels = tf.cast(labels, tf.float16)

        return images, labels

    def __conv_layer(self, name, x, W_shape, b_shape, padding='SAME'):
        """
        Convolutional Layer
        :param name: the name of the layer
        :param x: the data
        :param W_shape: the shape of the weights
        :param b_shape: the shape of the biases
        :param padding: padding time
        :return: return the next hidden layer
        """

        with tf.variable_scope(name):
            nr_units = functools.reduce(lambda x, y: x * y, W_shape)
            weights = self.__variable_with_weight_decay('weights', shape=W_shape,
                                                        stddev=1.0 / math.sqrt(float(nr_units)), wd=0.0)
            biases = self.__variable_on_cpu('biases', b_shape, tf.constant_initializer(0.0))

            hidden = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding=padding)
            hidden = tf.add(hidden, biases)
            hidden = tf.nn.relu(hidden)
            self.__activation_summary(hidden)

        return hidden

    def __deconv_layer(self, name, x, W_shape, b_shape, padding='SAME'):
        """
        Deconvolutional Layer
        :param name: the name of the layer
        :param x: the date
        :param W_shape: the shape of the weights
        :param b_shape:the shape of the biases
        :param padding: the padding time
        :return: the next hidden layer
        """

        with tf.variable_scope(name):
            nr_units = functools.reduce(lambda x, y: x * y, W_shape)
            weights = self.__variable_with_weight_decay('weights', shape=W_shape,
                                                        stddev=1.0 / math.sqrt(float(nr_units)), wd=0.0)
            biases = self.__variable_on_cpu('biases', b_shape, tf.constant_initializer(0.0))

            x_shape = tf.shape(x)
            out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
            hidden = tf.nn.conv2d_transpose(x, weights, out_shape, [1, 1, 1, 1], padding=padding)
            hidden = tf.add(hidden, biases)
            self.__activation_summary(hidden)

        return hidden

    def __pool_layer(self, x):
        """
        Pool Layer
        :param x: the data
        :return: the data after max pool layer
        """

        return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def __unravel_argmax(self, argmax, shape):
        """
        Unravel the argmax
        :param argmax: the argmax
        :param shape: the shape
        :return: argmax
        """

        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])

        return tf.pack(output_list)

    def __unpool_layer2x2(self, bottom, argmax):
        """
        Unpool Layer
        :param bottom: the data
        :param argmax: the position of the argument max of the previous max pool layer
        :return: the hidden layer
        """

        bottom_shape = tf.shape(bottom)
        top_shape = [bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3]]

        batch_size = top_shape[0]
        height = top_shape[1]
        width = top_shape[2]
        channels = top_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        argmax = self.__unravel_argmax(argmax, argmax_shape)

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

        t = tf.concat(4, [t2, t3, t1])
        indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

        x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])

        return tf.scatter_nd(indices, values, tf.to_int64(top_shape))

    def __inference(self, images):
        """
        Build the Segmentation module
        :param images: Images returned from distorted_inputs() or inputs().
        :return: Logits.
        """

        conv_1_1 = self.__conv_layer('conv_1_1', images, [3, 3, 1, 64], 64)
        conv_1_2 = self.__conv_layer('conv_1_2', conv_1_1, [3, 3, 64, 64], 64)

        pool_1, pool_1_argmax = self.__pool_layer(conv_1_2)

        conv_2_1 = self.__conv_layer('conv_2_1', pool_1, [3, 3, 64, 128], 128)
        conv_2_2 = self.__conv_layer('conv_2_2', conv_2_1, [3, 3, 128, 128], 128)

        pool_2, pool_2_argmax = self.__pool_layer(conv_2_2)

        conv_3_1 = self.__conv_layer('conv_3_1', pool_2, [3, 3, 128, 256], 256)
        conv_3_2 = self.__conv_layer('conv_3_2', conv_3_1, [3, 3, 256, 256], 256)
        conv_3_3 = self.__conv_layer('conv_3_3', conv_3_2, [3, 3, 256, 256], 256)

        pool_3, pool_3_argmax = self.__pool_layer(conv_3_3)

        conv_4_1 = self.__conv_layer('conv_4_1', pool_3, [3, 3, 256, 512], 512)
        conv_4_2 = self.__conv_layer('conv_4_2', conv_4_1, [3, 3, 512, 512], 512)
        conv_4_3 = self.__conv_layer('conv_4_3', conv_4_2, [3, 3, 512, 512], 512)

        pool_4, pool_4_argmax = self.__pool_layer(conv_4_3)

        conv_5_1 = self.__conv_layer('conv_5_1', pool_4, [3, 3, 512, 512], 512)
        conv_5_2 = self.__conv_layer('conv_5_2', conv_5_1, [3, 3, 512, 512], 512)
        conv_5_3 = self.__conv_layer('conv_5_3', conv_5_2, [3, 3, 512, 512], 512)

        pool_5, pool_5_argmax = self.__pool_layer(conv_5_3)

        fc_6 = self.__conv_layer('fc_6', pool_5, [7, 7, 512, 4096], 4096)
        fc_7 = self.__conv_layer('fc_7', fc_6, [1, 1, 4096, 4096], 4096)

        deconv_fc_6 = self.__deconv_layer('fc6_deconv', fc_7, [7, 7, 512, 4096], 512)

        unpool_5 = self.__unpool_layer2x2(deconv_fc_6, pool_5_argmax)

        deconv_5_3 = self.__deconv_layer('deconv_5_3', unpool_5, [3, 3, 512, 512], 512)
        deconv_5_2 = self.__deconv_layer('deconv_5_2', deconv_5_3, [3, 3, 512, 512], 512)
        deconv_5_1 = self.__deconv_layer('deconv_5_1', deconv_5_2, [3, 3, 512, 512], 512)

        unpool_4 = self.__unpool_layer2x2(deconv_5_1, pool_4_argmax)

        deconv_4_3 = self.__deconv_layer('deconv_4_3', unpool_4, [3, 3, 512, 512], 512)
        deconv_4_2 = self.__deconv_layer('deconv_4_2', deconv_4_3, [3, 3, 512, 512], 512)
        deconv_4_1 = self.__deconv_layer('deconv_4_1', deconv_4_2, [3, 3, 256, 512], 256)

        unpool_3 = self.__unpool_layer2x2(deconv_4_1, pool_3_argmax)

        deconv_3_3 = self.__deconv_layer('deconv_3_3', unpool_3, [3, 3, 256, 256], 256)
        deconv_3_2 = self.__deconv_layer('deconv_3_2', deconv_3_3, [3, 3, 256, 256], 256)
        deconv_3_1 = self.__deconv_layer('deconv_3_1', deconv_3_2, [3, 3, 128, 256], 128)

        unpool_2 = self.__unpool_layer2x2(deconv_3_1, pool_2_argmax)

        deconv_2_2 = self.__deconv_layer('deconv_2_2', unpool_2, [3, 3, 128, 128], 128)
        deconv_2_1 = self.__deconv_layer('deconv_2_1', deconv_2_2, [3, 3, 64, 128], 64)

        unpool_1 = self.__unpool_layer2x2(deconv_2_1, pool_1_argmax)

        deconv_1_2 = self.__deconv_layer('deconv_1_2', unpool_1, [3, 3, 64, 64], 64)
        deconv_1_1 = self.__deconv_layer('deconv_1_1', deconv_1_2, [3, 3, 32, 64], 32)

        score_1 = self.__deconv_layer('score_1', deconv_1_1, [1, 1, 2, 32], 2)

        logits = tf.reshape(score_1, (-1, 2))

        return logits

    def __loss(self, logits, labels):
        """
        Add L2Loss to all the trainable variables.
        Add summary for "Loss" and "Loss/avg".
        :param logits: Logits from inference().
        :param labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]
        :return: Loss tensor of type float.
        """

        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(tf.reshape(labels, [-1]), tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                       name='cross_entropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def __add_loss_summaries(self, total_loss):
        """
        Add summaries for losses in CIFAR-10 model.
        Generates moving average for all losses and associated summaries for visualizing the performance of the network.
        :param total_loss: Total loss from loss().
        :return: op for generating moving averages of losses.
        """

        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the same for the averaged
        # version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.contrib.deprecated.scalar_summary(l.op.name + ' (raw)', l)
            tf.contrib.deprecated.scalar_summary(l.op.name, loss_averages.average(l))

        return loss_averages_op

    def __train(self, total_loss, global_step):
        """
        Train Segmentation model.
        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.
        :param total_loss: Total loss from loss().
        :param global_step: Integer Variable counting the number of training steps processed.
        :return: op for training.
        """

        # Variables that affect learning rate.
        num_batches_per_epoch = self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(self.INITIAL_LEARNING_RATE, global_step, decay_steps,
                                        self.LEARNING_RATE_DECAY_FACTOR, staircase=True)
        tf.contrib.deprecated.scalar_summary('learning_rate', lr)

        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = self.__add_loss_summaries(total_loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.contrib.deprecated.histogram_summary(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.contrib.deprecated.histogram_summary(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op


def main(argv=None):  # pylint: disable=unused-argument
    train_ctrs, val_ctrs = sunnybrook.get_all_contours()
    model = LVSegmentation(train_ctrs, val_ctrs)

    if len(sys.argv) != 2:
        print('The program must be run as : python3.5 step2_train_segmenter.py [train|eval]')
        sys.exit(2)
    else:
        if sys.argv[1] == 'train':
            print('Run Train .....')

            if tf.gfile.Exists(FLAGS.train_dir):
                tf.gfile.DeleteRecursively(FLAGS.train_dir)

            tf.gfile.MakeDirs(FLAGS.train_dir)

            model.train()

        elif sys.argv[1] == 'eval':
            print('Run Eval .....')

            if tf.gfile.Exists(FLAGS.eval_dir):
                tf.gfile.DeleteRecursively(FLAGS.eval_dir)

            tf.gfile.MakeDirs(FLAGS.eval_dir)

            model.evaluate()

        else:
            print('The available options for this script are : train and eval')
            sys.exit(2)


if __name__ == "__main__":
    FLAGS = tf.app.flags.FLAGS

    # Basic model parameters.
    tf.app.flags.DEFINE_integer('batch_size', 50, """Number of images to process in a batch.""")
    tf.app.flags.DEFINE_integer('nr_epochs', 200, """Number of epochs to run.""")

    tf.app.flags.DEFINE_string('data_dir', 'data', """Path to the sunnybrook data directory.""")
    tf.app.flags.DEFINE_string('train_dir', 'result/segmenter/train_result',
                               """Directory where to write event logs and checkpoint.""")
    tf.app.flags.DEFINE_string('eval_dir', 'result/segmenter/eval_result', """Directory where to write event logs.""")
    tf.app.flags.DEFINE_string('checkpoint_dir', 'result/segmenter/train_result',
                               """Directory where to read model checkpoints.""")

    tf.app.run()
