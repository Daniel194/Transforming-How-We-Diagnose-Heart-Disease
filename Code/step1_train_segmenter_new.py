from datetime import datetime
import numpy as np
import tensorflow as tf
import functools
import time
import math
import os
import sys
import re


class ImageRecognition(object):
    def __init__(self):
        # Process images of this size. Note that this differs from the original CIFAR
        # image size of 32 x 32. If one alters this number, then the entire model
        # architecture will change and any model would need to be retrained.
        self.IMAGE_SIZE = 24

        # Global constants describing the CIFAR-10 data set.
        self.NUM_CLASSES = 10
        self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
        self.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

        # Constants describing the training process.
        self.MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
        self.NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
        self.LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
        self.INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.
        self.EPSILON = 1e-3  # Hyperparamter for Batch Normalization.

        # If a model is trained with multiple GPUs, prefix all Op names with tower_name
        # to differentiate the operations. Note that this prefix is removed from the
        # names of the summaries when visualizing a model.
        self.TOWER_NAME = 'tower'

    def train(self):
        """
        Train CIFAR-10 for a number of steps.
        :return: Nothing.
        """

        with tf.Graph().as_default():
            global_step = tf.contrib.framework.get_or_create_global_step()

            # Get images and labels for CIFAR-10.
            images, labels = self._distorted_inputs()

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
                        num_examples_per_step = FLAGS.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f  sec/batch)'
                        print(format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=FLAGS.train_dir,
                    hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps), tf.train.NanTensorHook(loss),
                           _LoggerHook()],
                    config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
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

    def __read_cifar10(self, filename_queue):
        """
        Reads and parses examples from CIFAR10 data files.
        :param filename_queue: filename_queue: A queue of strings with the filenames to read from.
        :return: An object representing a single example, with the following fields:
                 height: number of rows in the result (32)
                 width: number of columns in the result (32)
                 depth: number of color channels in the result (3)
                 key: a scalar string Tensor describing the filename & record number for this example.
                 label: an int32 Tensor with the label in the range 0..9.
                 uint8image: a [height, width, depth] uint8 Tensor with the image data
        """

        class CIFAR10Record(object):
            pass

        result = CIFAR10Record()

        # Dimensions of the images in the CIFAR-10 dataset.
        label_bytes = 1
        result.height = 32
        result.width = 32
        result.depth = 3
        image_bytes = result.height * result.width * result.depth

        # Every record consists of a label followed by the image, with a fixed number of bytes for each.
        record_bytes = label_bytes + image_bytes

        # Read a record, getting filenames from the filename_queue.  No header or footer in the CIFAR-10 format,
        # so we leave header_bytes and footer_bytes at their default of 0.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        result.key, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        # The first bytes represent the label, which we convert from uint8->int32.
        result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                                 [result.depth, result.height, result.width])

        # Convert from [depth, height, width] to [height, width, depth].
        result.uint8image = tf.transpose(depth_major, [1, 2, 0])

        return result

    def __generate_image_and_label_batch(self, image, label, min_queue_examples, batch_size, shuffle):
        """
        Construct a queued batch of images and labels.
        :param image: 3-D Tensor of [height, width, 3] of type.float32.
        :param label: 1-D Tensor of type.int32
        :param min_queue_examples:  int32, minimum number of samples to retain
                                    in the queue that provides of batches of examples.
        :param batch_size: Number of images per batch.
        :param shuffle: boolean indicating whether to use a shuffling queue.
        :return: images: Images. 4D tensor of [batch_size, height, width, 3] size.
                 labels: Labels. 1D tensor of [batch_size] size.
        """

        # Create a queue that shuffles the examples, and then read 'batch_size' images + labels from the example queue.
        num_preprocess_threads = 16

        if shuffle:
            images, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                         num_threads=num_preprocess_threads,
                                                         capacity=min_queue_examples + 3 * batch_size,
                                                         min_after_dequeue=min_queue_examples)
        else:
            images, label_batch = tf.train.batch([image, label], batch_size=batch_size,
                                                 num_threads=num_preprocess_threads,
                                                 capacity=min_queue_examples + 3 * batch_size)

        # Display the training images in the visualizer.
        tf.contrib.deprecated.image_summary('images', images)

        return images, tf.reshape(label_batch, [batch_size])

    def __distorted_inputs(self, data_dir, batch_size):
        """
        Construct distorted input for CIFAR training using the Reader ops.
        :param data_dir: Path to the CIFAR-10 data directory.
        :param batch_size: Number of images per batch.
        :return: images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
                 labels: Labels. 1D tensor of [batch_size] size.
        """

        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]

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

        # Image processing for training the network. Note the many random distortions applied to the image.

        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing the order their operation.
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

        print('Filling queue with %d CIFAR images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)

        # Generate a batch of images and labels by building up a queue of examples.
        return self.__generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size,
                                                     shuffle=True)

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
            dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)

        return var

    def __variable_with_weight_decay(self, name, shape, stddev, wd):
        """
        Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        :param name: name of the variable
        :param shape: list of ints
        :param stddev: standard deviation of a truncated Gaussian
        :param wd: add L2Loss weight decay multiplied by this float. If None, weight
                   decay is not added for this Variable.
        :return: Variable Tensor
        """

        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = self.__variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        return var

    def _distorted_inputs(self):
        """
        Construct distorted input for CIFAR training using the Reader ops.
        :return: images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
                 labels: Labels. 1D tensor of [batch_size] size.
        """

        if not FLAGS.data_dir:
            raise ValueError('Please supply a data_dir')

        data_dir = os.path.join(FLAGS.data_dir)
        images, labels = self.__distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)

        if FLAGS.use_fp16:
            images = tf.cast(images, tf.float16)
            labels = tf.cast(labels, tf.float16)

        return images, labels

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

    def __convolution_module(self, name, net, kernel_size, output_shape, batch_norm=True, down_pool=False,
                             up_pool=False, convolution=True):

        """
        Convolution module
        :param name: the name of the layer
        :param net: the data
        :param kernel_size: the kernel size
        :param output_shape: output shape for deconvolution layer
        :param batch_norm: batch normalization (bool value)
        :param down_pool: down pool (bool value)
        :param up_pool: up pool (bool value)
        :param convolution: convolutional layer (bool value)
        :return: return the next hidden layer
        """

        with tf.variable_scope(name) as scope:
            nr_units = functools.reduce(lambda x, y: x * y, kernel_size)
            weights = self.__variable_with_weight_decay('weights', shape=kernel_size,
                                                        stddev=1.0 / math.sqrt(float(nr_units)), wd=0.0)

            scale = self.__variable_on_cpu('scale', kernel_size[3], tf.constant_initializer(1.0))
            beta = self.__variable_on_cpu('beta', kernel_size[3], tf.constant_initializer(0.0))

            if up_pool:
                net = tf.nn.conv2d_transpose(net, weights, output_shape, padding='SAME', strides=[1, 2, 2, 1])
                batch_mean, batch_var = tf.nn.moments(net, [0])
                bn = tf.nn.batch_normalization(net, batch_mean, batch_var, beta, scale, self.EPSILON)
                net = tf.nn.relu(bn, name=scope.name)
                self.__activation_summary(net)

            if convolution:
                net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME')
                self.__activation_summary(net)

            if batch_norm:
                batch_mean, batch_var = tf.nn.moments(net, [0])
                bn = tf.nn.batch_normalization(net, batch_mean, batch_var, beta, scale, self.EPSILON)
                net = bn

            net = tf.nn.relu(net, name=scope.name)

            if down_pool:
                net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope.name)
                self.__activation_summary(net)

        return net

    def __inference(self, images):
        """
        Build the CIFAR-10 model.
        :param images: Images returned from distorted_inputs() or inputs().
        :return: Logits.
        """

        # First Convolutional Layer
        with tf.variable_scope('conv1') as scope:
            nr_units = functools.reduce(lambda x, y: x * y, [5, 5, 3, 64])
            weights = self.__variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                                        stddev=1.0 / math.sqrt(float(nr_units)), wd=0.0)
            scale = self.__variable_on_cpu('scale', [64], tf.constant_initializer(1.0))
            beta = self.__variable_on_cpu('beta', [64], tf.constant_initializer(0.0))

            z = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
            batch_mean, batch_var = tf.nn.moments(z, [0])
            bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.EPSILON)
            conv1 = tf.nn.relu(bn, name=scope.name)
            self.__activation_summary(conv1)

        # Second Convolutional Layer
        with tf.variable_scope('conv2') as scope:
            nr_units = functools.reduce(lambda x, y: x * y, [5, 5, 64, 128])
            weights = self.__variable_with_weight_decay('weights', shape=[5, 5, 64, 128],
                                                        stddev=1.0 / math.sqrt(float(nr_units)), wd=0.0)
            scale = self.__variable_on_cpu('scale', [128], tf.constant_initializer(1.0))
            beta = self.__variable_on_cpu('beta', [128], tf.constant_initializer(0.0))

            z = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME')
            batch_mean, batch_var = tf.nn.moments(z, [0])
            bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.EPSILON)
            conv2 = tf.nn.relu(bn, name=scope.name)
            self.__activation_summary(conv2)

        # First Pool Layer
        with tf.name_scope('pool1'):
            pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # Third Convolutional Layer
        with tf.variable_scope('conv3') as scope:
            nr_units = functools.reduce(lambda x, y: x * y, [5, 5, 128, 256])
            weights = self.__variable_with_weight_decay('weights', shape=[5, 5, 128, 256],
                                                        stddev=1.0 / math.sqrt(float(nr_units)), wd=0.0)
            scale = self.__variable_on_cpu('scale', [256], tf.constant_initializer(1.0))
            beta = self.__variable_on_cpu('beta', [256], tf.constant_initializer(0.0))

            z = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
            batch_mean, batch_var = tf.nn.moments(z, [0])
            bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.EPSILON)
            conv3 = tf.nn.relu(bn, name=scope.name)
            self.__activation_summary(conv3)

        # Fourth Convolutional Layer
        with tf.variable_scope('conv4') as scope:
            nr_units = functools.reduce(lambda x, y: x * y, [3, 3, 256, 256])
            weights = self.__variable_with_weight_decay('weights', shape=[3, 3, 256, 256],
                                                        stddev=1.0 / math.sqrt(float(nr_units)), wd=0.0)
            scale = self.__variable_on_cpu('scale', [256], tf.constant_initializer(1.0))
            beta = self.__variable_on_cpu('beta', [256], tf.constant_initializer(0.0))

            z = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME')
            batch_mean, batch_var = tf.nn.moments(z, [0])
            bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.EPSILON)
            conv4 = tf.nn.relu(bn, name=scope.name)
            self.__activation_summary(conv4)

        # Second Pool Layer
        with tf.name_scope('pool2'):
            pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First Dropout
        with tf.name_scope('dropout1'):
            dropout1 = tf.nn.dropout(pool2, 0.8)

        # Fifth Convolutional Layer
        with tf.variable_scope('conv5') as scope:
            nr_units = functools.reduce(lambda x, y: x * y, [3, 3, 256, 512])
            weights = self.__variable_with_weight_decay('weights', shape=[3, 3, 256, 512],
                                                        stddev=1.0 / math.sqrt(float(nr_units)), wd=0.0)
            scale = self.__variable_on_cpu('scale', [512], tf.constant_initializer(1.0))
            beta = self.__variable_on_cpu('beta', [512], tf.constant_initializer(0.0))

            z = tf.nn.conv2d(dropout1, weights, strides=[1, 1, 1, 1], padding='SAME')
            batch_mean, batch_var = tf.nn.moments(z, [0])
            bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.EPSILON)
            conv5 = tf.nn.relu(bn, name=scope.name)
            self.__activation_summary(conv5)

        # Second Dropout
        with tf.name_scope('dropout2'):
            dropout2 = tf.nn.dropout(conv5, 0.8)

        # Sixth Convolutional Layer
        with tf.variable_scope('conv6') as scope:
            nr_units = functools.reduce(lambda x, y: x * y, [3, 3, 512, 512])
            weights = self.__variable_with_weight_decay('weights', shape=[3, 3, 512, 512],
                                                        stddev=1.0 / math.sqrt(float(nr_units)), wd=0.0)
            scale = self.__variable_on_cpu('scale', [512], tf.constant_initializer(1.0))
            beta = self.__variable_on_cpu('beta', [512], tf.constant_initializer(0.0))

            z = tf.nn.conv2d(dropout2, weights, strides=[1, 1, 1, 1], padding='SAME')
            batch_mean, batch_var = tf.nn.moments(z, [0])
            bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.EPSILON)
            conv6 = tf.nn.relu(bn, name=scope.name)
            self.__activation_summary(conv6)

        # Third Pool Layer
        with tf.name_scope('pool3'):
            pool3 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Third Dropout
        with tf.name_scope('dropout3'):
            dropout3 = tf.nn.dropout(pool3, 0.5)

        # First Fully Connected Layer
        with tf.variable_scope('fc1') as scope:
            nr_units = functools.reduce(lambda x, y: x * y, [4608, 2048])
            dropout3_flat = tf.reshape(dropout3, [-1, 4608])

            weights = self.__variable_with_weight_decay('weights', shape=[4608, 2048],
                                                        stddev=1.0 / math.sqrt(float(nr_units)), wd=0.0)
            scale = self.__variable_on_cpu('scale', [2048], tf.constant_initializer(1.0))
            beta = self.__variable_on_cpu('beta', [2048], tf.constant_initializer(0.0))

            z = tf.matmul(dropout3_flat, weights)
            batch_mean, batch_var = tf.nn.moments(z, [0])
            bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.EPSILON)
            fc1 = tf.nn.relu(bn, name=scope.name)
            self.__activation_summary(fc1)

        # SoftMax Linear
        with tf.variable_scope('softmax_linear') as scope:
            nr_units = functools.reduce(lambda x, y: x * y, [2048, self.NUM_CLASSES])

            weights = self.__variable_with_weight_decay('weights', shape=[2048, self.NUM_CLASSES],
                                                        stddev=1.0 / math.sqrt(float(nr_units)), wd=0.0)
            biases = self.__variable_on_cpu('biases', [self.NUM_CLASSES], tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(fc1, weights), biases, name=scope.name)
            self.__activation_summary(softmax_linear)

        return softmax_linear

    def __loss(self, logits, labels):
        """
        Add L2Loss to all the trainable variables.
        Add summary for "Loss" and "Loss/avg".
        :param logits: Logits from inference().
        :param labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]
        :return: Loss tensor of type float.
        """

        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                       name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
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
        Train CIFAR-10 model.
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
    model = ImageRecognition()

    if len(sys.argv) != 2:
        print('The program must be run as : python3.5 CNN5.py [train|eval]')
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
    tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
    tf.app.flags.DEFINE_string('data_dir', 'data', """Path to the CIFAR-10 data directory.""")
    tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
    tf.app.flags.DEFINE_string('train_dir', 'result/CNN5/train_result',
                               """Directory where to write event logs and checkpoint.""")
    tf.app.flags.DEFINE_integer('max_steps', 100000, """Number of batches to run.""")
    tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
    tf.app.flags.DEFINE_string('eval_dir', 'result/CNN5/eval_result', """Directory where to write event logs.""")
    tf.app.flags.DEFINE_string('eval_data', 'test', """Either 'test' or 'train_eval'.""")
    tf.app.flags.DEFINE_string('checkpoint_dir', 'result/CNN5/train_result',
                               """Directory where to read model checkpoints.""")
    tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, """How often to run the eval.""")
    tf.app.flags.DEFINE_integer('num_examples', 10000, """Number of examples to run.""")
    tf.app.flags.DEFINE_boolean('run_once', False, """Whether to run eval only once.""")

    tf.app.run()
