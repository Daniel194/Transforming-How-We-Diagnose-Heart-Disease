import os
import random

import Code.utils as util
import cv2
import numpy as np

np.random.seed(1301)
random.seed(1301)


class FileIter():
    def __init__(self, root_dir, flist_name,
                 regress_overlay=True,
                 cut_off_size=None,
                 data_name="data",
                 label_name="softmax_label",
                 batch_size=1,
                 augment=False,
                 mean_image=None,
                 crop_size=0,
                 random_crop=False,
                 shuffle=False,
                 scale_size=None,
                 crop_indent_x=None,
                 crop_indent_y=None):

        self.regress_overlay = regress_overlay
        self.file_lines = []
        self.epoch = 0
        self.scale_size = scale_size
        self.shuffle = shuffle
        self.label_files = []
        self.image_files = []
        self.batch_size = batch_size
        self.Augment = augment
        self.random = random.Random()
        self.random.seed(1301)
        self.root_dir = root_dir
        self.flist_name = os.path.join(self.root_dir, flist_name)
        self.mean = cv2.imread(mean_image, cv2.IMREAD_GRAYSCALE)
        self.cut_off_size = cut_off_size
        self.data_name = data_name
        self.label_name = label_name
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.crop_indent_x = crop_indent_x
        self.crop_indent_y = crop_indent_y

        self.num_data = len(open(self.flist_name, 'r').readlines())
        self.cursor = -1
        self.read_lines()
        self.data, self.label = self._read()
        self.reset()

    def _read(self):
        """
        Get two list, each list contains two elements: name and nd.array value
        :return: return a list with two elements (labesl and data)
        """

        data = {}
        label = {}

        dd = []
        ll = []

        for i in range(0, self.batch_size):
            line = self.get_line()
            data_img_name, label_img_name = line.strip('\n').split("\t")
            d, l = self._read_img(data_img_name, label_img_name)
            dd.append(d)
            ll.append(l)

        d = np.vstack(dd)
        l = np.vstack(ll)
        data[self.data_name] = d

        if not self.regress_overlay:
            l = l.reshape(l.shape[0])

        label[self.label_name] = l

        return list(data.items()), list(label.items())

    def _read_img(self, img_name, label_name):
        """
        Read image and label
        :param img_name: the name of the image
        :param label_name: the name of the label
        :return: return the image and the label
        """

        img_path = os.path.join(self.root_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)

        if self.regress_overlay:
            label_path = os.path.join(self.root_dir, label_name)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(float)
        else:
            label_path = label_name
            label = float(label_name)

        if self.scale_size is not None:
            img = cv2.resize(img, (self.scale_size, self.scale_size), interpolation=cv2.INTER_AREA).astype(float)

            if self.regress_overlay:
                label = cv2.resize(label, (self.scale_size, self.scale_size), interpolation=cv2.INTER_AREA).astype(
                    float)

        self.image_files.append(img_path)
        self.label_files.append(label_path)

        if not self.regress_overlay:
            img = np.array(img, dtype=np.float32)  # (h, w, c)
            img = img - self.mean

        if self.Augment:
            rnd_val = self.random.randint(0, 100)
            if rnd_val > 10:

                img = util.elastic_transform(img, 150, 15)

                if self.regress_overlay:
                    label = util.elastic_transform(label, 150, 15)

        img = img.reshape(img.shape[0], img.shape[1], 1)

        img /= 256.

        if self.regress_overlay:
            label /= 256.
        else:
            label /= 30.

        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)  # (c, h, w)
        label = np.array(label)  # (h, w)

        if self.crop_size != 0:
            crop_max = img.shape[1] - self.crop_size
            crop_x = crop_max / 2
            crop_y = crop_max / 2

            if self.crop_indent_x is not None:
                crop_x = self.crop_indent_x
            if self.crop_indent_y is not None:
                crop_y = self.crop_indent_y

            if self.random_crop:
                crop_x = self.random.randint(0, crop_max)
                crop_y = self.random.randint(0, crop_max)

            img = img[:, crop_y:crop_y + self.crop_size, crop_x: crop_x + self.crop_size]

            if self.regress_overlay:
                label = label[crop_y:crop_y + self.crop_size, crop_x: crop_x + self.crop_size]

        img = np.expand_dims(img, axis=0)  # (1, c, h, w) or (1, h, w)

        if self.regress_overlay:
            label = label.reshape(1, label.shape[0] * label.shape[1])

        return img, label

    @property
    def provide_data(self):
        """
        The name and shape of data provided by this iterator
        :return: the data
        """

        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.data]

        return res

    @property
    def provide_label(self):
        """
        The name and shape of label provided by this iterator
        :return: the label
        """

        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.label]
        print("label : " + str(res))

        return res

    def reset(self):
        """
        Reset old state
        :return: nothing
        """

        self.cursor = -1
        self.read_lines()
        util.ELASTIC_INDICES = None
        self.label_files = []
        self.image_files = []
        self.epoch += 1

    def read_lines(self):
        """
        Read lines
        :return: nothing
        """

        self.current_line_no = -1

        with open(self.flist_name, 'r') as f:
            self.file_lines = f.readlines()
            if self.shuffle:
                self.random.shuffle(self.file_lines)

    def get_line(self):
        """
        Get nex line
        :return: return next line
        """

        self.current_line_no += 1

        return self.file_lines[self.current_line_no]

    def iter_next(self):
        """
        Test if there is a nex iteration
        :return: a bool value
        """

        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def eof(self):
        """
        The if it is end of file
        :return: a boolean value
        """

        res = self.cursor >= self.num_data

        return res

    def next(self):
        """
        :return: one dict which contains "data" and "label"
        """

        if self.iter_next():
            self.data, self.label = self._read()

            return self.data[0][1], self.label[0][1]
        else:
            raise StopIteration
