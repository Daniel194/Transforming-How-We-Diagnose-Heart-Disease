import dicom
import cv2
import re
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy
import scipy.misc
import sys
import random

warnings.filterwarnings('ignore')  # we ignore a RuntimeWarning produced from dividing by zero
np.random.seed(1301)
random.seed(1301)

SUNNYBROOK_ROOT_PATH = "data/"
TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH, "Sunnybrook Cardiac MR Database ContoursPart3",
                                  "TrainingDataContours")
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH, "challenge_training")
SPLIT_RATIO = 0.1
SAX_SERIES = {
    # challenge training
    "SC-HF-I-1": "0004",
    "SC-HF-I-2": "0106",
    "SC-HF-I-4": "0116",
    "SC-HF-I-40": "0134",
    "SC-HF-NI-3": "0379",
    "SC-HF-NI-4": "0501",
    "SC-HF-NI-34": "0446",
    "SC-HF-NI-36": "0474",
    "SC-HYP-1": "0550",
    "SC-HYP-3": "0650",
    "SC-HYP-38": "0734",
    "SC-HYP-40": "0755",
    "SC-N-2": "0898",
    "SC-N-3": "0915",
    "SC-N-40": "0944",
}


class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r"/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-icontour-manual.txt", ctr_path)
        self.case = self.__shrink_case(match.group(1))
        self.img_no = int(match.group(2))

    def __str__(self):
        return "<Contour for case %s, image %d>" % (self.case, self.img_no)

    __repr__ = __str__

    def __shrink_case(self, case):
        toks = case.split("-")

        def shrink_if_number(x):
            try:
                cvt = int(x)
                return str(cvt)
            except ValueError:
                return x

        return "-".join([shrink_if_number(t) for t in toks])


def load_contour(contour, img_path):
    """
    Function maps each contour file to a DICOM image, according to a case study number,
    using the SAX_SERIES dictionary and the extracted image number.
    Upon successful mapping, the function returns NumPy arrays for a pair of DICOM image and contour file, or label.
    :param contour: and array which contains all contour path
    :param img_path: path to image folder
    :return: return images and label
    """

    filename = "IM-%s-%04d.png" % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(img_path, contour.case, filename)
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE).astype(float)
    ctrs = np.loadtxt(contour.ctr_path, delimiter=" ").astype(np.int)
    label = np.zeros_like(img, dtype="uint8")
    cv2.fillPoly(label, [ctrs], 1)

    return img, label


def __get_all_contours(contour_path):
    """
    Function walks through a directory containing contour files,
    and extracts the necessary case study number and image number from a contour filename using the Contour class.
    :param contour_path: path to contour file
    :return: extracted path
    """

    contours = [os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(contour_path)
                for f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt')]

    print("Shuffle data")
    np.random.shuffle(contours)
    print("Number of examples: {:d}".format(len(contours)))
    extracted = list(map(Contour, contours))

    return extracted


def export_all_contours(batch, img_path):
    """
    Function return an array with all images and labels for a specific batch.
    :param batch: an array with all path to contour file.
    :param img_path: image path to all images
    :return: return two arrays, one for images and one for labels
    """

    imgs, labels = [], []

    if len(batch) == 0:
        return

    for idx, ctr in enumerate(batch):
        try:
            img, label = load_contour(ctr, img_path)
            imgs.append(img)
            labels.append(label)

            if idx % 50 == 0:
                print(ctr)
                # plt.imshow(img, cmap='gray')
                # plt.show()
                # plt.imshow(label)
                # plt.show()

        except IOError:
            continue

    return imgs, labels


def convert_dicom_to_png(ctrs, img_path):
    """
    Convert DICOM format to PNG format
    :param ctrs: dicom paths
    :param img_path: image folder path
    :return: nothing
    """

    for idx, ctr in enumerate(ctrs):
        filename = "IM-%s-%04d.dcm" % (SAX_SERIES[ctr.case], ctr.img_no)
        full_path = os.path.join(img_path, ctr.case, filename)
        dicom_data = dicom.read_file(full_path)

        img_new_path = full_path.replace(".dcm", ".png")
        scipy.misc.imsave(img_new_path, dicom_data.pixel_array)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)
        clahe = cv2.createCLAHE(tileGridSize=(1, 1))
        cl_img = clahe.apply(img)
        cv2.imwrite(img_new_path, cl_img)


def distorted_image(image, label):
    """
    Generate new image an label
    :param image: the image
    :param label: the label
    :return: return new image and label
    """

    crop_x = random.randint(0, 16)
    crop_y = random.randint(0, 16)

    image = image[crop_y:crop_y + 224, crop_x: crop_x + 224]  # [224 x 224]
    label = label[crop_y:crop_y + 224, crop_x: crop_x + 224]  # [224 x 224]

    return image, label


def get_all_contours():
    """
    :return: return paths to all countours
    """

    ctrs = __get_all_contours(TRAIN_CONTOUR_PATH)
    val_ctrs = ctrs[0:int(SPLIT_RATIO * len(ctrs))]
    train_ctrs = ctrs[int(SPLIT_RATIO * len(ctrs)):]

    return train_ctrs, val_ctrs


if __name__ == "__main__":

    BATCHSIZE = 100
    NR_EPOCHS = 1

    if len(sys.argv) > 1 and str(sys.argv[1]) == 'convert':
        ctrs = __get_all_contours(TRAIN_CONTOUR_PATH)
        convert_dicom_to_png(ctrs, TRAIN_IMG_PATH)

    train_ctrs, val_ctrs = get_all_contours()

    for nr_epoch in range(NR_EPOCHS):
        print("\n Epoch number {:d} \n".format(nr_epoch))

        np.random.shuffle(train_ctrs)

        for i in range(int(np.ceil(len(train_ctrs) / float(BATCHSIZE)))):
            batch = train_ctrs[(BATCHSIZE * i):(BATCHSIZE * (i + 1))]

            imgs_train, labels_train = export_all_contours(batch, TRAIN_IMG_PATH)

    print("Processing {:d} images and labels...".format(len(val_ctrs)))

    imgs_val, labels_val = export_all_contours(val_ctrs, TRAIN_IMG_PATH)
