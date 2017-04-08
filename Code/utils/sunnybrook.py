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
import random

warnings.filterwarnings('ignore')  # we ignore a RuntimeWarning produced from dividing by zero
random.seed(1301)

SUNNYBROOK_ROOT_PATH = "../../data/"
CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH, "Sunnybrook Contours")
IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH, "Sunnybrook IMG")

SAX_SERIES = {
    'SC-HF-I-1': '0004',
    'SC-HF-I-2': '0106',
    'SC-HF-I-4': '0116',
    'SC-HF-I-5': '0156',
    'SC-HF-I-6': '0180',
    'SC-HF-I-7': '0209',
    'SC-HF-I-8': '0226',
    'SC-HF-I-9': '0241',
    'SC-HF-I-10': '0024',
    'SC-HF-I-11': '0043',
    'SC-HF-I-12': '0062',
    'SC-HF-I-40': '0134',
    'SC-HF-NI-3': '0379',
    'SC-HF-NI-4': '0501',
    'SC-HF-NI-7': '0523',
    'SC-HF-NI-12': '0286',
    'SC-HF-NI-11': '0270',
    'SC-HF-NI-13': '0304',
    'SC-HF-NI-14': '0331',
    'SC-HF-NI-15': '0359',
    'SC-HF-NI-31': '0401',
    'SC-HF-NI-33': '0424',
    'SC-HF-NI-34': '0446',
    'SC-HF-NI-36': '0474',
    'SC-HYP-1': '0550',
    'SC-HYP-3': '0650',
    'SC-HYP-6': '0767',
    'SC-HYP-7': '0007',
    'SC-HYP-8': '0796',
    'SC-HYP-9': '0003',
    'SC-HYP-10': '0579',
    'SC-HYP-11': '0601',
    'SC-HYP-12': '0629',
    'SC-HYP-37': '0702',
    'SC-HYP-38': '0734',
    'SC-HYP-40': '0755',
    'SC-N-2': '0898',
    'SC-N-3': '0915',
    'SC-N-5': '0963',
    'SC-N-6': '0981',
    'SC-N-7': '1009',
    'SC-N-9': '1031',
    'SC-N-10': '0851',
    'SC-N-11': '0878',
    'SC-N-40': '0944',
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

    print("Number of examples: {:d}".format(len(contours)))
    extracted = list(map(Contour, contours))

    return extracted


def __export_all_contours(batch, img_path):
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
                plt.imshow(img, cmap='gray')
                plt.show()
                plt.imshow(label)
                plt.show()

        except IOError:
            continue

    return imgs, labels


def export_all_contours(batch):
    """
    Function return an array with all images and labels for a specific batch.
    :param batch: an array with all path to contour file.
    :return: return two arrays, one for images and one for labels
    """

    imgs, labels = [], []

    if len(batch) == 0:
        return

    for idx, ctr in enumerate(batch):
        try:
            img, label = load_contour(ctr, IMG_PATH)
            imgs.append(img)
            labels.append(label)

        except IOError:
            continue

    return np.array(imgs), np.array(labels)


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

        img = cv2.imread(img_new_path, cv2.IMREAD_GRAYSCALE)
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

    ctrs = __get_all_contours(CONTOUR_PATH)

    val_ctrs = ctrs[0:5]
    train_ctrs = ctrs[5:]

    return train_ctrs, val_ctrs


if __name__ == "__main__":
    # Convert Sunnybrook dataset from DICOM form to PNG format

    ctrs = __get_all_contours(CONTOUR_PATH)

    convert_dicom_to_png(ctrs, IMG_PATH)
