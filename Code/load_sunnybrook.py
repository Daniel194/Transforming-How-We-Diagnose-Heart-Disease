import dicom, cv2, re
import os, fnmatch, shutil
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')  # we ignore a RuntimeWarning produced from dividing by zero
np.random.seed(1234)

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

SUNNYBROOK_ROOT_PATH = "data/"

TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH, "Sunnybrook Cardiac MR Database ContoursPart3",
                                  "TrainingDataContours")
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH, "challenge_training")


def shrink_case(case):
    toks = case.split("-")

    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x

    return "-".join([shrink_if_number(t) for t in toks])


class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r"/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-icontour-manual.txt", ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))

    def __str__(self):
        return "<Contour for case %s, image %d>" % (self.case, self.img_no)

    __repr__ = __str__


def load_contour(contour, img_path):
    filename = "IM-%s-%04d.dcm" % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(img_path, contour.case, filename)
    f = dicom.read_file(full_path)
    img = f.pixel_array.astype(np.int)
    ctrs = np.loadtxt(contour.ctr_path, delimiter=" ").astype(np.int)
    label = np.zeros_like(img, dtype="uint8")
    cv2.fillPoly(label, [ctrs], 1)

    return img, label


def get_all_contours(contour_path):
    contours = [os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(contour_path)
                for f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt')]

    print("Shuffle data")
    np.random.shuffle(contours)
    print("Number of examples: {:d}".format(len(contours)))
    extracted = map(Contour, contours)

    return extracted


def export_all_contours(contours, img_path, lmdb_img_name, lmdb_label_name):
    for lmdb_name in [lmdb_img_name, lmdb_label_name]:
        db_path = os.path.abspath(lmdb_name)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

    counter_img = 0
    counter_label = 0
    batchsz = 100

    print("Processing {:d} images and labels...".format(len(contours)))

    for i in range(int(np.ceil(len(contours) / float(batchsz)))):
        batch = contours[(batchsz * i):(batchsz * (i + 1))]

        if len(batch) == 0:
            break

        imgs, labels = [], []

        for idx, ctr in enumerate(batch):
            try:
                img, label = load_contour(ctr, img_path)
                imgs.append(img)
                labels.append(label)
                if idx % 20 == 0:
                    print(ctr)
                    plt.imshow(img)
                    plt.show()
                    plt.imshow(label)
                    plt.show()
            except IOError:
                continue

        counter_img += 1
        print("Processed {:d} images".format(counter_img))

        counter_label += 1
        print("Processed {:d} labels".format(counter_label))


if __name__ == "__main__":
    SPLIT_RATIO = 0.1
    print("Mapping ground truth contours to images...")
    ctrs = get_all_contours(TRAIN_CONTOUR_PATH)
    val_ctrs = ctrs[0:int(SPLIT_RATIO * len(ctrs))]
    train_ctrs = ctrs[int(SPLIT_RATIO * len(ctrs)):]
    print("Done mapping ground truth contours to images")
    print("\nBuilding LMDB for train...")
    export_all_contours(train_ctrs, TRAIN_IMG_PATH, "train_images_lmdb", "train_labels_lmdb")
    print("\nBuilding LMDB for val...")
    export_all_contours(val_ctrs, TRAIN_IMG_PATH, "val_images_lmdb", "val_labels_lmdb")
