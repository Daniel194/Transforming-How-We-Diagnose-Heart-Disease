import csv
import pandas
import scipy
import scipy.misc
import cv2
import numpy as np
import os
import utils.sunnybrook as sunnybrook
import utils.utils as utils
import utils.settings as settings


def create_csv_data():
    """
    Create CSV file
    :return: nothing
    """

    print("   > Salvarea datelor DICOM intr-un fisier CSV")

    row_no = 0

    with open(settings.BASE_DIR + settings.RESULT_DIR + "dicom_data.csv", "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(
            ["patient_id", "slice_no", "frame_no", "rows", "columns", "spacing", "slice_thickness", "slice_location",
             "slice_location2", "plane", "image_position", "sv", "time", "manufact", "modelname", "age", "birth", "sex",
             "file_name", "angle", "o1", "o2", "o3", "o4", "o5", "o6"])

        for dicom_data in utils.enumerate_sax_files():
            row_no += 1

            csv_writer.writerow([
                str(dicom_data.patient_id),
                str(dicom_data.series_number),
                str(dicom_data.instance_number),
                str(dicom_data.rows),
                str(dicom_data.columns),
                str(dicom_data.spacing[0]),
                str(dicom_data.slice_thickness),
                str(dicom_data.slice_location),
                str(dicom_data.get_location()),
                dicom_data.in_plane_encoding_direction,
                str(dicom_data.image_position),
                str(dicom_data.get_value("SequenceVariant")),
                str(dicom_data.get_value("InstanceCreationTime")),
                str(dicom_data.get_value("Manufacturer")),
                str(dicom_data.get_value("ManufacturerModelName")),
                str(dicom_data.get_value("PatientAge")),
                str(dicom_data.get_value("PatientBirthDate")),
                str(dicom_data.get_value("PatientSex")),
                dicom_data.file_name.replace(".dcm", ""),
                str(dicom_data.get_value("FlipAngle")),
                str(round(dicom_data.image_orientation_patient[0], 2)),
                str(round(dicom_data.image_orientation_patient[1], 2)),
                str(round(dicom_data.image_orientation_patient[2], 2)),
                str(round(dicom_data.image_orientation_patient[3], 2)),
                str(round(dicom_data.image_orientation_patient[4], 2)),
                str(round(dicom_data.image_orientation_patient[5], 2))
            ])


def up_down(current_value, previous_value):
    delta = current_value - previous_value
    delta = delta.fillna(0)
    updown = pandas.Series(delta.apply(lambda x: 0 if x == 0 else 1 if x > 0 else -1))

    return updown


def slice_delta(current_value, next_value):
    delta = current_value - next_value
    delta = delta.fillna(999)
    return delta


def count_small_deltas(current_value):
    res = len(current_value[abs(current_value) < 2])
    return res


def get_age_years(age_string):
    """
    Get patient age
    :param age_string: string whcich contains age information.
    :return: paitent age
    """

    res = 0

    if "Y" in age_string:
        age_string = age_string.replace("Y", "")
        res = float(age_string)
    if "M" in age_string:
        age_string = age_string.replace("M", "")
        res = round(float(age_string) / 12., 2)
    if "W" in age_string:
        age_string = age_string.replace("W", "")
        res = round(float(age_string) / 52., 2)

    return res


def enrich_dicom_csvdata():
    """
    Write in csv file
    :return: nothing
    """

    print("   > Adaugarea de noi coloane la fisierul CSV")

    dicom_data = pandas.read_csv(settings.BASE_DIR + settings.RESULT_DIR + "dicom_data.csv", sep=";")
    dicom_data["age_years"] = dicom_data["age"].apply(lambda x: get_age_years(x))
    dicom_data["patient_id_frame"] = dicom_data["patient_id"].map(str) + "_" + dicom_data["frame_no"].map(str)
    dicom_data = dicom_data.sort(["patient_id", "frame_no", "slice_location", "file_name"], ascending=[1, 1, 1, 1])

    patient_grouped = dicom_data.groupby("patient_id_frame")
    dicom_data['up_down'] = patient_grouped['time'].apply(lambda x: up_down(x, x.shift(1)))
    dicom_data['up_down_agg'] = patient_grouped["up_down"].transform(lambda x: sum(x))
    dicom_data['slice_location_sort'] = dicom_data['slice_location'] * dicom_data['up_down_agg']
    dicom_data = dicom_data.sort(["patient_id", "frame_no", "slice_location_sort", "slice_location", "file_name"])

    patient_grouped = dicom_data.groupby("patient_id_frame")
    dicom_data['slice_location_delta'] = patient_grouped['slice_location'].apply(lambda x: slice_delta(x, x.shift(-1)))
    dicom_data['small_slice_count'] = patient_grouped['slice_location_delta'].transform(lambda x: count_small_deltas(x))
    dicom_data["slice_count"] = patient_grouped["up_down"].transform("count")
    dicom_data["normal_slice_count"] = dicom_data["slice_count"] - dicom_data['small_slice_count']

    dicom_data = dicom_data[dicom_data["slice_location_delta"] != 0]

    patient_grouped = dicom_data.groupby("patient_id_frame")
    dicom_data['up_down'] = patient_grouped['time'].apply(lambda x: up_down(x, x.shift(1)))
    dicom_data['up_down_agg'] = patient_grouped["up_down"].transform(lambda x: sum(x))

    dicom_data.to_csv(settings.BASE_DIR + settings.RESULT_DIR + "dicom_data_enriched.csv", sep=";")

    dicom_data = dicom_data[dicom_data["frame_no"] == 1]
    dicom_data.to_csv(settings.BASE_DIR + settings.RESULT_DIR + "dicom_data_enriched_frame1.csv", sep=";")


def enrich_traindata():
    """
    Write train data in CSV file
    :return: nothing
    """

    print("   > Adaugarea de noi coloane la datele de antrenare")

    train_data = pandas.read_csv(settings.BASE_DIR + settings.DATA_DIR + "train_validate.csv", sep=",")
    dicom_data = pandas.read_csv(settings.BASE_DIR + settings.RESULT_DIR + "dicom_data_enriched_frame1.csv", sep=";")
    patient_grouped = dicom_data.groupby("patient_id")

    enriched_traindata = patient_grouped.first().reset_index()
    enriched_traindata = enriched_traindata[
        ["patient_id", "rows", "columns", "spacing", "slice_thickness", "plane", "slice_count", "up_down_agg",
         "age_years", "sex", "small_slice_count", "normal_slice_count", "angle"]]
    enriched_traindata = pandas.merge(left=enriched_traindata, right=train_data, how='left', left_on='patient_id',
                                      right_on='Id')

    enriched_traindata["pred_dia"] = 0
    enriched_traindata["error_dia"] = 0
    enriched_traindata["abserr_dia"] = 0
    enriched_traindata["pred_sys"] = 0
    enriched_traindata["error_sys"] = 0
    enriched_traindata["abserr_sys"] = 0

    enriched_traindata.to_csv(settings.BASE_DIR + settings.RESULT_DIR + "train_enriched.csv", sep=";")


def get_patient_id(dir):
    """
    Get patient id
    :param dir: directory name
    :return: patient id
    """

    parts = dir.split('\\')
    res = parts[len(parts) - 3]

    return res


def get_slice_type(dir_name):
    """
    Get slice type
    :param dir_name: directory name
    :return: slice type
    """

    parts = dir_name.split('/')
    res = parts[len(parts) - 1]

    return res


def get_square_crop(img, base_size=256, crop_size=256):
    """
    Crop the img
    :param img: the img
    :param base_size: base siez
    :param crop_size: crop size
    :return: return the new img
    """

    res = img
    height, width = res.shape
    if height < base_size:
        diff = base_size - height
        extend_top = int(diff / 2)
        extend_bottom = diff - extend_top
        res = cv2.copyMakeBorder(res, extend_top, extend_bottom, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)
        height = base_size

    if width < base_size:
        diff = base_size - width
        extend_top = int(diff / 2)
        extend_bottom = diff - extend_top
        res = cv2.copyMakeBorder(res, 0, 0, extend_top, extend_bottom, borderType=cv2.BORDER_CONSTANT, value=0)
        width = base_size

    crop_y_start = int((height - crop_size) / 2)
    crop_x_start = int((width - crop_size) / 2)
    res = res[crop_y_start:(crop_y_start + crop_size), crop_x_start:(crop_x_start + crop_size)]

    return res


def convert_sax_images(rescale=True, base_size=256, crop_size=256):
    """
    Convert dicom format in png images and write them in a specific folder.
    :param rescale: boolean value for rescaling image.
    :param base_size: base size
    :param crop_size: crop size
    :return: nothing
    """

    target_dir = settings.BASE_PREPROCESSEDIMAGES_DIR

    utils.delete_files(target_dir, "*.png")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print('   > Convertirea fisierelor DICOM in fisiere PNG')

    file_count = 0
    for dicom_data in utils.enumerate_sax_files():
        file_count += 1

        if dicom_data.in_plane_encoding_direction not in ["ROW", "COL"]:
            raise Exception("ROW,COL")

        if dicom_data.spacing[0] != dicom_data.spacing[1]:
            raise Exception("Data spacings not equal")

        location_id = int(dicom_data.slice_location) + 10000
        location_id_str = str(location_id).rjust(5, '0')

        img_path = target_dir + str(dicom_data.patient_id).rjust(4, '0') + "_" + dicom_data.series_description.rjust(8,
                                                                                                                     '0') + "_" + str(
            dicom_data.instance_number).rjust(2, '0') + "_" + location_id_str + "_" + dicom_data.file_name.replace(
            ".dcm", ".png")
        scipy.misc.imsave(img_path, dicom_data.pixel_array)

        img = cv2.imread(img_path, 0)
        if dicom_data.in_plane_encoding_direction == "COL":
            img = cv2.transpose(img)
            img = cv2.flip(img, 0)

        if rescale:
            scale = dicom_data.spacing[0]
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        sq_img = get_square_crop(img, base_size=base_size, crop_size=crop_size)
        clahe = cv2.createCLAHE(tileGridSize=(1, 1))
        cl_img = clahe.apply(sq_img)
        cv2.imwrite(img_path, cl_img)


if __name__ == "__main__":
    convert_sax_images(rescale=True, base_size=256, crop_size=256)
    create_csv_data()
    enrich_dicom_csvdata()
    enrich_traindata()

    train, val = sunnybrook.get_all_contours()
    ctrs = np.append(train, val)
    sunnybrook.convert_dicom_to_png(ctrs)
