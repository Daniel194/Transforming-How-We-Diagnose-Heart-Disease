import step1_preprocess as preprocess
import step3_predict_volumes as predict
import step4_calibrate as calibrate
import utils.settings as settings

import pandas

########## STEP 1 - Preprocess ##########

preprocess.convert_sax_images(rescale=True, base_size=256, crop_size=256)
preprocess.create_csv_data()
preprocess.enrich_dicom_csvdata()
preprocess.enrich_traindata()

print("Done - Step 1")

########## STEP 2 & 3 - Segmentation & predict volum ##########

slice_data = pandas.read_csv(settings.RESULT_DIR + "dicom_data_enriched.csv", sep=";")
predict.predict_patient(148, slice_data, '')

print("Done - Step 2 & 3")

########## STEP 4 - Calibrate ##########

calibrate.calibrate_volume()

print("Done - Step 4")
