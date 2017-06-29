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

######### STEP 5 - Calculate ejection fraction ##########

PREDICT_FILE_PATH = "result/prediction_calibrated_vgg.csv"
data = pandas.read_csv(PREDICT_FILE_PATH, sep=";")

vd = data['cal_pred_dia'][0]
vs = data['cal_pred_sys'][0]

ej = 100 * ((vd - vs) / vd)

print('Diastola prezisa: ', vd)
print('Systola prezisa: ', vs)
print('Fractia de ejectie prezisa: ', ej)

print('---------------------------------------------------------------------------------------------------')

vd = data['Diastole'][0]
vs = data['Systole'][0]

ej = 100 * ((vd - vs) / vd)

print('Diastola reala: ', vd)
print('Systola reala: ', vs)
print('Fractia de ejectie reala: ', ej)
