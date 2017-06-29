import step1_preprocess as preprocess
import step3_predict_volumes as predict
import step4_calibrate as calibrate
import step5_diagnostic as diagnostic

import utils.settings as settings

import pandas

print("Start - Step 1 - Preprocess")

preprocess.convert_sax_images(rescale=True, base_size=256, crop_size=256)
preprocess.create_csv_data()
preprocess.enrich_dicom_csvdata()
preprocess.enrich_traindata()

print("Done - Step 1 - Preprocess")

slice_data = pandas.read_csv(settings.RESULT_DIR + "dicom_data_enriched.csv", sep=";")
predict.predict_patient(148, slice_data, '')

print("Start - Step 4 - Calibrate")
print("   > Calibrate prediction")

calibrate.calibrate_volume()

print("Done - Step 4 - Calibrate")

print("Start - Step 4 - Ejection fraction")
print("   > Calculate ejection fraction")

print('---------------------------------------------------------------------------------------------------')

diagnostic.calculate_ej_predicted()

print('---------------------------------------------------------------------------------------------------')

diagnostic.calculate_ej_real()

print('---------------------------------------------------------------------------------------------------')

print("Done - Step 4 - Ejection fraction")
