import step1_preprocess as preprocess
import step3_predict_volumes as predict
import step4_calibrate as calibrate
import step5_diagnostic as diagnostic

import utils.settings as settings

import pandas

print("Inceput - Pas 1 - Preprocesare")

preprocess.convert_sax_images(rescale=True, base_size=256, crop_size=256)
preprocess.create_csv_data()
preprocess.enrich_dicom_csvdata()
preprocess.enrich_traindata()

print("Terminat - Pas 1 - Preprocesare")

slice_data = pandas.read_csv(settings.RESULT_DIR + "dicom_data_enriched.csv", sep=";")
predict.predict_patient(148, slice_data, '')

print("Inceput - Pas 4 - Calibrare")
print("   > Calibrarea prezicerilor")

calibrate.calibrate_volume()

print("Terminat - Pas 4 - Calibrare")

print("Inceput - Pas 5 - Fractia de ejectie")

print("   > Calcularea fractiei de ejectie")
print("   > Stabilirea diagnosticului")

print('---------------------------------------------------------------------------------------------------')

diagnostic.calculate_ej_predicted()

print('---------------------------------------------------------------------------------------------------')

diagnostic.calculate_ej_real()

print('---------------------------------------------------------------------------------------------------')

print("Terminat - Pas 5 - Fractia de ejectie")
