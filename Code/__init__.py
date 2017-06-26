import step1_preprocess as preprocess

########## STEP 1 - Preprocess ##########

preprocess.convert_sax_images(rescale=True, base_size=256, crop_size=256)
preprocess.create_csv_data()
preprocess.enrich_dicom_csvdata()
preprocess.enrich_traindata()

print("Done - Step 1")

########## STEP 2 - Segmentation ##########
