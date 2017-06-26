import step1_preprocess as preprocess
import utils.sunnybrook as sunnybrook
import numpy as np

##### STEP 1 - Preprocess #####

preprocess.convert_sax_images(rescale=True, base_size=256, crop_size=256)
# preprocess.create_csv_data()
# preprocess.enrich_dicom_csvdata()
# preprocess.enrich_traindata()
#
# # Convert Sunnybrook dataset from DICOM form to PNG format
# train, val = sunnybrook.get_all_contours()
# ctrs = np.append(train, val)
# sunnybrook.convert_dicom_to_png(ctrs)

print("Done - Step 1")
