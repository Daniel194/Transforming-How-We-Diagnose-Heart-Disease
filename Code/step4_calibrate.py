import pandas
from sklearn.ensemble import GradientBoostingRegressor

import utils.settings as settings

MODEL_NAME = settings.MODEL_NAME
TRAIN_PATH = settings.DATA_DIR + "train_gbr.csv"
PREDICT_PATH = settings.RESULT_DIR + "prediction_raw_" + MODEL_NAME + ".csv"


def calibrate_volume():
    """
    Calibrate volume predicted using Gradient Boosting Regression
    :return: nothing
    """
    train_data = pandas.read_csv(TRAIN_PATH, sep=";")
    pred_data = pandas.read_csv(PREDICT_PATH, sep=";")
    ext = ""

    for dia_sys in ["dia" + ext, "sys" + ext]:
        new_predictions = []
        real_value_col = "Diastole" if dia_sys.startswith("dia") else "Systole"
        mul_col = "mul_" + dia_sys
        plane_map = {"ROW": 1, "COL": 0}
        sex_map = {"M": 1, "F": 0}
        feature_names = ["rows", "columns", "spacing", "slice_thickness", "slice_count", "up_down_agg", "age_years",
                         "small_slice_count", "pred_sys" + ext, "pred_dia" + ext, "angle", real_value_col, mul_col]

        train_data[mul_col] = train_data[real_value_col] / train_data["pred_" + dia_sys]
        train_data["sex_val"] = train_data["sex"].map(sex_map)
        train_data["plane_val"] = train_data["plane"].map(plane_map)
        train_data["pixels"] = train_data["rows"] * train_data["columns"] * train_data["spacing"]

        pred_data[mul_col] = pred_data[real_value_col] / pred_data["pred_" + dia_sys]
        pred_data["sex_val"] = pred_data["sex"].map(sex_map)
        pred_data["plane_val"] = pred_data["plane"].map(plane_map)
        pred_data["pixels"] = pred_data["rows"] * pred_data["columns"] * pred_data["spacing"]

        tmp_train = train_data[(train_data["patient_id"] <= 700) & (train_data["slice_count"] > 7)]

        x_train = tmp_train[feature_names]
        y_train = tmp_train["error_" + dia_sys]

        tmp_validate = pred_data

        x_validate = tmp_validate[feature_names]

        del x_train[real_value_col]
        del x_validate[real_value_col]
        del x_validate[mul_col]
        del x_train[mul_col]

        cls = GradientBoostingRegressor(learning_rate=0.001, n_estimators=2500, verbose=False, max_depth=3,
                                        min_samples_leaf=2, loss="ls", random_state=1301)
        cls.fit(x_train, y_train)

        y_pred = cls.predict(x_validate)

        new_predictions += (x_validate["pred_" + dia_sys] - y_pred).map(lambda x: round(x, 2)).values.tolist()

        pred_data["cal_pred_" + dia_sys] = new_predictions
        pred_data["cal_error_" + dia_sys] = new_predictions - pred_data[real_value_col]
        pred_data["cal_abserr_" + dia_sys] = abs(pred_data["cal_error_" + dia_sys])

    pred_data = pred_data[
        ["patient_id", "slice_count", "age_years", "sex", "normal_slice_count", "Diastole", "Systole", "cal_pred_dia",
         "cal_error_dia", "cal_abserr_dia", "cal_pred_sys", "cal_error_sys", "cal_abserr_sys", "pred_dia", "error_dia",
         "abserr_dia", "pred_sys", "error_sys", "abserr_sys"]]

    pred_data.to_csv(settings.RESULT_DIR + "prediction_calibrated_" + MODEL_NAME + ".csv", sep=";")


if __name__ == "__main__":
    calibrate_volume()
