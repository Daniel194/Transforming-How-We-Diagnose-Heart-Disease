import pandas

PREDICT_FILE_PATH = "result/prediction_calibrated_vgg.csv"


def diagnostic(ej):
    """
    Diagnostic based on ejection fraction
    :param ej: ejection fraction
    :return: nothing
    """

    d = ""

    if (ej >= 75):
        d = "Hiperdinamic"
    elif (ej < 75 and ej >= 55):
        d = "Normala"
    elif (ej < 55 and ej >= 45):
        d = "Usor anormala"
    elif (ej < 45 and ej >= 35):
        d = "Moderat anormala"
    else:
        d = "Sever anormala"

    print("Diagnostic : ", d)


def ejection_fraction(vd, vs):
    """
    Ejection fraction
    :param vd: diastole volume
    :param vs: systole volume
    :return: ejection fraction
    """

    return 100 * ((vd - vs) / vd)


def calculate_ej_predicted():
    """
    Calculate ejection fraction for predicted data
    :return: nothing
    """

    data = pandas.read_csv(PREDICT_FILE_PATH, sep=";")

    vd = data['cal_pred_dia'][0]
    vs = data['cal_pred_sys'][0]

    ej = ejection_fraction(vd, vs)

    print('Diastola prezisa: ', vd)
    print('Systola prezisa: ', vs)
    print('Fractia de ejectie prezisa: ', ej)

    diagnostic(ej)


def calculate_ej_real():
    """
    Calculate ejection fraction for true value
    :return: nothing
    """

    data = pandas.read_csv(PREDICT_FILE_PATH, sep=";")

    vd = data['Diastole'][0]
    vs = data['Systole'][0]

    ej = ejection_fraction(vd, vs)

    print('Diastola reala: ', vd)
    print('Systola reala: ', vs)
    print('Fractia de ejectie reala: ', ej)

    diagnostic(ej)
