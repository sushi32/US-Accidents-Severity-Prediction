from flask import Flask, render_template, request
import json
import pickle as pkl
import tensorflow as tf
import numpy as np


app = Flask(__name__)


@app.route('/')
def index_render():
    return render_template("index.html")


@app.route('/models')
def model_page_view():
    return render_template("model_page.html")


@app.route('/charts')
def test_page_view():
    with open('../json_files/nan_counts_percent.json', 'r') as f:
        nan_json = json.load(f)
    with open('../json_files/state_accidents_counts.json', 'r') as f:
        states_json = json.load(f)
    return render_template("plots_charts.html", nan_json = nan_json, states_json = states_json)


@app.route('/visualization')
def new_page_view():
    return render_template("visualization.html")


@app.route('/predict', methods=['POST'])
def predict():
    """
    form details values order:
    0:Hour, 1:weekDay, 2:traffic signal, 3:crossing, 4:junction, 5:Temperature,
    6:wind chill, 7:wind speed, 8:humidity, 9:pressure, 10:visibility

    minMaxScaler input order:
    'Wind_Speed(mph)','Pressure(in)','Humidity(%)','Visibility(mi)','Temperature(F)', 'Wind_Chill(F)'

    standardScaler input order:
    'Wind_Speed(mph)','Pressure(in)','Humidity(%)','Visibility(mi)','Temperature(F)', 'Wind_Chill(F)',
    'Traffic_Signal', 'Crossing','Junction','Hour','Day'

    predictor model input order:
    'Wind_Speed(mph)','Pressure(in)','Humidity(%)','Visibility(mi)','Temperature(F)', 'Wind_Chill(F)',
    'Traffic_Signal', 'Crossing','Junction','Hour','Day'
    """

    stdScalerModel = pkl.load(open('../data/standardScalerModel.pkl', 'rb'))
    predictorModel = tf.keras.models.load_model('../data/bestModel.hdf5', custom_objects=None,
                                                compile=True, options=None)

    inputValues = [float(x) for x in request.form.values()]
    print(inputValues)
    stdValues = stdScalerModel.transform([[inputValues[7], inputValues[9], inputValues[8], inputValues[10],
                                           inputValues[5], inputValues[6], inputValues[2], inputValues[3],
                                           inputValues[4], inputValues[0], inputValues[1]]])
    predictedScores = predictorModel.predict(stdValues)
    print(predictedScores)
    # classification = np.where(predictedScores[0] == np.amax(predictedScores[0]))[0][0]
    classification = np.argmax(predictedScores) + 1
    ind = np.argpartition(predictedScores[0], -2)
    print(ind)
    percent = round(((predictedScores[0][ind[-1]] + predictedScores[0][ind[-2]]) * 100), 2)
    severity = "low"
    if classification > 2:
        severity = "high"
    return render_template("severity_prediction.html", predicted=classification, severity=severity, percent=percent)


if __name__ == '__main__':
    app.run()
