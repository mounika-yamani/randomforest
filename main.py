from flask import Flask, render_template, request
import jsonify
from pycaret.regression import *
import pandas as pd
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('first_assignment.pkl', 'rb'))
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():

    CRIM = float(request.form['CRIM'])
    ZN=float(request.form['ZN'])
    INDUS=float(request.form['INDUS'])
    CHAS=request.form['CHAS']
    if(CHAS=='0'):
        CHAS=0
    else:
        CHAS=1
    NOX = float(request.form['NOX'])
    RM = float(request.form['RM'])
    AGE = float(request.form['AGE'])
    DIS = float(request.form['DIS'])
    RAD = float(request.form['RAD'])
    TAX = float(request.form['TAX'])
    PTRATIO = float(request.form['PTRATIO'])
    B = float(request.form['B'])
    LSTAT = float(request.form['LSTAT'])
    final_features = np.array([CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT])
    data_unseen = pd.DataFrame([final_features],columns=cols)
    data_unseen.reset_index(inplace=True)
    prediction = predict_model(model,data=data_unseen, round=0)
    output = float(prediction.Label[0])
    return render_template('index.html',prediction_text="House price is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)