from flask import Flask, render_template, request
import jsonify
import pandas as pd
import requests
import pickle
import numpy as np
import joblib
app = Flask(__name__)
model = pickle.load(open('first_assignment.sav','rb'))
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
    prediction = model.predict(data_unseen)
    return render_template('index.html',prediction_text="House price is {}".format(prediction))



if __name__=="__main__":
    app.run(debug=True)