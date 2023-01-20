from flask import Flask, request, jsonify, render_template, session, redirect, url_for, session
import requests
import pandas as pd
import numpy as np
import pickle
import os

from typing import Dict, List, Optional
from datetime import datetime

from prometheus_client import make_asgi_app, Counter, Histogram


# Prometheus Metrics
# ------------------
## Request counts
prediction_counter = Counter('num_prediction_requests', "Number of 'prediction' requests made")
model_information_counter = Counter('num_model_info_requests', "Number of 'model_information' requests made")
## Response Distributions
prediction_hist = Histogram('prediction_output_distribution', 'Distribution of prediction outputs')
prediction_score_hist = Histogram('prediction_score_distribution', 'Distribution of prediction scores')
prediction_latency_hist = Histogram('prediction_latency_distribution', 'Distribution of prediction response latency')


model_dir = os.path.abspath('../models')
model = pickle.load(open(model_dir+'/model.pkl', 'rb'))

template_dir = os.path.abspath('../templates')
app = Flask(__name__, template_folder=template_dir)
metrics_app = make_asgi_app()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def getvalue():
    taxUser = request.form['tax']
    incomeUser = request.form['income']
    highwaysUser = request.form['highways']
    licenseUser = request.form['license']
    res = model.predict([[taxUser, incomeUser, highwaysUser, licenseUser]])
    return render_template('index.html', res = res)

@app.route('/prediction', methods =['GET','POST'])
def predict():
    """function to predict

    Args:
        req (dict): feature vector

    Example:
        json = {
            "features" : [0,0,1,1],
        }
    """
    req = request.json

    start = datetime.now()
    prediction_counter.inc()

    response = {}
    
    prediction = model.predict([req['features']])

    response["score"] = float(prediction[0])

    # record prediction score in histogram
    prediction_score_hist.observe(response["score"])
    
    latency = datetime.now() - start
    
    # record latency in histogram
    prediction_latency_hist.observe(latency.total_seconds())
    
    return response

@app.route("/model_information", methods =['GET'])
def model_information():
    '''
    Get the model's parameters
    '''
    # increment the count of 'model_information' requests made every time a request is made
    model_information_counter.inc()

    return model.get_params()
    
if __name__ =='__main__':
    app.run(debug=True)