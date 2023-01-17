from flask import Flask, request, jsonify, render_template, session, redirect, url_for, session
import requests
import pandas as pd
import numpy as np
import pickle
import os

model_dir = os.path.abspath('../models')
model_rf = pickle.load(open(model_dir+'/model.pkl', 'rb'))

template_dir = os.path.abspath('../templates')
app = Flask(__name__, template_folder=template_dir)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def getvalue():
    taxUser = request.form['tax']
    incomeUser = request.form['income']
    highwaysUser = request.form['highways']
    licenseUser = request.form['license']
    res = model_rf.predict([[taxUser, incomeUser, highwaysUser, licenseUser]])
    return render_template('index.html', res = res)
    
if __name__ =='__main__':
    app.run(debug=True)