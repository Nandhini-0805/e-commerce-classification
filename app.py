import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Create a Flask app
app = Flask(__name__)

#load the model
model = pickle.load(open("random_forest_model.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    text_features = [x for x in request.form.values()]
    features = [np.array (text_features)]
    prediction = model.predict(features)

    return render_template('index.html', prediction_text='Prediction is {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)
