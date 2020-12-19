import pickle
import numpy as np
from model import text_processing
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
with open('finalized_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features = request.form["Tweet"]
	int_features=[int_features]
	prediction = model.predict(int_features)
	if prediction==0:
		output="Negative Sentiment"
	else:
		output = "Positive Sentiment "
	return render_template('index.html', prediction_text='Tweet has {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
	prediction = model.predict(int_features)
	if prediction==0:
		output="Negative Sentiment"
	else:
		output = "Positive Sentiment "
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)