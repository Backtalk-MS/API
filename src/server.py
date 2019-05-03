'''
This code takes the JSON data while POST request an performs a sentiment analysis using TextBlob model and returns
the results in JSON format.
'''

# Import libraries
from flask import Flask, request, jsonify, flash, redirect, url_for, json
import os, train, helpers
import pandas as pd
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '../uploads'
ALLOWED_EXTENSIONS = set(['json'])

# Instantiate Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'my key'
localPath = os.path.dirname(os.path.abspath(__file__))

#TODO: On server startup, deserialize dictionary containing model names,
# mapped to location of models on disk

# Prediction Function as an endpoint
@app.route('/predict',methods=['POST'])
def predict():

    # Get the data from the POST request.
    data = request.get_json(force=True)

    text = data['text']
    category = data['category']
    endpoint = data['endpoint']
    model_id = data['model_id'] #If reading model from DB
    model_name = data['model_name'] # ex: thisModel.h5

    #if category == review then get rating

    # Output only the sentiment
    sentiment = helpers.get_sentiment(text)
    prediction = helpers.predictCategory(text, model_name)
    # Return output

    return jsonify(prediction[0])


# Upload dataset in JSON format
@app.route('/train', methods=['GET','POST'])
def upload_file():

    modelID = request.form['modelID']

    myfile = request.files['file']

    data = json.loads(myfile.read().decode('utf8'))
    training_data = helpers.prepare_json_data(data, 'category', 'content')
    json_model = helpers.load_local_json()
    train.train_new_model(json_model, training_data, 'content', 'category', modelID)

    return '''
    Model Successfully Trained
    '''



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(port=5000, debug=True)

    # once done training, turn model id ready to true