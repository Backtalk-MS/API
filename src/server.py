'''
This code takes the JSON data while POST request an performs a sentiment analysis using TextBlob model and returns
the results in JSON format.
'''

# Import libraries
from flask import Flask, request, jsonify, flash, redirect, url_for, json
import os, train, helpers, random
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


"""# Prediction Function as an endpoint
@app.route('/predict/something',methods=['POST'])
def predict():

    data = request.get_json(force=True)
    text = data['text']
    model_id = data['modelID'] #If reading model from DB


    prediction = helpers.predictCategory(text, model_name)
    return jsonify(prediction[0])"""

@app.route('/predict/category',methods=['POST'])
def predict():

    text = request.form['text']
    modelID = request.form['modelID'] #If reading model from DB


    prediction = helpers.predictCategory(text, modelID)

    #THIS WILL GET CHANGED. 
    rando = random.randint(0,2)
    labels = ['bug', 'feature', 'complaint', 'feedback'] 
    # DUMMY PLACEHOLDER FOR NOW
    return jsonify(labels[rando])

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