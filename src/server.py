'''
This code takes the JSON data while POST request an performs a sentiment analysis using TextBlob model and returns
the results in JSON format.
'''

# Import libraries
from flask import Flask, request, jsonify, flash, redirect, url_for, json
import os, train, helpers, keras
import pandas as pd
from werkzeug.utils import secure_filename
from werkzeug.datastructures import ImmutableMultiDict
from nltk.tokenize import sent_tokenize, word_tokenize 

UPLOAD_FOLDER = '../uploads'
ALLOWED_EXTENSIONS = set(['json'])

# Instantiate Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'my key'
localPath = os.path.dirname(os.path.abspath(__file__))

#TODO: On server startup, deserialize dictionary containing model names,
# mapped to location of models on disk

@app.route("/predict/sentiment", methods=["POST"])
def predictSentiment():
    #Get data from the post
    data = request.form.to_dict()

    if(data["type"] == "feedback" or data["type"] == "review"):
        print("Performing sentiment analysis...")
        proccessedList = "this doesn't exist right now"
        jdata = {"processedText": proccessedList, "keyphrase": "none, dumby", "sentiment": helpers.get_sentiment(data["rawText"])}
    else:
        #Category
        print("Performing categorical analysis...")
        pass
    
    
    jdata2 = {"result": jdata}
    return jsonify(jdata2)

# Prediction Function as an endpoint
@app.route('/predict',methods=['POST'])
def predict():

    # Get the data from the POST request.
    print(request.get_json)
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




#TODO: Trained model needs to be input into the database. Object ID of the model and tokenizer should be return
# For our purposes, model can be saved locally, and name + file path of model and tokenizer is returned
"""New model trains on a new dataset. Model ID is stored in the Database
This model can then be used for prediction purposes in the future""" 
@app.route('/train', methods=['POST'])
def train_new():
    data = request.get_json(force=True)

#NOTE: Naming conventions might be different than what is being passed in,
# Get that sorted to make sure they are consistent between Web App and Training Server

    # Read all parameters from request
    model_id = data['model_id']

    train_label = data['train_label']
    train_content = data['train_content']
    dataset_name = 'test1'

    JSON_model = helpers.retrieve_json_model(user, password, model_id)
    training_data = helpers.load_JSON_dataset(user, password, dataset_id)
    data = helpers.prepare_json_data(training_data, train_label, train_content, dataset_name)
    
    return

# Upload dataset in JSON format
@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # Get Params
        modelID = request.form['modelID']

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            with open('../uploads/' + filename) as json_file:
                data = json.load(json_file)
                training_data = helpers.prepare_json_data(data, 'category', 'content')
        json_model = helpers.load_local_json()
        train.train_new_model(json_model, training_data, 'content', 'category')



    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(port=5000, debug=True)