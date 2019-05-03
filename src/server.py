'''
This code takes the JSON data while POST request an performs a sentiment analysis using TextBlob model and returns
the results in JSON format.
'''

# Import libraries
from flask import Flask, request, jsonify, flash, redirect, url_for
import os, train, helpers
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '../uploads'
ALLOWED_EXTENSIONS = set(['txt', 'json', 'pdf'])

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
    user = data['username']
    password = data['password']
    model_id = data['model_id']
    dataset_id = data['dataset_id']

    train_label = data['train_label']
    train_content = data['train_content']
    dataset_name = 'test1'

    JSON_model = helpers.retrieve_json_model(user, password, model_id)
    training_data = helpers.load_JSON_dataset(user, password, dataset_id)
    data = helpers.prepare_json_data(training_data, train_label, train_content, dataset_name)
    
    return

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
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
            
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

"""Generic model trains on a new dataset. Model ID is stored in the Database
This model can then be used for prediction purposes in the future""" 
# Necessary Params:
# Location of the generic model
# Location of the dataset
# Label, and category columns identified
# - Output -
# Trained model will then be sent to the database.
@app.route('/train_generic', methods=['POST'])
def train_generic():
    data = request.get_json(force=True)

    # Retrieves the model from the database
    model = helpers.retrieve_json_model(data['modelSource']['username'], data['modelSource']['password'], data['modelSource']['modelID'])
    categories_to_train_on = data['categories']
    training_labels = data['labels']

    trained_model = train.train_new_model(model)

    train.save_model_to_db(trained_model)
    #TODO: Request JSON containing all info such as link to Database that will contain the dataset
    # Also, needs a "labels" and "categories" from the categories so model knows which items to train on
    return


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(port=5000, debug=True)