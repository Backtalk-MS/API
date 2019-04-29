'''
This code takes the JSON data while POST request an performs a sentiment analysis using TextBlob model and returns
the results in JSON format.
'''

# Import libraries
from flask import Flask, request, jsonify
import os, helpers

# Instantiate Flask
app = Flask(__name__)

localPath = os.path.dirname(os.path.abspath(__file__))


# Prediction Function as an endpoint
@app.route('/api',methods=['POST'])
def predict():

    # Get the data from the POST request.
    data = request.get_json(force=True)

    text = data['text']
    category = data['category']
    endpoint = data['endpoint']
    #if category == review then get rating

    # Output only the sentiment
    sentiment = helpers.get_sentiment(text)
    #prediction = helpers.predictCategory(text)
    # Return output
    return jsonify(sentiment)


"""Generic model trains on a new dataset. Model ID is stored in the Database
This model can then be used for prediction purposes in the future""" 

@app.route('/train_generic', methods=['POST'])
def train_generic():
    #TODO: Request JSON containing all info such as link to Database that will contain the dataset
    # Also, needs a "labels" and "categories" from the categories so model knows which items to train on
    return


"""New model trains on a new dataset. Model ID is stored in the Database
This model can then be used for prediction purposes in the future""" 
@app.route('/train', methods=['POST'])
def train():
    #TODO: Request JSON containing all info such as link to Database that will contain the dataset
    # Also, needs a "labels" and "categories" from the categories so model knows which items to train on
    # Also, json input will contain the format of a model. (Look at keras function save_json() for examples on this format. 
    # Keras can use this to construct a model from JSON rather than from code. Model.fit() function can then be run in order
    # to train this model. The .h5 file model weights, and .JSON model structure are both saved to the DB after training.
    return

if __name__ == '__main__':
    app.run(port=5000, debug=True)