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
    prediction = helpers.predictCategory(text)
    # Return output
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)