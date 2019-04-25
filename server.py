'''
This code takes the JSON data while POST request an performs a sentiment analysis using TextBlob model and returns
the results in JSON format.
'''

# Import libraries
from textblob import TextBlob
from flask import Flask, request, jsonify
import os, helpers


app = Flask(__name__)
localPath = os.path.dirname(os.path.abspath(__file__))


@app.route('/api',methods=['POST'])
def predict():

    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Get only the text data from the json (Currently the only thing being passed in for testing purposes)
    text = data['text']

    # Perform sentiment analysis on the text input
    analysis = TextBlob(text).sentiment

    # Output only the sentiment, and not polarity or subjectivity
    output = analysis[0]

    # Return output
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)