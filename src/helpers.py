import pandas as pd
import nltk, re, string, numpy, pickle, os, sys
from pathlib import Path
from tensorflow.keras import models
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Activation, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from textblob import TextBlob
from pymongo import MongoClient

"""
Helper functions to be used by server.py
"""

# For testing purposes only
def test_db_insert(data):
    client = MongoClient('mongodb://backtalk:backtalk123@ds038888.mlab.com:38888/backtalkdev')
    db = client['backtalkdev']
    coll = db['newCollection']
    coll.insert_one(data)
    return

def retrieve_JSON_model(username, password, modelID):
    return


"""Loads the locally saved ML model. To be used for classification of bugs"""
def load_model_from_database(modelSource):

    #model = load_model
    return


# Possibly input this into server.py
# Server.py will load model once at start and use it for all
# predictions. helpers.py loads it every time a prediction is made,
# and this is not very efficient
"""Returns prediction of bug classifier"""
def predictCategory(text, modelSource):

    # Currently loading model and tokenizer from local machine. Needs to load from mongoDB
    model = load_model('../model/trainedModel.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    with open('../model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    matrixedInput = tokenizer.texts_to_matrix(text, mode='tfidf')    

    prediction = model.predict(numpy.array(matrixedInput))
    print(prediction[0])
    return prediction


#Not sure what I was planning to do with this function
"""Takes in parameters of all attributtes that need to be returned.
Constructs properly formatted json of these attributes. Returns said json.
Might need to create a separate function for each individual JSON object"""
def construct_json(text, category, rating, endpoint):
    return


# Ayy, one function that works properly
"""Return the sentiment of a given text input [-1, 1]"""
def get_sentiment(text):
    analysis = TextBlob(text).sentiment
    return analysis[0]
