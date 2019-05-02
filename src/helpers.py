import pandas as pd
import nltk, re, string, numpy, pickle, os, sys, json, pprint
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
from bson.objectid import ObjectId
from bson.json_util import dumps

"""
Helper functions to be used by server.py
"""
#TEST
# For testing purposes only
def test_db_insert(data):
    client = MongoClient('mongodb://backtalk:backtalk123@ds038888.mlab.com:38888/backtalkdev')
    db = client['backtalkdev']
    coll = db['newCollection']
    coll.insert_one(data)
    return

# Will only retreive the JSON architecture of the model
# WILL NOT run model.compile to build the model
""" Loads a JSON file containing the model architecture. This is then compiled, trained, and saved to database."""
def retrieve_json_model(username, password, modelID):
    newmodelID = "5cc76ae2e7179a596b183e02"
    client = MongoClient('mongodb://'+ username + ':' + password + '@ds038888.mlab.com:38888/backtalkdev')
    db = client['backtalkdev']
    coll = db['jsonModels']

    JSON_model = coll.find_one({'_id' : ObjectId(newmodelID)})
    x = bson.json_util.dumps(JSON_model)
    print(JSON_model)
    #print(coll.find_one({'_id' : ObjectId(newmodelID)}))
    return




""" Loads the serialized, trained model from the database. No training required """
def load_model_from_database(username, password, modelID):

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
    print(prediction[0:])
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

retrieve_json_model('backtalk', 'backtalk123', 'test')
exit()