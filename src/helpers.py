import pandas as pd
import nltk, re, string, numpy, pickle, os, sys, json, pprint
from pathlib import Path
from tensorflow.keras import models  # pylint: disable=import-error
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Activation, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from textblob import TextBlob
from pymongo import MongoClient
from bson import ObjectId
from bson.json_util import dumps

"""
Helper functions to be used by server.py
"""

# Had to make a JSON encoder for ObjectID.
# Have no idea why but it solved my problem?
class JSONEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=method-hidden
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o) 


# Will only retreive the JSON architecture of the model
# WILL NOT run model.compile to build the model
""" Loads a JSON file containing the model architecture. This can then be compiled, trained, and saved to database."""
def retrieve_json_model(username, password, modelID):
    #newmodelID = "5cc76ae2e7179a596b183e02"
    client = MongoClient('mongodb://'+ username + ':' + password + '@ds038888.mlab.com:38888/backtalkdev')
    db = client['backtalkdev']
    coll = db['jsonModels']

    # Finds JSON model based on model ID
    JSON_model = coll.find_one({'_id' : ObjectId(modelID)})
    JSON_model = JSONEncoder().encode(JSON_model)
    JSON_model = json.loads(JSON_model)
    
    #print(JSON_model)
    return JSON_model



def load_local_json():
    with open('../model/json_model/generic_model.json') as json_file:
        model = json.load(json_file)
    return model


# Possibly input this into server.py
# Server.py will load model once at start and use it for all
# predictions. helpers.py loads it every time a prediction is made,
# and this is not very efficient
"""Returns prediction of bug classifier"""
def predictCategory(text, model_name):
    # path[] = ['model_path', 'pickle_path]
    path = load_from_path(model_name)
    
    #pickle_path = load_from_path(model_name) # EX:pickle_path = '../model/tokenizer.pickle'

    model = load_model(path[0]) #ex: model_path = '../model/trainedModel.h5'
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Opening tokenizer might not be necessary for accurate predictions
    # and may be the reason for poor predictions
    # Will sort this out later
    with open(path[1], 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    matrixedInput = tokenizer.texts_to_matrix(text, mode='tfidf')    

    prediction = model.predict(numpy.array(matrixedInput))
    #print(prediction[0:])
    return prediction



""" Loads a dataset in JSON format from the specified database"""
def load_JSON_dataset(username, password, datasetID):
    client = MongoClient('mongodb://'+ username + ':' + password + '@ds038888.mlab.com:38888/backtalkdev')
    db = client['backtalkdev']
    coll = db['jsonDatasets']

    # Finds JSON model based on model ID
    dataset = coll.find_one({'_id' : ObjectId(datasetID)})
    dataset = JSONEncoder().encode(dataset)
    dataset = json.loads(dataset)
    return dataset


""" Loads json dataset into Pandas DataFrame"""
def prepare_json_data(data, train_label, train_content):
    train_texts = []
    # Load json data into pandas DataFrame
    # Return DataFrame with data
    #json_file = json.load(data)
    for p in data:
        train_texts.append((p[train_content], p[train_label]))

    training_data = pd.DataFrame.from_records(train_texts, columns=[train_content, train_label])
    training_data = training_data.sample(frac=1).reset_index(drop=True)

    return training_data

# Ayy, one function that works properly
"""Return the sentiment of a given text input [-1, 1]"""
def get_sentiment(text):
    analysis = TextBlob(text).sentiment
    return analysis[0]

""" Loads the path of the model based on model name. """
def load_from_path(model_name):
    #TODO: Retrieve the model path based on its name.
    model_path = []

    #Returns stubs for now.
    model_path.append('model_path')
    model_path.append('pickle_path')
    return model_path





#retrieve_json_model('backtalk', 'backtalk123', 'test')
#dataset = load_JSON_dataset('backtalk', 'backtalk123', '5cca6f6ffb6fc00ed59f48ec')
#data = prepare_json_data(dataset, 'category', 'content', 'test')
#print(dataset)

