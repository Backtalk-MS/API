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

"""
Helper functions to be used by server.py
"""

"""Loads the locally saved ML model. To be used for classification of bugs"""
def load_model_():
    #model = load_model
    return


# Possibly input this into server.py
# Server.py will load model once at start and use it for all
# predictions. helpers.py loads it every time a prediction is made,
# and this is not very efficient
"""Returns prediction of bug classifier"""
def predictCategory(text):
    model = load_model('trainedModel.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    matrixedInput = tokenizer.texts_to_matrix(text, mode='tfidf')    

    prediction = model.predict(numpy.array(matrixedInput))
    print(prediction[0])
    return prediction

"""Takes in parameters of all attributtes that need to be returned.
Constructs properly formatted json of these attributes. Returns said json.
Might need to create a separate function for each individual JSON object"""
def construct_json(text, category, rating, endpoint):
    return


"""Return the sentiment of a given text input [-1, 1]"""
def get_sentiment(text):
    analysis = TextBlob(text).sentiment
    return analysis[0]