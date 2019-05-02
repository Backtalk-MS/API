import pandas as pd
import nltk, re, string, numpy, pickle, os, sys, gridfs, io
from pathlib import Path
from tensorflow.keras import models
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Activation, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from pymongo import MongoClient


# train.py
# used for training machine learning models and uploading to database

# Train the original generic model on a new dataset
def train_generic(jsonModel):

    return

# NOTE TO SELF:
# Combine both functions into one. When training on original model, modelID points to original model
# When training new model, modelID points to new model ID

#TEST
# Train a new model as described by the developer on a new dataset.
def train_new_model(JSON_model):
    #Build model from JSON file
    trained_model = model_from_json(JSON_model)

    #Compile model
    trained_model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    
    #model.fit(x_train, y_train, batch_size=batch_size, epochs=8, verbose=1, validation_data=(x_test, y_test))

    return trained_model


#TEST
# Currently just a test implementation to see if a model can be loaded into database
def save_model_to_db():
    client = MongoClient('mongodb://backtalk:backtalk123@ds038888.mlab.com:38888/backtalkdev')
    db = client['backtalkdev']
    fs = gridfs.GridFS(db, collection='modelUploadTest')

    f = io.FileIO('../model/trainedModel.h5', 'r')
    fileId = fs.put(f)
    f.close
    print(fileId)

    return

# TEST
def load_model_from_db():

    return


save_model_to_db()
exit()