import pandas as pd
import nltk, re, string, numpy, pickle, os, sys, gridfs, io, json
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
def train_new_model(JSON_model, dataset):
    #Build model from JSON file
    trained_model = model_from_json('../model/trainedModel.h5')

    #Compile model
    trained_model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    
    #model.fit(x_train, y_train, batch_size=batch_size, epochs=8, verbose=1, validation_data=(x_test, y_test))

    return trained_model


#TEST
# Currently just a test implementation to see if a model can be loaded into database
def save_model_to_db():
    client = MongoClient('mongodb://backtalk:backtalk123@ds038888.mlab.com:38888/backtalkdev')
    db = client['backtalkdev']

    coll = db["modelTest"]
    trained_model = open('../model/trainedModel.h5', 'r')

    coll.insert_one(trained_model)
    #fs = gridfs.GridFS(db, collection='modelUploadTest')

    #f = open('../model/trainedModel.h5', 'r')
    #f = io.FileIO('../model/trainedModel.h5', 'r')
    #fileId = fs.put(f, filename='myModel')
    #f.close
    #print(fileId)

    return

# TEST
def load_model_from_db():
    client = MongoClient('mongodb://backtalk:backtalk123@ds038888.mlab.com:38888/backtalkdev')
    db = client['backtalkdev']
    #fs = gridfs.GridFS(db, collection='modelUploadTest')

    modelUploadTest = gridfs.GridFS(db, collection="modelUploadTest")
    file = modelUploadTest.find_one({"filename" : "myModel"})
    #print(file.read())

    #with open("modelsavetest.h5")
    #file.save("modelsavetest.h5")
    #print(modelUploadTest.exists(filename="myModel"))
    #newFile = modelUploadTest.get("5cca363bd98fa4da7f115a0d")
    #newFile.read()
    return

""" Loads json dataset into Pandas DataFrame"""
def prepare_json_data(data, train_label, train_content):
    train_texts = []
    # Load json data into pandas DataFrame
    # Return DataFrame with data
    json_file = json.load(data)
    for p in json_file:
        train_texts.append((p[train_content], p[train_label]))

    training_data = pd.DataFrame.from_records(train_texts, columns=[train_content, train_label])
    training_data = training_data.sample(frac=1).reset_index(drop=True)

    return training_data

""" Loads CSV dataset into Pandas DataFrame """
def prepare_csv_data(data):
    #training_data = pd.read_csv()
    return data


save_model_to_db()
#load_model_from_db()
exit()