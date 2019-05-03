import pandas as pd
import nltk, re, string, numpy, pickle, os, sys, gridfs, io, json, helpers
from pathlib import Path
from tensorflow.keras import models  # pylint: disable=import-error
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Activation, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from pymongo import MongoClient


# train.py
# used for training machine learning models and uploading to database

# NOTE TO SELF:
# Combine both functions into one. When training on original model, modelID points to original model
# When training new model, modelID points to new model ID

# Train a new model as described by the developer on a new dataset.
"""Params: JSON_model architecture file, prepared dataset in a python dataframe.
Content, and label. These should be equal to the names of the columns in the data frame.
Content is the text, and label is the label associated with that text."""
def train_new_model(JSON_model, dataset, content, label):
    #Build model from JSON file

    train_size = int(len(dataset) * .8)

    train_text = dataset[content][:train_size]
    train_tags = dataset[label][:train_size]

    test_text = dataset[content][train_size:]
    test_tags = dataset[label][train_size:]

    num_labels = len(set(dataset[label]))
    vocab_size = 1000
    batch_size = 20

    #should be :
    #tokenizer = text.Tonekiner(num_words=vocab_size)
    #where text = all the words in the text corpus
    tokenizer = Tokenizer(num_words=vocab_size)

    tokenizer.fit_on_texts(train_text)

    x_train = tokenizer.texts_to_matrix(train_text, mode='tfidf')
    x_test = tokenizer.texts_to_matrix(test_text, mode='tfidf')

    encoder = LabelBinarizer()
    encoder.fit(train_tags)
    y_train = encoder.transform(train_tags)
    y_test = encoder.transform(test_tags)

    model = json.dumps(JSON_model)
    compiled_model = model_from_json(model)
    compiled_model.summary()
    #Compile model
    compiled_model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    
    compiled_model.fit(x_train, y_train, batch_size=batch_size, epochs=8, verbose=1, validation_data=(x_test, y_test))
    #score = myModel.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

    #print('Test Accuracy: ', score[1])
    #print("Score: ", score)
    return# compiled


#TEST
# Currently just a test implementation to see if a model can be loaded into database
# Params to add:
# dbuser, dbpass, either model location or actual model,
# Return ObjectID of model once inserted into DB
#NOTE: Probably won't be part of demo so won't work on it for now
def save_model_to_db():
    client = MongoClient('mongodb://backtalk:backtalk123@ds038888.mlab.com:38888/backtalkdev')
    db = client['backtalkdev']

    coll = db["modelTest"]
    trained_model = open('../model/trainedModel.h5', 'r')

    coll.insert_one(trained_model)

    return

# TEST
def load_model_from_db():
    client = MongoClient('mongodb://backtalk:backtalk123@ds038888.mlab.com:38888/backtalkdev')
    db = client['backtalkdev']
    #fs = gridfs.GridFS(db, collection='modelUploadTest')

    modelUploadTest = gridfs.GridFS(db, collection="modelUploadTest")
    file = modelUploadTest.find_one({"filename" : "myModel"})
    #print(file.read())
    return
    
#TODO: Won't be used for demo. Ignore for now
""" Loads CSV dataset into Pandas DataFrame """
def prepare_csv_data(data):
    #training_data = pd.read_csv()
    return data


# Used for testing database insertion. Delete later
def insertJsonDataset():
    client = MongoClient('mongodb://backtalk:backtalk123@ds038888.mlab.com:38888/backtalkdev')
    db = client['backtalkdev']
    coll = db['jsonDatasets']
    with open('test1.json') as js:
        jsonfile = json.load(js)
    coll.insert_one(jsonfile)

    return


#dataset = helpers.load_JSON_dataset('backtalk', 'backtalk123', '5ccaa7cc82543b408f03faec')
#data = helpers.prepare_json_data(dataset, 'category', 'content', 'test1')
#json_model = helpers.retrieve_json_model('backtalk', 'backtalk123', '5cc76ae2e7179a596b183e02')
#train_new_model(json_model, data, 'content', 'category')

#exit()