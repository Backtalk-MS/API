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


"""Params: JSON_model architecture file, prepared dataset in a python dataframe.
Content, and label. These should be equal to the names of the columns in the data frame.
Content is the text, and label is the label associated with that text."""
def train_new_model(JSON_model, dataset, content, label, modelID):
    #Build model from JSON file
    train_size = int(len(dataset) * .8)

    train_text = dataset[content][:train_size]
    train_tags = dataset[label][:train_size]

    test_text = dataset[content][train_size:]
    test_tags = dataset[label][train_size:]

    num_labels = len(set(dataset[label]))
    vocab_size = 1000
    batch_size = 20

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
    
    save_trained_model(compiled_model, tokenizer, modelID)
    helpers.model_ready(modelID)
    return# compiled

""" Saves a reference of a local model to model dictionary"""
def save_trained_model(trained_model, tokenizer, modelID):
    # Opens serialized model info
    with open('../model/model_dict.pickle', 'rb') as handle:
        model_dict = pickle.load(handle)
    
    model_path = '../model/trained/'

    my_dic = {}
    my_dic['tokenizer_path'] = model_path + modelID + '.pickle'
    my_dic['model_path'] = model_path + modelID + '.h5'
    model_dict[modelID] = my_dic

    trained_model.save(model_dict[modelID]['model_path'])
    
    # Saves tokenizer
    with open(model_dict[modelID]['tokenizer_path'], 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Saves serialized model info
    with open('../model/model_dict.pickle', 'wb') as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return
