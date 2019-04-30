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


# train.py
# used for training machine learning models and uploading to database

# Train the original generic model on a new dataset
def train_generic(jsonModel):

    return

# NOTE TO SELF:
# Combine both functions into one. When training on original model, modelID points to original model
# When training new model, modelID points to new model ID

# Train a new model as described by the developer on a new dataset.
def train_new_model(JSON_model):
    trained_model = JSON_model
    return trained_model


def save_model_to_db(trained_model):
    return