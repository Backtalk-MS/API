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

"""
Helper functions to be used by server.py
"""

"""Loads the locally saved ML model. To be used for classification of bugs"""
def load_model_():
    return

"""Returns prediction of bug classifier"""
def predict():
    return

"""Takes in parameters of all attributtes that need to be returned.
Constructs properly formatted json of these attributes. Returns said json.
Might need to create a separate function for each individual JSON object"""
def construct_json():
    return
