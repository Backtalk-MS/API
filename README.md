# API

Python libraries that need to be installed: 
Flask, TextBlob


Machine Learning models are deployed using flask. Server.py receives a json object containing the text of the review (as well as
other attributes in the future). 
Using the TextBlob library, a sentiment analysis is run on the text and returned as a result.
