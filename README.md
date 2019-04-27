# API

Python libraries that need to be installed: 
Flask, TextBlob


Machine Learning models are deployed using flask. Server.py receives a json object containing the text of the review, the category
that the text is classified as, and any other attributes associated with that text input.
Using the TextBlob library, a sentiment analysis is run on the text and returned as a result.
If the text is categorized as a bug, it is also run through the bug classification machine learning model.
