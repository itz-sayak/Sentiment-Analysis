# Sentiment-Analysis
This project is a web application that allows users to enter movie reviews, and the application will predict whether the reviews are positive or negative.

## DATASET
The IMDB movie reviews dataset is available on the following website:
https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

for more info. about the dataset see...
http://ai.stanford.edu/~amaas/data/sentiment/

The dataset consists of 50,000 movie reviews, with 25,000 reviews for training and 25,000 reviews for testing. Each review is labeled as positive or negative, and the dataset is evenly balanced between positive and negative reviews. You can download the dataset in a compressed format, and extract it using a zip utility. Once extracted, you should see separate folders for the training and testing data, each containing subfolders for positive and negative reviews.

To deploy the project as a web application, we can use a web framework like Flask or Django. Here's an example of how to deploy the sentiment analysis model as a Flask web application:

## 1. Install Flask and the required packages:
```
pip install Flask pandas numpy tensorflow matplotlib
```
## 2.Create a new file called app.py
Create two new HTML files in the same directory as app.py: index.html and result.html. index.html will contain the form for the user to input a review, and result.html will display the sentiment prediction.

## 3.Run the Flask app
```
python app.py
```

## 4.Open a web browser and go to http://localhost:5000 to see the web application.

## PROJECT STRUCTURE
```
sentiment-analysis/
├── app.py
├── model/
│   ├── model.h5
│   └── tokenizer.pkl
├── static/
│   └── styles.css
└── templates/
    └── index.html
    └── result.html
```
#### app.py: 
This file contains the Flask application that serves the web pages and makes predictions using the trained model.
#### model.h5: 
This file contains the trained RNN model that is used to make predictions on new reviews.
#### tokenizer.pickle: 
This file contains the tokenizer object that is used to preprocess the text data before feeding it to the model.
#### templates/:
This folder contains the HTML templates for the web pages served by the Flask application.
#### static/:
This folder contains the CSS file for styling the HTML templates.

## MODEL ARCHITECTURE
The RNN model architecture used in this project is a bidirectional LSTM with two layers of LSTM cells and a dense output layer with a sigmoid activation function. The model takes in sequences of integers representing words and outputs a binary classification result (positive or negative sentiment).
It  uses dropout layers too to prevent overfitting.

## REST API
This project includes a simple REST API that can be used to make predictions programmatically. The API has a single endpoint (/predict) that accepts a JSON object containing a single key (text) with the movie review text. The API returns a JSON object with a single key (sentiment) that is either "positive" or "negative".
To use the API, make a POST request to http://localhost:5000/predict with the following JSON body:
```
{
    "text": "This movie was really great. I loved it!"
}
```

### save tokenizer object as a pickle file when training before saving the model
```
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
```
## Credits
This project was created by me. The dataset used for training the model was obtained from Kaggle.
