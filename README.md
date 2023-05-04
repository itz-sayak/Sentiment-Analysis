# Sentiment-Analysis
## DATASET
The IMDB movie reviews dataset is available on the following website:

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
### Separately run this code in a .py file to save the tokenizer and dump as pickle format


# save tokenizer object as a pickle file when training before saving the model
```
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
```
