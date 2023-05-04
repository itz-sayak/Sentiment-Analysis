# Sentiment-Analysis
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
