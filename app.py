from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('imdb_reviews.csv')

# Split the data into training and testing sets
train_data = df[:40000]
test_data = df[40000:]

# Preprocess the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data['review'].values)
train_sequences = tokenizer.texts_to_sequences(train_data['review'].values)
test_sequences = tokenizer.texts_to_sequences(test_data['review'].values)

train_sequences_padded = pad_sequences(train_sequences, maxlen=100)
test_sequences_padded = pad_sequences(test_sequences, maxlen=100)

# Define the RNN architecture
inputs = Input(shape=(100,))
x = Embedding(5000, 32)(inputs)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.5)(x)
x = Bidirectional(LSTM(32))(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_sequences_padded, train_data['sentiment'].values, batch_size=64, epochs=5, validation_split=0.2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    review_sequence = tokenizer.texts_to_sequences([review])
    review_sequence_padded = pad_sequences(review_sequence, maxlen=100)
    sentiment_prediction = model.predict(review_sequence_padded)
    if sentiment_prediction > 0.5:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
    return render_template('result.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
