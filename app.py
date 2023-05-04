from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load pre-trained model and tokenizer
model = load_model('model/model.h5')
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define maximum input length and label names
MAX_LEN = 100
LABELS = ['negative', 'positive']

@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from user
    text = request.json['text']
    
    # Tokenize input text and pad sequences to fixed length
    tokenized_text = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(tokenized_text, maxlen=MAX_LEN)
    
    # Make prediction with pre-trained model
    prediction = model.predict(padded_text)[0]
    predicted_label = LABELS[int(round(prediction[0]))]
    
    # Return JSON response with predicted label
    return jsonify({'label': predicted_label})
