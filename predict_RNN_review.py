import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

max_features = 10000 
max_len = 500 #input length

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

word_index = imdb.get_word_index()

# index -> word
index_to_word = {index + 3: word for word, index in word_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[0] = "<UNK>"

# word -> index
word_to_index = {word: index for index, word in index_to_word.items()}

model = load_model("RNN_sentiment_analysis_model.h5")
print("Model loaded")

def predict_review(text):
    words = text_to_word_sequence(text)
    cleaned_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    encoded = [word_to_index.get(word, 2) for word in cleaned_words]

    padded = pad_sequences([encoded], maxlen=max_len)

    pred = model.predict(padded)[0][0]

    print(f"Probability of prediction: {pred:.4f}")
    if pred > 0.5:
        print("Positive")
    else: print("Negative")

user_review = input("Enter a movie comment: ")
predict_review(user_review)
    