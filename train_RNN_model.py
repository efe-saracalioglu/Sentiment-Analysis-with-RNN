import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

max_features = 10000
max_len = 500

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = max_features)

original_word_index = imdb.get_word_index()

inv_word_index = {index + 3: word for word, index in original_word_index.items()}
inv_word_index[0] = "<PAD>"
inv_word_index[1] = "<START>"
inv_word_index[2] = "<UNK>"

def decode_review(encoded_review):
    return " ".join([inv_word_index.get(i, "?") for i in encoded_review])

movie_index = 0

print("First comment: (number array)")
print(X_train[movie_index])

print("First comment (with words):")
print(decode_review(X_train[movie_index]))

print(f"Label: {'Positive' if y_train[movie_index]== 1 else 'Negative'}")

# word to index, then index to word
word_index = imdb.get_word_index()
index_to_word = {index + 3: word for word, index in word_index.items()} #from words to index
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"
word_to_index = {word: index for index, word in index_to_word.items()} #from index to words

# data preprocessing
def preprocess_review(encoded_review):
    #index to words
    words = [index_to_word.get(i,"") for i in encoded_review if i >= 3]
    #take words that is not stopwords
    cleaned = [
        word.lower() for word in words if word.isalpha() and word.lower() not in stop_words
    ]

    #cleaned words into index
    return [word_to_index.get(word, 2) for word in cleaned]

X_train = [preprocess_review(review) for review in X_train]
X_test = [preprocess_review(review) for review in X_test]

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# modeling
model = Sequential()

model.add(Embedding(input_dim = max_features, output_dim = 32, input_length = max_len))
model.add(SimpleRNN(32))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

history = model.fit(
    X_train, y_train,
    epochs = 2,
    batch_size = 64,
    validation_split = 0.2
)

def plot_history(history):
    plt.subplot(1,2,1)

    # accuracy plot
    plt.plot(history.history["accuracy"], label="Training")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # loss plot
    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

plot_history(history)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

model.save("RNN_sentiment_analysis_model.h5")
print("Model is saved: RNN_sentiment_analysis_model.h5")