# RNN Movie Review Sentiment Analysis

This project is a deep learning application that performs **Sentiment Analysis** (Positive/Negative) on movie reviews using **Recurrent Neural Networks (RNN)**. It is trained on the famous IMDB dataset.

---

## Key Features

* **Natural Language Processing:** Custom preprocessing pipeline using `NLTK`.
* **Stop-word Removal:** Filters out common words to focus on meaningful content.
* **Deep Learning:** Built with `TensorFlow/Keras` using `SimpleRNN` layers.
* **Interactive Prediction:** A dedicated script to test your own reviews in real-time.

---

## Project Structure

| File | Description |
| :--- | :--- |
| `train_RNN_model.py` | Preprocesses data, builds and trains the RNN model. |
| `predict_RNN_review.py` | Loads the trained model for real-time user predictions. |
| `RNN_sentiment_analysis_model.h5` | The saved weights and architecture of the trained model. |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
Copy and paste these lines to the terminal sequentially.

git clone "https://github.com/efe-saracalioglu/Sentiment-Analysis-with-RNN"

cd repo-name

### 2. Create and activate a virtual environment

* Create the environment
python -m venv venv

* Activate for Windows:
venv\Scripts\activate

* Activate for macOS/Linux:
source venv/bin/activate

### 3. Install Dependencies

pip install numpy nltk tensorflow matplotlib

## How to Use
### Step 1: Training
To train the model from scratch and see the accuracy/loss plots:

Run command:

python train_RNN_model.py

* This will download the dataset, preprocess the text, and start the training.

* After training, it will display Accuracy and Loss charts.

* The model will be saved as RNN_sentiment_analysis_model.h5

### Step 2: Running Predictions
To test the model with your own custom movie reviews:

Run command:

python predict_RNN_review.py

**Example**: When prompted, type a review like:

"The cinematography was breathtaking and the plot was solid!" -> The script will output the probability and the sentiment **(Positive/Negative)**.

## Technologies Used
* Python 3.11
* TensorFlow / Keras: For building and training the SimpleRNN model.
* NLTK: For natural language preprocessing (stop-words removal).
* Matplotlib: For visualizing training performance.
* NumPy: For data processing
