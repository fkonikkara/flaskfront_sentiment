from flask import Flask, render_template, request
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib

nltk.download('vader_lexicon')

app = Flask(__name__)

# Load your trained model
model = joblib.load('path_to_your_model_file')  # Replace with your model's file path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        review = request.form['review']
        # Use your trained model for prediction
        prediction = model.predict([review])  # Replace with your model's prediction method
        # Process prediction and return result to template
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
