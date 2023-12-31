from flask import Flask, render_template, request
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib
#import pickle
#nltk.download('vader_lexicon')

app = Flask(__name__, static_url_path='/static')
sid = SentimentIntensityAnalyzer()
# with open('logreg.pkl','rb')as file:
#     #model = pickle.load('logreg.pkl')
#     x = pickle.Unpickler(file)
#     #s = x.load()
#     print(x)

# model = joblib.load('path_to_your_joblib_model_file') #replace with model location 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        review = request.form['review']
        scores = sid.polarity_scores(review)
        sentiment = get_sentiment(scores)
        return render_template('result.html', review=review, sentiment=sentiment)

def get_sentiment(scores):
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    # else:
    #     return 'Neutral'

if __name__ == '__main__':
    app.run(debug=True)
