from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)


sentiment_pipeline = pipeline("sentiment-analysis")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    result = sentiment_pipeline(text)
    sentiment = result[0]['label']
    score = result[0]['score']
    return render_template('index.html', sentiment=sentiment, score=score, text=text)

if __name__ == '__main__':
    app.run(debug=True)