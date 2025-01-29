import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['Text'] = df['Text'].str.lower()
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Text'])
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Emotion'])
    
    return X, y, vectorizer, label_encoder

def train_model(X, y):
    model = MultinomialNB()
    model.fit(X, y)
    return model

def predict_mood_probabilities(text, model, vectorizer, label_encoder):
    text = text.lower()
    vectorized_text = vectorizer.transform([text])
    probabilities = model.predict_proba(vectorized_text)[0]
    emotions = label_encoder.classes_
    result = {emotion: prob * 100 for emotion, prob in zip(emotions, probabilities)}
    return result

X, y, vectorizer, label_encoder = preprocess_data('data/emotion_final.csv')
model = train_model(X, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    text = ""
    if request.method == 'POST':
        text = request.form.get('text', '')
        if text:
            prediction = predict_mood_probabilities(text, model, vectorizer, label_encoder)
    return render_template('index.html', text=text, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7000)
