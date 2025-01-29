import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
last_prediction = {}
last_text = ""

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

@app.route('/', methods=['POST'])
def predict():
    global last_prediction, last_text
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "Text is required."}), 400
    
    last_text = text
    X, y, vectorizer, label_encoder = preprocess_data('data/emotion_final.csv')
    model = train_model(X, y)
    last_prediction = predict_mood_probabilities(text, model, vectorizer, label_encoder)
    
    return jsonify({
        "text": {"user": text},
        "mood": last_prediction
    })

@app.route('/result', methods=['GET'])
def result():
    if last_prediction:
        return jsonify({
            "text": {"user": last_text},
            "mood": last_prediction
        })
    else:
        return jsonify({"error": "No prediction has been made yet."}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7000)