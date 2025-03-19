import joblib

model = joblib.load("logistic_regression_model/lr_ai_text_generated_detection.pkl")
vectorizer = joblib.load("logistic_regression_model/tfidf_vectorizer.pkl")

def predict_ai_generated(text):
    text_tfidf = vectorizer.transform([text])
    probability = model.predict_proba(text_tfidf)[:, 1][0]
    return probability
