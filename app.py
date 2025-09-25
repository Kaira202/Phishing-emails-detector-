from flask import Flask, request, jsonify
import joblib
from scipy.sparse import hstack, csr_matrix

# Import helper functions from your train.py
from train import (
    clean_text_for_tfidf,
    extract_url_info,
    extract_basic_features,
    suspicious_counts,
    predict_mail
)

# -------------------- Flask Setup --------------------
app = Flask(__name__)

# -------------------- Load Trained Objects --------------------
model = joblib.load("phishing_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -------------------- Predict Route --------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}

        # Accept either 'text' or 'subject' + 'body'
        if 'text' in data and data['text'].strip():
            subject = ""
            body = data['text']
        else:
            subject = data.get('subject', '')
            body = data.get('body', '')

        if not (subject.strip() or body.strip()):
            return jsonify({'error': "Missing input text"}), 400

        # Use your existing predict_mail function
        prediction, features = predict_mail(
            subject,
            body,
            model=model,
            tfidf=tfidf,
            scaler=scaler,
            feature_columns=feature_columns
        )

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Run Server --------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
