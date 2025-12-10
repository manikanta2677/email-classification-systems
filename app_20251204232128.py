from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re

app = Flask(__name__)
CORS(app)   # 🔥 VERY IMPORTANT

model = pickle.load(open("email_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    email_text = data["email"]

    vect = vectorizer.transform([email_text])
    pred = model.predict(vect)[0]
    confidence = float(max(model.predict_proba(vect)[0]))

    # Simple heuristic spam detection (keyword + pattern based)
    def detect_spam(text, pred_label, conf):
        t = text.lower()
        spam_keywords = [
            'win', 'winner', 'prize', 'lottery', 'free', 'click', 'click here', 'subscribe',
            'buy now', 'limited time', 'act now', 'congratulations', 'offer', 'urgent',
            'password', 'credit card', 'bank', 'account', 'dear friend'
        ]

        kw_matches = sum(1 for kw in spam_keywords if kw in t)
        has_url = bool(re.search(r'https?://|www\.', t))
        many_exclaims = t.count('!') >= 2
        long_all_caps = any(word.isupper() and len(word) > 3 for word in text.split())
        low_confidence = conf < 0.5

        # If model directly predicts a spam label, trust it
        if isinstance(pred_label, str) and 'spam' in pred_label.lower():
            return True

        # Heuristic rules
        if kw_matches >= 2:
            return True
        if kw_matches >= 1 and (has_url or many_exclaims or long_all_caps or low_confidence):
            return True

        return False

    is_spam = detect_spam(email_text, pred, confidence)

    # If heuristic marks as spam, present category as 'Spam' like other labels
    if is_spam:
        pred = 'Spam'

    return jsonify({
        "success": True,
        "category": pred,
        "confidence": confidence,
        "is_spam": is_spam
    })

if __name__ == "__main__":
    app.run(debug=True)
