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

    # Improved heuristic spam detection (keyword + pattern based)
    def detect_spam(text, pred_label, conf):
        t = text.lower()
        
        # If model directly predicts Spam, trust it
        if isinstance(pred_label, str) and pred_label.lower() == 'spam':
            return True
        
        # Spam indicator keywords (weighted)
        high_spam_keywords = [
            'winner', 'won', 'claim', 'prize', 'lottery', 'congratulations',
            'verify account', 'confirm identity', 'update payment', 'click here',
            'act now', 'limited time', 'urgent action', 'credit card number',
            'bank account', 'password', 'dear friend', 'inheritance', 'millions'
        ]
        
        medium_spam_keywords = [
            'win', 'free', 'offer', 'buy now', 'subscribe', 'unsubscribe',
            'click', 'link', 'urgent', 'help', 'reset password'
        ]
        
        # Pattern checks
        has_url = bool(re.search(r'https?://|www\.|\.com|\.net', t))
        has_multiple_urls = len(re.findall(r'https?://|www\.|\.com', t)) >= 2
        exclaim_count = t.count('!')
        many_exclaims = exclaim_count >= 3
        multiple_caps_words = sum(1 for word in text.split() if word.isupper() and len(word) > 2)
        has_all_caps_section = bool(re.search(r'[A-Z]{5,}', text))
        
        # Calculate spam score
        spam_score = 0
        
        # High-confidence spam keywords: +2 points each
        high_kw_matches = sum(1 for kw in high_spam_keywords if kw in t)
        spam_score += high_kw_matches * 2
        
        # Medium-confidence spam keywords: +1 point each
        medium_kw_matches = sum(1 for kw in medium_spam_keywords if kw in t)
        spam_score += medium_kw_matches * 1
        
        # Pattern scoring
        if has_multiple_urls:
            spam_score += 3
        elif has_url:
            spam_score += 1
        
        if has_all_caps_section:
            spam_score += 2
        elif multiple_caps_words >= 2:
            spam_score += 1
        
        if many_exclaims:
            spam_score += 2
        elif exclaim_count >= 2:
            spam_score += 1
        
        # Low model confidence + spam indicators = likely spam
        if conf < 0.5 and (high_kw_matches >= 1 or has_url):
            spam_score += 1
        
        # Text length: extremely short + high spam score = likely spam
        if len(text.split()) < 15 and spam_score >= 3:
            spam_score += 1
        
        # Threshold: score >= 3 is spam
        return spam_score >= 3

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
