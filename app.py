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

    # Advanced heuristic spam detection with better accuracy
    def detect_spam(text, pred_label, conf):
        t = text.lower()
        
        # If model directly predicts Spam, trust it (100% confidence)
        if isinstance(pred_label, str) and pred_label.lower() == 'spam':
            return True
        
        # ===== SPAM KEYWORDS =====
        # High-confidence spam keywords (very strong indicators)
        high_spam_keywords = [
            'winner', 'won', 'claim', 'prize', 'lottery', 'congratulations',
            'verify account', 'confirm account', 'confirm identity', 'update payment',
            'verify payment', 'click here', 'click now', 'act now', 'limited time',
            'urgent action needed', 'credit card', 'bank account', 'password reset',
            'reset password', 'dear friend', 'inheritance', 'millions', 'reward',
            'phishing', 'malware', 'suspicious', 'confirm credentials'
        ]
        
        # Medium-confidence spam keywords
        medium_spam_keywords = [
            'win', 'free', 'offer', 'buy now', 'subscribe', 'unsubscribe',
            'click', 'link', 'urgent', 'call now', 'act now', 'asap',
            'no cost', 'no fees', 'guaranteed', 'exclusive', 'special offer'
        ]
        
        # ===== PATTERN DETECTION =====
        # URL patterns
        has_url = bool(re.search(r'https?://|www\.|\.com|\.net|\.info|\.biz', t))
        has_multiple_urls = len(re.findall(r'https?://|www\.|\.com', t)) >= 2
        has_shortened_url = bool(re.search(r'bit\.ly|tinyurl|short\.link', t))
        
        # Punctuation patterns
        exclaim_count = t.count('!')
        question_count = t.count('?')
        many_exclaims = exclaim_count >= 3
        
        # Capitalization patterns
        caps_words = sum(1 for word in text.split() if word.isupper() and len(word) > 2)
        has_consecutive_caps = bool(re.search(r'[A-Z]{5,}', text))
        
        # Email length (very short emails are often spam)
        word_count = len(text.split())
        is_very_short = word_count < 10
        
        # ===== SCORING ALGORITHM =====
        spam_score = 0
        
        # 1. High-priority keyword matching
        high_kw_count = sum(1 for kw in high_spam_keywords if kw in t)
        spam_score += high_kw_count * 3  # Increased from 2 to 3
        
        # 2. Medium-priority keyword matching
        medium_kw_count = sum(1 for kw in medium_spam_keywords if kw in t)
        spam_score += medium_kw_count * 1
        
        # 3. URL patterns (strong spam indicator)
        if has_multiple_urls:
            spam_score += 4  # Increased from 3
        elif has_shortened_url:
            spam_score += 3
        elif has_url and high_kw_count > 0:
            spam_score += 2
        elif has_url:
            spam_score += 1
        
        # 4. Capitalization abuse
        if has_consecutive_caps:
            spam_score += 3  # Increased from 2
        elif caps_words >= 3:
            spam_score += 2
        elif caps_words >= 1:
            spam_score += 1
        
        # 5. Excessive punctuation
        if many_exclaims:
            spam_score += 3  # Increased from 2
        elif exclaim_count >= 2:
            spam_score += 1
        
        if question_count >= 3:
            spam_score += 1
        
        # 6. Confidence + keywords (model uncertainty with spam signals)
        if conf < 0.4:  # Lowered threshold from 0.5
            if high_kw_count >= 1:
                spam_score += 2
            elif has_url:
                spam_score += 1
        
        # 7. Very short text with spam indicators (phishing SMS-like emails)
        if is_very_short and (high_kw_count >= 1 or many_exclaims or has_url):
            spam_score += 2
        
        # 8. Combination of patterns (very strong indicator)
        pattern_count = sum([
            has_multiple_urls,
            has_consecutive_caps,
            many_exclaims,
            high_kw_count >= 1
        ])
        if pattern_count >= 2:
            spam_score += 2  # Bonus for multiple pattern matches
        
        # ===== THRESHOLD LOGIC =====
        # Adjusted threshold for better detection
        # Score >= 4: Definitely spam
        # Score 2-3: Likely spam (depends on pattern combination)
        if spam_score >= 4:
            return True
        
        if spam_score >= 2 and (high_kw_count >= 1 or has_multiple_urls or has_consecutive_caps):
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
