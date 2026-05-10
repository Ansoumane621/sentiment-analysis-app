"""
Test du modèle chargé localement
"""
import joblib
import re

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Chargement
print("🔧 Chargement du modèle...")
model_data = joblib.load('model/sentiment140_model.pkl')
vectorizer = model_data['vectorizer']
classifier = model_data['classifier']
print("✅ Modèle chargé\n")

# Tests
test_phrases = [
    "I love this product! It's amazing!",
    "This is terrible, worst ever",
    "Just okay, nothing special",
    "Very happy with my purchase",
    "Disappointed, waste of money",
    "Je n'aime pas du tout ce produit",  # French
]

print("=" * 50)
print("🧪 TEST DU MODÈLE")
print("=" * 50)

for phrase in test_phrases:
    cleaned = clean_tweet(phrase)
    vec = vectorizer.transform([cleaned])
    pred = classifier.predict(vec)[0]
    proba = classifier.predict_proba(vec)[0]
    confidence = max(proba)
    
    emoji = "😊" if pred == "positif" else "😞"
    print(f"\n📝 \"{phrase}\"")
    print(f"   → {emoji} {pred} (confiance: {confidence:.1%})")