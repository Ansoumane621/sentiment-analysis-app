import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Créer le dossier model s'il n'existe pas
os.makedirs('model', exist_ok=True)

print("=" * 50)
print("ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION DE SENTIMENTS")
print("=" * 50)

# Option 1: Utiliser un dataset existant (exemple avec des données factices)
# Option 2: Créer un petit dataset d'exemple pour tester

# Création d'un dataset d'exemple (à remplacer par votre vrai dataset)
data = {
    'text': [
        # Sentiments positifs
        "I love this product, it's amazing!",
        "Great service, very happy with everything",
        "Excellent quality, I recommend",
        "Superb, fantastic experience",
        "Very satisfied, will buy again",
        "J'adore ce produit, il est génial",
        "Service excellent, très content",
        "Qualité exceptionnelle, je recommande",
        "Superbe expérience fantastique",
        "Très satisfait, racheterai",
        # Sentiments négatifs
        "Terrible, worst purchase ever",
        "Very disappointed with this service",
        "Poor quality, do not buy",
        "Awful experience, regret it",
        "Bad customer service, never again",
        "Horrible, pire achat de ma vie",
        "Très déçu par ce service",
        "Mauvaise qualité, n'achetez pas",
        "Expérience affreuse, je regrette",
        "Service client nul, plus jamais",
        # Sentiments neutres
        "It's okay, nothing special",
        "Average product, works as expected",
        "Not bad, not great either",
        "Standard service, acceptable",
        "C'est correct, rien de spécial",
        "Produit moyen, fonctionne comme prévu",
        "Pas mal, pas génial non plus",
        "Service standard, acceptable",
    ],
    'sentiment': [
        'positif', 'positif', 'positif', 'positif', 'positif',
        'positif', 'positif', 'positif', 'positif', 'positif',
        'negatif', 'negatif', 'negatif', 'negatif', 'negatif',
        'negatif', 'negatif', 'negatif', 'negatif', 'negatif',
        'neutre', 'neutre', 'neutre', 'neutre', 'neutre',
        'neutre', 'neutre', 'neutre'
    ]
}

df = pd.DataFrame(data)
print(f"\n📊 Dataset chargé : {len(df)} commentaires")
print(f"   - Positifs : {sum(df['sentiment'] == 'positif')}")
print(f"   - Négatifs : {sum(df['sentiment'] == 'negatif')}")
print(f"   - Neutres  : {sum(df['sentiment'] == 'neutre')}")

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['sentiment']
)

print(f"\n📊 Split des données:")
print(f"   - Train : {len(X_train)} échantillons")
print(f"   - Test  : {len(X_test)} échantillons")

# Création du pipeline TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )),
    ('classifier', LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    ))
])

print("\n🔄 Entraînement du modèle...")
pipeline.fit(X_train, y_train)

# Évaluation
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Précision du modèle : {accuracy:.2%}")
print("\n📋 Rapport de classification:")
print(classification_report(y_test, y_pred))

# Sauvegarde du modèle et du vectorizer
joblib.dump(pipeline, 'model/sentiment_model.pkl')
print("\n💾 Modèle sauvegardé dans 'model/sentiment_model.pkl'")

# Test du modèle avec quelques exemples
print("\n" + "=" * 50)
print("TEST DU MODÈLE AVEC EXEMPLES")
print("=" * 50)

test_phrases = [
    "This is the best thing ever! I love it!",
    "I hate this, it's terrible and useless.",
    "It's fine, nothing special.",
    "Magnifique ! Je suis ravi de cet achat.",
    "Décevant, je ne recommande pas du tout.",
    "C'est correct sans plus."
]

for phrase in test_phrases:
    pred = pipeline.predict([phrase])[0]
    proba = pipeline.predict_proba([phrase]).max()
    print(f"\n📝 '{phrase}'")
    print(f"   → Sentiment : {pred} (confiance: {proba:.1%})")

print("\n✨ Entraînement terminé ! Vous pouvez lancer l'app Flask.")