from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config
import joblib
import os
import re
from datetime import datetime

app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = "Please log in to access the application"

# ==================== CONFIGURATION MODE ====================
# 🔧 Maintenant on utilise le VRAI modèle !
USE_SIMULATION = False  # ← Changé à False

# Chemins des modèles
MODEL_PIPELINE_PATH = 'model/sentiment140_model.pkl'
MODEL_CLASSIFIER_PATH = 'model/model_classifier.pkl'
VECTORIZER_PATH = 'model/vectorizer.pkl'

# Variables globales pour le modèle
model_pipeline = None
vectorizer = None
classifier = None

# ==================== FONCTION DE NETTOYAGE (identique à Kaggle) ====================
def clean_tweet(text):
    """
    Nettoie un texte exactement comme dans l'entraînement Kaggle
    Doit être IDENTIQUE à la fonction utilisée pendant l'entraînement !
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Mise en minuscules
    text = text.lower()
    # Suppression des mentions @username
    text = re.sub(r'@\w+', '', text)
    # Suppression des URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Suppression des hashtags (garde le mot sans #)
    text = re.sub(r'#(\w+)', r'\1', text)
    # Suppression des caractères spéciaux (garde lettres et espaces)
    text = re.sub(r'[^a-z\s]', '', text)
    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ==================== CHARGEMENT DU MODÈLE ====================
def load_model():
    """Charge le modèle entraîné sur Kaggle"""
    global model_pipeline, vectorizer, classifier
    
    print("\n" + "=" * 50)
    print("🔧 LOADING MODEL")
    print("=" * 50)
    
    # Option 1: Charger le pipeline complet (recommandé)
    if os.path.exists(MODEL_PIPELINE_PATH):
        try:
            model_data = joblib.load(MODEL_PIPELINE_PATH)
            # Si c'est un dictionnaire avec vectorizer + classifier
            if isinstance(model_data, dict):
                vectorizer = model_data.get('vectorizer')
                classifier = model_data.get('classifier')
                print("✅ Model loaded from sentiment140_model.pkl (dictionary format)")
            else:
                # Si c'est un pipeline scikit-learn
                model_pipeline = model_data
                print("✅ Full pipeline loaded from sentiment140_model.pkl")
            return True
        except Exception as e:
            print(f"⚠️ Pipeline loading error: {e}")
    
    # Option 2: Charger vectorizer et classifier séparément
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_CLASSIFIER_PATH):
        try:
            vectorizer = joblib.load(VECTORIZER_PATH)
            classifier = joblib.load(MODEL_CLASSIFIER_PATH)
            print("✅ Vectorizer and classifier loaded separately")
            return True
        except Exception as e:
            print(f"⚠️ Separate loading error: {e}")
    
    print("❌ No model found!")
    print("   Check that the files are in the 'model/' folder")
    print("   Expected files:")
    print("   - sentiment140_model.pkl")
    print("   - model_classifier.pkl + vectorizer.pkl")
    return False

# ==================== FONCTION DE PRÉDICTION ====================
def predict_sentiment(text):
    """
    Prédiction avec le vrai modèle entraîné sur Sentiment140
    Retourne: (sentiment, confidence)
    """
    global model_pipeline, vectorizer, classifier
    
    # Nettoyer le texte
    cleaned_text = clean_tweet(text)
    
    # Si le texte est vide après nettoyage
    if not cleaned_text:
        return 'neutre', 0.5
    
    # Prédiction selon le type de modèle chargé
    try:
        if model_pipeline is not None:
            # Cas 1: Pipeline complet
            sentiment = model_pipeline.predict([cleaned_text])[0]
            probabilities = model_pipeline.predict_proba([cleaned_text])[0]
            confidence = max(probabilities)
            
        elif vectorizer is not None and classifier is not None:
            # Cas 2: Vectorizer + Classifieur séparés
            text_vectorized = vectorizer.transform([cleaned_text])
            sentiment = classifier.predict(text_vectorized)[0]
            probabilities = classifier.predict_proba(text_vectorized)[0]
            confidence = max(probabilities)
        else:
            # Fallback vers simulation si modèle non chargé
            print("⚠️ Model not available, using simulation")
            return simulate_sentiment_fallback(text)
        
        # S'assurer que le sentiment est dans le bon format
        if sentiment not in ['positif', 'negatif']:
            # Le modèle Sentiment140 est binaire (positif/negatif)
            # Pas de neutre dans ce dataset
            sentiment = 'positif' if sentiment == 'positif' else 'negatif'
        
        return sentiment, confidence
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        # Fallback sécurisé
        return 'neutre', 0.5

def simulate_sentiment_fallback(text):
    """
    Simulation de secours en cas d'erreur du modèle
    """
    text_lower = text.lower()
    
    positive_words = ['love', 'good', 'great', 'amazing', 'excellent', 'happy', 
                      'perfect', 'wonderful', 'fantastic', 'awesome']
    negative_words = ['hate', 'bad', 'terrible', 'awful', 'horrible', 'poor', 
                      'worst', 'disappointed', 'sad', 'angry']
    
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    if pos_count > neg_count:
        return 'positif', 0.7
    elif neg_count > pos_count:
        return 'negatif', 0.7
    else:
        return 'neutre', 0.5

# ==================== CHARGEMENT DU MODÈLE AU DÉMARRAGE ====================
MODEL_LOADED = load_model()

if not MODEL_LOADED:
    print("\n⚠️ WARNING: Simulation mode activated by default")
    USE_SIMULATION = True

# ==================== MODÈLE UTILISATEUR ====================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# ==================== MODÈLE POUR L'HISTORIQUE ====================
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==================== ROUTES ====================
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash(f'Welcome {username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Incorrect username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')
        
        if password != confirm:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(username=username).first():
            flash('This username already exists', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('This email is already in use', 'danger')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'danger')
            return render_template('register.html')
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! You can now log in', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user_predictions = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.created_at.desc()).limit(10).all()
    
    total_predictions = PredictionHistory.query.filter_by(user_id=current_user.id).count()
    positive_count = PredictionHistory.query.filter_by(user_id=current_user.id, sentiment='positif').count()
    negative_count = PredictionHistory.query.filter_by(user_id=current_user.id, sentiment='negatif').count()
    neutral_count = PredictionHistory.query.filter_by(user_id=current_user.id, sentiment='neutre').count()
    
    return render_template('dashboard.html', 
                         user=current_user,
                         predictions=user_predictions,
                         total=total_predictions,
                         positive=positive_count,
                         negative=negative_count,
                         neutral=neutral_count)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        
        if not text:
            flash('Please enter text to analyze', 'warning')
            return redirect(url_for('predict'))
        
        if len(text) < 3:
            flash('Text is too short (minimum 3 characters)', 'warning')
            return redirect(url_for('predict'))
        
        # 🌟 UTILISATION DU VRAI MODÈLE 🌟
        sentiment, confidence = predict_sentiment(text)
        
        # Sauvegarder dans l'historique
        history = PredictionHistory(
            user_id=current_user.id,
            text=text[:500],
            sentiment=sentiment,
            confidence=confidence
        )
        db.session.add(history)
        db.session.commit()
        
        # Mapping des couleurs et icônes
        sentiment_info = {
            'positif': {'icon': 'fa-smile-wink', 'color': 'success'},
            'negatif': {'icon': 'fa-frown', 'color': 'danger'},
            'neutre': {'icon': 'fa-meh', 'color': 'secondary'}
        }
        
        info = sentiment_info.get(sentiment, sentiment_info['neutre'])
        
        return render_template('result.html',
                             text=text,
                             sentiment=sentiment,
                             confidence=confidence,
                             icon=info['icon'],
                             color=info['color'])
    
    return render_template('predict.html')

@app.route('/history')
@login_required
def history():
    all_predictions = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.created_at.desc()).all()
    return render_template('history.html', predictions=all_predictions)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Endpoint API pour les requêtes JSON"""
    if not current_user.is_authenticated:
        return jsonify({'error': 'Authentication required'}), 401
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    sentiment, confidence = predict_sentiment(text)
    
    return jsonify({
        'text': text,
        'sentiment': sentiment,
        'confidence': float(confidence),
        'timestamp': datetime.utcnow().isoformat()
    })

# ==================== INITIALISATION DE LA BASE DE DONNÉES ====================
def init_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='test').first():
            test_user = User(username='test', email='test@example.com')
            test_user.set_password('test123')
            db.session.add(test_user)
            db.session.commit()
            print("✅ Test user created: test / test123")

if __name__ == '__main__':
    init_db()
    
    print("\n" + "=" * 60)
    print("🚀 APPLICATION READY")
    print("=" * 60)
    if MODEL_LOADED:
        print("✅ ML MODEL LOADED (Sentiment140 - 1.6M tweets)")
        print("📊 Type: Binary classification (positive/negative)")
    else:
        print("⚠️ Simulation mode active (model not found)")
    print("\n🌐 Access the application: http://localhost:5000")
    print("👤 Test account: test / test123")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)