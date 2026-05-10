FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

# Exposer le port (Render utilise le port 10000 par défaut)
EXPOSE 10000

# Démarrer l'application avec gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]