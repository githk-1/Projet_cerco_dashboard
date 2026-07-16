# FROM python:3.12-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY app/ ./app
# EXPOSE 8501
# CMD ["streamlit", "run", "app/Acceuil.py", "--server.port=8501", "--server.address=0.0.0.0"]

#############################################################################################################

# FROM python:3.12

# WORKDIR /app

# # Copier les fichiers de dépendances et les installer
# COPY requirements.txt .

# # RUN apt-get update && apt-get install -y ca-certificates openssl

# # RUN update-ca-certificates

# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir  --default-timeout=1000 --index-url https://pypi.org/simple -r requirements.txt
# #RUN pip install --no-cache-dir -r requirements.txt files.pythonhosted.org

# # --trusted-host pypi.org --trusted-host

# # Copier le reste du projet
# COPY . .

# # Exposer le port Streamlit
# EXPOSE 8501

# # Définir la commande pour démarrer l'application
# CMD ["streamlit", "run", "app/Acceuil.py", "--server.address=0.0.0.0"]


FROM python:3.12-slim

#ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS="0.0.0.0"
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ENABLE_CORS=false

WORKDIR /app

# Copier les fichiers de dépendances et les installer
COPY requirements.txt .

RUN apt-get update && apt-get install -y libgomp1
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org --default-timeout=1000 -r requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet
COPY . .

# Exposer le port Streamlit
EXPOSE 8080

# Définir la commande pour démarrer l'application
CMD ["streamlit", "run", "app/Accueil.py", "--server.port=8080", "--server.address=0.0.0.0"]



# FROM python:3.12-slim
 
# # Installer les certificats et utilitaires
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ca-certificates curl \
# && update-ca-certificates \
# && rm -rf /var/lib/apt/lists/*
 
# # Upgrade pip + certifi (bundle CA Python)
# RUN pip install --upgrade pip certifi
 
# # Forcer Python à utiliser le bundle certifi
# ENV SSL_CERT_FILE=/usr/local/lib/python3.12/site-packages/certifi/cacert.pem
 
# # Dossier de travail
# WORKDIR /app
 
# # Copier requirements
# COPY requirements.txt .
 
# # Installer dépendances
# RUN pip install --no-cache-dir -r requirements.txt
 
# # Copier le projet
# COPY . .
 
# # Commande de lancement (à adapter selon ton app)
# CMD ["python", "main.py"]


# FROM python:3.12-slim

# LABEL description="Projet Cerco - Application Streamlit"
# LABEL version="1.0.0"

# # Variables d'environnement
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV STREAMLIT_SERVER_PORT=8501
# ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
# ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
# ENV STREAMLIT_SERVER_ENABLE_CORS=false

# # Dépendances système (adapte si besoin)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     g++ \
#     python3-dev \
#     libpq-dev \
#     curl \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

# # Création d'un utilisateur non-root
# RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# # Création explicite de l'arborescence de ton projet
# RUN mkdir -p /app/app/pages \
#     && mkdir -p /app/data/data_brute \
#     && mkdir -p /app/data/data_traite \
#     && mkdir -p /app/data/models_data \
#     && mkdir -p /app/fichier_models \
#     && mkdir -p /app/models \
#     && mkdir -p /app/orchestration \
#     && mkdir -p /app/pretraitement_bases \
#     && mkdir -p /app/ressources \
#     && mkdir -p /app/src \
#     && mkdir -p /home/appuser/.streamlit

# # Positionne le dossier de travail
# WORKDIR /app

# # Copie et installation des dépendances Python
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# # Copie du reste du projet (structure respectée)
# COPY --chown=appuser:appuser app/ ./app/
# COPY --chown=appuser:appuser data/ ./data/
# COPY --chown=appuser:appuser fichier_models/ ./fichier_models/
# COPY --chown=appuser:appuser models/ ./models/
# COPY --chown=appuser:appuser orchestration/ ./orchestration/
# COPY --chown=appuser:appuser pretraitement_bases/ ./pretraitement_bases/
# COPY --chown=appuser:appuser src/ ./src/

# # Config Streamlit (personnalisable)
# #RUN printf '[server]\nport = 8501\naddress = "0.0.0.0"\nenableCORS = false\nmaxUploadSize = 200\n\n[browser]\ngatherUsageStats = false\n\n' > /home/appuser/.streamlit/config.toml

# # Permissions
# # RUN chown -R appuser:appuser /app /home/appuser && \
# #     chmod -R 755 /app && \
# #     chmod 700 /home/appuser/.streamlit && \
# #     chmod 644 /home/appuser/.streamlit/config.toml

# USER appuser

# EXPOSE 8501

# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#     CMD curl -f http://localhost:8501/_stcore/health || exit 1

# CMD ["streamlit", "run", "app/Acceuil.py", "--server.port=8501", "--server.address=0.0.0.0"]