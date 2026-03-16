# Projet Cercos

## Description
Ce projet vise à modéliser les tendances des indicateurs de la maladie de la cercosporiose noire. Il comprend plusieurs éléments clés :

- Un pipeline complet pour la récupération et le traitement automatique des données.
- Un pipeline de modélisation complet pour analyser et prédire les tendances des indicateurs.
- Une application interactive et intuitive permettant de visualiser l'ensemble des résultats et du travail réalisé de manière simple et efficace.

⚠️ Accès aux données
Ce projet a été réalisé dans le cadre d'un stage en entreprise. Pour des raisons de confidentialité, les données utilisées ne sont pas disponibles dans ce dépôt public. Le code source est mis à disposition à titre de démonstration afin d'illustrer les choix techniques et l'architecture de l'application.
Sans les données, l'application ne peut pas être lancée localement. Des aperçus visuels de l'interface sont disponibles ci-dessous.

## Fonctionnalités
- [Liste des principales fonctionnalités]
- Interface utilisateur avec Streamlit
- Modèles de prédiction pour différentes zones géographiques
- Prétraitement des données
- Orchestration des tâches

## Installation
### Prérequis
- Python 3.12
- Docker (optionnel pour le déploiement)

### Installation locale
1. Cloner le repository :
   ```bash
   git clone https://github.com/githk-1/Projet_cercos.git
   cd projet_cerco
   ```

2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Lancer l'application :
   ```bash
   streamlit run app/Acceuil.py
   ```

### Avec Docker
1. Construire l'image :
   ```bash
   docker build -t projet_cerco .
   ```

2. Lancer le conteneur :
   ```bash
   docker run -p 8501:8501 projet_cerco
   ```

## Utilisation

### Étape 1 : Prétraitement des données
Le prétraitement des données est une étape essentielle pour mettre les bases sous la bonne forme avant la modélisation. Voici comment procéder :

- Le dossier `pretraitement_bases/` contient des fichiers Python (`.py`) pour le traitement individuel de chaque base de données. Vous pouvez exécuter ces fichiers un par un pour préparer les données.
- Si vous souhaitez traiter toutes les bases en une seule fois, exécutez le fichier `traitement_bases.py` dans le même dossier. Cela appliquera le traitement à toutes les bases automatiquement.
- Les données prétraitées sont enregistrées dans le dossier `data/data_traite/`.

### Étape 2 : Modélisation
La modélisation suit un ordre précis et repose sur les fichiers situés dans le dossier `fichier_models/`.

1. **Optimisation des modèles** :
   - Lancez le fichier `train.py` pour démarrer l'optimisation des modèles de prévision.
   - Une fois l'optimisation terminée, un dossier `models/` est créé à la racine du projet. Ce dossier contient plusieurs sous-dossiers organisés par indicateur et zone (par exemple, `models_indicateur_Zone`).
   - Chaque sous-dossier contient les fichiers suivants :
     - `best_params.json` : Les meilleurs paramètres du modèle.
     - `intervals.pkl` : Les intervalles de confiance.
     - `models.pkl` : Le modèle entraîné.
     - `ts.pkl` : Les séries temporelles utilisées.

2. **Test des modèles** :
   - Exécutez le fichier `test.py` pour charger les modèles optimisés et lancer les prédictions.
   - Les prédictions sont enregistrées dans le dossier `data/models_data/`, organisé en sous-dossiers par indicateur et zone (par exemple, `models_Indicateur_Zone`).
   - Chaque sous-dossier contient :
     - Les prédictions sur les données d'entraînement.
     - Les prédictions sur les données de test.
     - L'importance des variables pour les prédictions de test.

3. **Prédictions futures** :
   - Lancez le fichier `model_futur.py` pour prédire les valeurs des séries non observées.
   - Les exogènes sont imputés, puis utilisés pour générer les prédictions futures.
   - Les résultats sont enregistrés dans le dossier `data/models_data/` sous forme de fichiers CSV, avec les noms suivants :
     - `predictions_train_futur`
     - `predictions_futur`
     - `importance_futur`

### Étape 3 : Utilisation de l'application
Une fois les données prétraitées et les modèles optimisés, vous pouvez lancer l'application pour visualiser les résultats et les analyses :

- Exécutez la commande suivante pour démarrer l'application Streamlit :
  ```bash
  streamlit run app/Acceuil.py
  ```
- L'application permet de visualiser :
  - Les données brutes et prétraitées avec la possibilité de télécharger la base prête.
  - Une exploration détaillée de chacune des bases de données utilisées dans le projet.
  - Une transparence totale sur les données, permettant de mieux comprendre leur structure et leur contenu.
  - Les résultats des prédictions.
  - Les analyses des modèles et des variables importantes.
  

Cette interface interactive simplifie la prise de décision en regroupant toutes les informations nécessaires en un seul endroit.

### Étape 4 : Orchestration complète
Pour automatiser l'ensemble des étapes précédentes, vous pouvez utiliser le fichier `orchestration.py` situé dans le dossier `orchestration/`.

- Ce fichier récupère automatiquement les données depuis un dossier de dépôt, permettant ainsi d'actualiser les données utilisées.
- Il exécute ensuite toutes les étapes du projet, depuis le prétraitement des données jusqu'à la modélisation et les prédictions futures.
- Ce fichier est idéal pour relancer l'ensemble du pipeline de modélisation de manière automatisée et cohérente.

## Architecture du Projet
```
projet_cerco/
├── Dockerfile
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── Acceuil.py
│   └── pages/
├── data/
│   ├── data_brute/
│   ├── data_traite/
│   └── models_data/
│   
├── fichier_models/
│   ├── model_futur.py
│   ├── test.py
│   └── train.py
├── models/
│   ├── models_Dp_moy_Bananord/
│   ├── models_Dp_moy_Banasud/
│   ├── models_Dp_moy_Brimbo Zone 1/
│   └── ... (plusieurs sous-dossiers de modèles)
├── orchestration/
│   ├── BDD.py/
│   └──  orchestration.py/
├── pretraitement_bases/
│   ├── base_cerco.py/
│   ├── base_intrant.py/
│   ├── base_meteo.py/
│   ├── merge.py/
│   └── traitement_bases.py
└── src/
```

## Détails des Dossiers
- **app/** : Contient l'application principale Streamlit, avec le fichier d'accueil `Acceuil.py` et les pages supplémentaires.
- **data/** : Dossiers pour les données brutes, traitées et les données relatives aux résultats de nos modèles.
- **fichier_models/** : Scripts pour l'optimisation, le test et les modèles futurs (`train.py`, `test.py`, `model_futur.py`).
- **models/** : Modèles entraînés et optimisés pour différentes zones géographiques et indicateurs (Dp_moy, Etat_devolution_moy, etc.).
- **orchestration/** : Scripts pour lancement de l'ensemble des tâches,  en allant de la récuparation des données pour le prétraitement aux enregistrements des résultats de nos modélisations.
- **pretraitement_bases/** : Code pour le prétraitement des bases de données.
- **src/** : Ce dossier regroupe toutes les fonctions personnalisées développées pour le projet. Ces fonctions sont utilisées dans les différents fichiers Python pour :
  - Le prétraitement des données.
  - La modélisation et l'optimisation des modèles.
  - L'application Streamlit.
  - Les visualisations et analyses des données.
  
  En somme, ce dossier contient l'ensemble des outils nécessaires pour exécuter les pipelines et les étapes clés du projet.

## Technologies Utilisées
- Python
- Pandas
- Numpy
- Streamlit
- Docker
- Mlforecast
- Optuna
- Scikit-learn 