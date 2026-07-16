import pandas as pd
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")
import missingno as msno 
import sidetable
import missingno as msno

# Pour les modèles
import random
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, ExpandingMean, ExponentiallyWeightedMean, RollingQuantile,ExpandingStd,SeasonalRollingMean, RollingStd, RollingMax,RollingMin
from mlforecast.target_transforms import Differences, AutoSeasonalityAndDifferences
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from mlforecast.utils import PredictionIntervals
import os
import contextlib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import streamlit as st

from tqdm.notebook import tqdm
from datetime import datetime, timedelta
import importlib
import pkg_resources
import joblib
import json
import pickle 
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from typing import List
import optuna
from typing import Dict, Any
from mlforecast.lag_transforms import (
    ExponentiallyWeightedMean, 
    RollingMean, 
    RollingStd,
    RollingMax,      
    RollingMin,      
    ExpandingMean
)
from mlforecast.target_transforms import Differences
from mlforecast.feature_engineering import transform_exog



def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE) compatible avec arrays numpy, pandas, listes...
    Ignore les divisions par zéro et gère les valeurs nulles.
    Retourne le score en pourcentage.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = (np.abs(y_true) > 1e-8) & ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.clip(np.abs(y_true[mask]), 1e-8, None))) * 100

class PatienceEarlyStopper:
    """Callback Optuna pour early stopping avec patience (comme EarlyStoppingCallback natif)."""
    def __init__(self, patience):
        self.patience = patience
        self.best_value = None
        self.num_bad_trials = 0

    def __call__(self, study, trial):
        # On considère que l'objectif est minimisé (comme 'direction=minimize')
        current_value = trial.value
        if self.best_value is None or current_value < self.best_value:
            self.best_value = current_value
            self.num_bad_trials = 0
        else:
            self.num_bad_trials += 1
        if self.num_bad_trials >= self.patience:
            print(f"[EarlyStopping] Arrêt anticipé : aucune amélioration en {self.patience} essais consécutifs.")
            study.stop()

def get_optimization_config(model_name, trial):
    """
    Propose des hyperparamètres pour le modèle donné, à utiliser avec Optuna.
    Retourne un dictionnaire de kwargs à passer au constructeur du modèle.
    """
    if model_name == 'xgb':
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 50, 150),
            max_depth=trial.suggest_int('max_depth', 3, 7),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.15),
            reg_alpha=trial.suggest_float('reg_alpha', 0, 2),
            reg_lambda=trial.suggest_float('reg_lambda', 0, 2),
            random_state=42,
            verbosity=0
        )
    elif model_name == 'lgbm':
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 50, 150),
            max_depth=trial.suggest_int('max_depth', 3, 7),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.15),
            reg_alpha=trial.suggest_float('reg_alpha', 0, 2),
            reg_lambda=trial.suggest_float('reg_lambda', 0, 2),
            random_state=42,
            verbosity=-1
        )
    elif model_name == 'cat':
        return dict(
            iterations=trial.suggest_int('iterations', 50, 150),
            depth=trial.suggest_int('depth', 3, 7),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.15),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 10),
            bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 0.2),
            random_state=42,
            verbose=0
        )
    else:
        raise ValueError(f"Modèle inconnu: {model_name}")
    
def select_best_model_by_vote(predictions_dict, y_true, verbose=True):
    """
    Sélectionne le meilleur modèle par système de vote sur 3 métriques : RMSE, MAE, MAPE.
    Chaque métrique "vote" pour son meilleur modèle, le modèle avec le plus de votes gagne.
    Si le MAPE est NaN (div par zéro), le vote ne se fait que sur RMSE et MAE.
    Si égalité parfaite (1 vote chacun), on prend le modèle qui a la somme des 2 métriques la plus basse.
    """
    models = list(predictions_dict.keys())
    if len(models) == 0:
        print("Aucun modèle fourni")
        return None, None

    if verbose:
        print(f"Système de vote sur {len(models)} modèles avec 3 métriques...")

    model_metrics = {}
    for model_name, y_pred in predictions_dict.items():
        try:
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = np.array(y_true)[mask]
            y_pred_clean = np.array(y_pred)[mask]

            if len(y_true_clean) == 0:
                print(f"Modèle {model_name}: Aucune donnée valide")
                continue

            rmse = mean_squared_error(y_true_clean, y_pred_clean, squared=False)
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            mape_score = mape(y_true_clean, y_pred_clean)

            model_metrics[model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape_score
            }

            if verbose:
                print(f"{model_name}: RMSE={rmse:.3f}, MAE={mae:.3f}, MAPE={mape_score:.2f}")

        except Exception as e:
            print(f"Erreur pour {model_name}: {e}")
            continue

    if len(model_metrics) == 0:
        print("Aucun modèle n'a pu être évalué")
        return None, None

    # Détecter si MAPE est NaN pour tous les modèles (div par zéro)
    mape_is_nan = all(np.isnan(metrics['MAPE']) for metrics in model_metrics.values())

    # Si le MAPE est NaN pour tous les modèles ou partiellement, ignorer le vote MAPE
    metrics_to_vote = ['RMSE', 'MAE']
    if not mape_is_nan and not all(np.isnan(model_metrics[m]['MAPE']) for m in model_metrics):
        metrics_to_vote.append('MAPE')

    votes = {model: 0 for model in model_metrics.keys()}
    winners_by_metric = {}
    for metric in metrics_to_vote:
        try:
            # On ignore les modèles dont la métrique est NaN
            valid_models = [m for m in model_metrics if not np.isnan(model_metrics[m][metric])]
            if not valid_models:
                continue
            best_model_for_metric = min(valid_models, key=lambda m: model_metrics[m][metric])
            votes[best_model_for_metric] += 1
            winners_by_metric[metric] = best_model_for_metric

            if verbose:
                best_value = model_metrics[best_model_for_metric][metric]
                print(f"{metric}: {best_model_for_metric} ({best_value:.3f})")

        except Exception as e:
            print(f"Erreur pour métrique {metric}: {e}")
            continue

    # On cherche le(s) modèle(s) avec le max de votes
    max_votes = max(votes.values())
    best_models = [m for m, v in votes.items() if v == max_votes]

    if len(best_models) == 1:
        best_model = best_models[0]
    else:
        # Cas d'égalité pour 2 modèles avec 1 vote chacun sur 2 métriques (si MAPE ignoré)
        if len(metrics_to_vote) == 2 and len(best_models) == 2:
            # On fait la somme des 2 métriques (RMSE + MAE) et on prend le plus bas
            sums = {m: model_metrics[m]['RMSE'] + model_metrics[m]['MAE'] for m in best_models}
            best_model = min(sums, key=sums.get)
            if verbose:
                print(f"ÉGALITÉ: sélection par somme RMSE+MAE :")
                for m, s in sums.items():
                    print(f"   {m}: somme={s:.3f}")
        else:
            # En cas d'égalité sur 3 métriques ou plus de 2 modèles, on garde le premier arbitrairement
            best_model = best_models[0]

    if verbose:
        print(f"\nRESULTATS DU VOTE:")
        for model, vote_count in sorted(votes.items(), key=lambda x: x[1], reverse=True):
            print(f"   {model}: {vote_count}/{len(metrics_to_vote)} votes")
        print(f"\nGAGNANT: {best_model} avec {votes[best_model]}/{len(metrics_to_vote)} votes")

    best_model_metrics = model_metrics[best_model]
    return best_model, best_model_metrics



def preprocess_exog_and_target(df_train, df_test, exog_cols):
    """
    Prétraite les variables exogènes et la cible pour l'entraînement et le test.
    
    Cette fonction réalise les étapes suivantes :
        - Détecte les colonnes numériques et catégorielles parmi les exogènes.
        - Applique un encodage OneHot sur les variables catégorielles (avec gestion des catégories inconnues).
        - Met à jour la liste des colonnes exogènes après encodage.
        - Standardise la variable cible 'y' (fit sur train, transform sur test).
        - Standardise les variables exogènes numériques (hors colonnes temporelles 'Annee', 'Semaine').
        - Retourne les DataFrames transformés, les objets d'encodage/scaling, et les listes de colonnes utiles.

    Args:
        df_train (pd.DataFrame): Données d'entraînement contenant la cible 'y' et les exogènes.
        df_test (pd.DataFrame): Données de test contenant la cible 'y' et les exogènes.
        exog_cols (list): Liste des colonnes exogènes à traiter.

    Returns:
        tuple:
            - df_train (pd.DataFrame): Données d'entraînement transformées.
            - df_test (pd.DataFrame): Données de test transformées.
            - final_exog_cols (list): Liste finale des colonnes exogènes après encodage/standardisation.
            - ohe_dict (dict): Dictionnaire des OneHotEncoder utilisés pour chaque variable catégorielle.
            - scaler_exog (StandardScaler or None): Scaler utilisé pour les exogènes numériques (None si aucune).
            - scaler_y (StandardScaler): Scaler utilisé pour la cible 'y'.
            - numerical_exog_cols (list): Liste des colonnes exogènes numériques standardisées.
            - categorical_cols (list): Liste des colonnes exogènes catégorielles encodées.
    """
    # Détection colonnes numériques/catégorielles
    numeric_cols = []
    categorical_cols = []
    for col in exog_cols:
        if col in df_train.columns:
            if pd.api.types.is_numeric_dtype(df_train[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

    # Encodage OneHot pour les colonnes catégorielles
    ohe_dict = {}
    for col in categorical_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_train = ohe.fit_transform(df_train[[col]].astype(str))
        encoded_test = ohe.transform(df_test[[col]].astype(str))
        encoded_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
        encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_cols, index=df_train.index)
        encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_cols, index=df_test.index)
        df_train = pd.concat([df_train.drop(columns=[col]), encoded_train_df], axis=1)
        df_test = pd.concat([df_test.drop(columns=[col]), encoded_test_df], axis=1)
        ohe_dict[col] = ohe
        # Mise à jour exog_cols
        exog_cols = [c for c in exog_cols if c != col] + encoded_cols

    # Standardisation de la target (remplacement de 'y' par la version standardisée)
    scaler_y = StandardScaler()
    df_train['y'] = scaler_y.fit_transform(df_train[['y']])
    df_test['y'] = scaler_y.transform(df_test[['y']])

    # Standardisation des variables numériques exogènes
    time_cols = ['Annee', 'Semaine']
    numerical_exog_cols = [col for col in exog_cols if col not in time_cols and pd.api.types.is_numeric_dtype(df_train[col])]
    scaler_exog = StandardScaler()
    if numerical_exog_cols:
        df_train_exog_scaled = scaler_exog.fit_transform(df_train[numerical_exog_cols])
        df_test_exog_scaled = scaler_exog.transform(df_test[numerical_exog_cols])
        for i, col in enumerate(numerical_exog_cols):
            df_train[col] = df_train_exog_scaled[:, i]
            df_test[col] = df_test_exog_scaled[:, i]
    else:
        scaler_exog = None

    # Colonnes exogènes finales (post-encodage)
    final_exog_cols = [col for col in exog_cols if col in df_train.columns]

    return df_train, df_test, final_exog_cols, ohe_dict, scaler_exog, scaler_y, numerical_exog_cols, categorical_cols

# Version modifiée de preprocess_exog_and_target2 (pas de standardisation, que l'encodage)
def preprocess_exog_and_target2(df_train, df_test, exog_cols):
    """
    Prétraite les variables exogènes pour l'entraînement et le test (encodage uniquement, pas de standardisation).

    Cette fonction :
        - Détecte les colonnes numériques et catégorielles parmi les exogènes.
        - Applique un encodage OneHot sur les variables catégorielles (avec gestion des catégories inconnues).
        - Met à jour la liste des colonnes exogènes après encodage.
        - Ne standardise pas la cible 'y' ni les variables numériques.

    Args:
        df_train (pd.DataFrame): Données d'entraînement contenant les exogènes.
        df_test (pd.DataFrame): Données de test contenant les exogènes.
        exog_cols (list): Liste des colonnes exogènes à traiter.

    Returns:
        tuple:
            - df_train (pd.DataFrame): Données d'entraînement transformées.
            - df_test (pd.DataFrame): Données de test transformées.
            - final_exog_cols (list): Liste finale des colonnes exogènes après encodage.
            - ohe_dict (dict): Dictionnaire des OneHotEncoder utilisés pour chaque variable catégorielle.
            - categorical_cols (list): Liste des colonnes exogènes catégorielles encodées.
    """
    

    numeric_cols = []
    categorical_cols = []
    for col in exog_cols:
        if col in df_train.columns:
            if pd.api.types.is_numeric_dtype(df_train[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

    ohe_dict = {}
    for col in categorical_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_train = ohe.fit_transform(df_train[[col]].astype(str))
        encoded_test = ohe.transform(df_test[[col]].astype(str))
        encoded_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
        encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_cols, index=df_train.index)
        encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_cols, index=df_test.index)
        df_train = pd.concat([df_train.drop(columns=[col]), encoded_train_df], axis=1)
        df_test = pd.concat([df_test.drop(columns=[col]), encoded_test_df], axis=1)
        ohe_dict[col] = ohe
        exog_cols = [c for c in exog_cols if c != col] + encoded_cols

    final_exog_cols = [col for col in exog_cols if col in df_train.columns]

    return df_train, df_test, final_exog_cols, ohe_dict, categorical_cols


## fonction de préparation des données pour les zones

def prepare_data_zones(df: pd.DataFrame, target_col: str = 'Pjft_moyen', 
                      zone_col: str = 'Zone_traitement',
                      exog_vars: list = None):
    """
    Prépare un DataFrame unique pour la modélisation multi-zones à partir de données brutes (cerco + météo).

    Cette fonction réalise les étapes suivantes :
        1. Vérifie la présence des colonnes essentielles (zone, année, semaine, cible).
        2. Crée une colonne de date ('ds') à partir de l'année et de la semaine ISO.
        3. Crée une colonne identifiant unique de zone ('unique_id').
        4. Sélectionne et conserve uniquement les colonnes nécessaires à la modélisation :
            - Identifiants ('unique_id', 'ds', 'Annee', 'Semaine')
            - Cible (renommée en 'y')
            - Variables exogènes disponibles (parmi exog_vars)
        5. Trie les données par zone et date.
        6. Nettoie les zones invalides (aucune donnée ou toutes cibles manquantes).
        7. Affiche un résumé statistique par zone (nombre d'observations, taux de valeurs manquantes, période).
        8. Retourne le DataFrame final prêt pour la modélisation, le nom de la cible, et la liste des exogènes retenues.

    Args:
        df (pd.DataFrame): Données d'entrée contenant au moins les colonnes zone, année, semaine, cible et exogènes.
        target_col (str): Nom de la colonne cible à prédire (ex: 'Pjft_moyen').
        zone_col (str): Nom de la colonne identifiant la zone de traitement.
        exog_vars (list, optionnel): Liste des variables exogènes à inclure. Si None, utilise la liste par défaut.

    Returns:
        tuple:
            - df_processed (pd.DataFrame): DataFrame unique multi-zones, formaté et nettoyé, prêt pour la modélisation.
            - target_col_name (str): Nom de la colonne cible d'origine (avant renommage en 'y').
            - available_exog (list): Liste des variables exogènes effectivement présentes et conservées.

    Raises:
        ValueError: Si des colonnes essentielles sont manquantes dans le DataFrame d'entrée.
    """
    target_col_name = target_col

    if exog_vars is None:
        exog_vars = [
            'Tmoy_mean', 'Tmoy_d1', 'Tmoy_d2', 'Tmoy_d3', 'Tmoy_d4', 'Tmoy_d5', 'Tmoy_d6', 'Tmoy_d7', 'Tmoy_d8', 'Tmoy_d9',
            'Tmax_mean', 'Tmax_d1', 'Tmax_d2', 'Tmax_d3', 'Tmax_d4', 'Tmax_d5', 'Tmax_d6', 'Tmax_d7', 'Tmax_d8', 'Tmax_d9',
            'Tmin_mean', 'Tmin_d1', 'Tmin_d2', 'Tmin_d3', 'Tmin_d4', 'Tmin_d5', 'Tmin_d6', 'Tmin_d7', 'Tmin_d8', 'Tmin_d9',
            'Pluie_sum', 'Pluie_d1', 'Pluie_d2', 'Pluie_d3', 'Pluie_d4', 'Pluie_d5', 'Pluie_d6', 'Pluie_d7', 'Pluie_d8', 'Pluie_d9',
            'Humidite_max_mean', 'Humidite_max_d1', 'Humidite_max_d2', 'Humidite_max_d3', 'Humidite_max_d4', 'Humidite_max_d5',
            'Humidite_max_d6', 'Humidite_max_d7', 'Humidite_max_d8', 'Humidite_max_d9',
            'Humidite_min_mean', 'Humidite_min_d1', 'Humidite_min_d2', 'Humidite_min_d3', 'Humidite_min_d4', 'Humidite_min_d5',
            'Humidite_min_d6', 'Humidite_min_d7', 'Humidite_min_d8', 'Humidite_min_d9', 'Intensite_pluie'
        ]

    print(" Début de la préparation des données...")

    # Vérification des colonnes requises
    required_cols = [zone_col, 'Annee', 'Semaine', target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le DataFrame: {missing_cols}")

    # Copie du DataFrame pour ne pas modifier l'original
    df_processed = df.copy()

    # Détecter toutes les zones uniques dans la base
    zones_uniques = df_processed[zone_col].unique()
    print(f" Détection automatique: {len(zones_uniques)} zones trouvées dans la base")
    print(f"Zones détectées: {list(zones_uniques)}")

    # 1. CRÉATION DE LA COLONNE DS pour toute la base
    print(" Création de la colonne 'ds' à partir d'Annee et Semaine...")
    df_processed['ds'] = pd.to_datetime(
        df_processed['Annee'].astype(str) + df_processed['Semaine'].astype(str).str.zfill(2) + '1', 
        format='%Y%W%w'
    )

    # 2. CRÉATION DE LA COLONNE UNIQUE_ID pour toute la base
    print(" Création de la colonne 'unique_id'...")
    df_processed['unique_id'] = df_processed[zone_col]

    # 3. SÉLECTION DES COLONNES DE BASE
    base_cols = ['unique_id', 'ds', target_col, 'Annee', 'Semaine']

    # 4. VÉRIFICATION DES VARIABLES EXOGÈNES DISPONIBLES
    print(" Vérification des variables exogènes...")
    available_exog = [col for col in exog_vars if col in df_processed.columns]
    missing_exog = [col for col in exog_vars if col not in df_processed.columns]

    print(f" Variables exogènes disponibles: {len(available_exog)}/{len(exog_vars)}")
    if missing_exog:
        print(f"  Variables exogènes manquantes: {len(missing_exog)} variables")
        print(f"   Manquantes: {missing_exog[:5]}{'...' if len(missing_exog) > 5 else ''}")

    # 5. COLONNES À CONSERVER
    cols_to_keep = base_cols + available_exog
    print(f" Colonnes à conserver: {len(cols_to_keep)} colonnes")

    # 6. FILTRAGE DES COLONNES
    df_processed = df_processed[cols_to_keep]

    # 7. TRI PAR ZONE ET DATE
    print(" Tri des données par zone et date...")
    df_processed = df_processed.sort_values(['unique_id', 'Annee', 'Semaine'])

    # 8. RENOMMAGE DE LA COLONNE CIBLE
    print(f" Renommage de la colonne cible '{target_col}' -> 'y'")
    df_processed = df_processed.rename(columns={target_col: 'y'})

    # 9. NETTOYAGE ET STATISTIQUES PAR ZONE
    print(" Calcul des statistiques par zone...")
    zones_valides = []
    zones_invalides = []

    for zone in zones_uniques:
        df_zone = df_processed[df_processed['unique_id'] == zone]

        # Vérifier si la zone a des données valides
        if df_zone.empty:
            zones_invalides.append(f"{zone} (aucune donnée)")
            continue

        if df_zone['y'].isna().all():
            zones_invalides.append(f"{zone} (toutes valeurs cibles manquantes)")
            continue

        # Statistiques de la zone
        n_obs = len(df_zone)
        pct_missing = (df_zone['y'].isna().sum() / n_obs) * 100
        date_debut = df_zone['ds'].min()
        date_fin = df_zone['ds'].max()

        zones_valides.append({
            'zone': zone,
            'observations': n_obs,
            'pct_missing': pct_missing,
            'debut': date_debut,
            'fin': date_fin
        })

    # 10. SUPPRESSION DES ZONES INVALIDES
    if zones_invalides:
        print(f"  Zones avec problèmes: {len(zones_invalides)}")
        for zone_invalide in zones_invalides:
            print(f"   • {zone_invalide}")

        # Supprimer les zones invalides du DataFrame
        zones_a_garder = [z['zone'] for z in zones_valides]
        df_processed = df_processed[df_processed['unique_id'].isin(zones_a_garder)]

    # 11. RESET DE L'INDEX
    df_processed = df_processed.reset_index(drop=True)

    # 12. RÉSUMÉ FINAL
    print(f"\n{'='*60}")
    print(f" RÉSUMÉ FINAL DE LA PRÉPARATION")
    print(f"{'='*60}")
    print(f" Zones valides: {len(zones_valides)}")
    print(f" Zones supprimées: {len(zones_invalides)}")
    print(f" Total observations: {len(df_processed)}")
    print(f" Période couverte: {df_processed['ds'].min().strftime('%Y-%m-%d')} à {df_processed['ds'].max().strftime('%Y-%m-%d')}")
    print(f" Colonnes finales: {len(df_processed.columns)}")
    print(f" Colonnes: {list(df_processed.columns)}")

    # Statistiques par zone (top 5)
    print(f"\n Aperçu des zones (top 5):")
    for i, zone_info in enumerate(zones_valides[:5]):
        print(f"   {i+1}. {zone_info['zone']}: {zone_info['observations']} obs, "
              f"{zone_info['pct_missing']:.1f}% manquant")

    if len(zones_valides) > 5:
        print(f"   ... et {len(zones_valides) - 5} autres zones")

    print(f"\n Données préparées avec succès!")
    print(f" Retour d'un DataFrame unique avec {len(df_processed)} lignes et {len(df_processed.columns)} colonnes")

    return df_processed, target_col_name, available_exog

def prepare_data_zones2(df: pd.DataFrame, target_col: str = 'Pjft_moyen', 
                      zone_col: str = 'Zone_traitement',
                      exog_vars: list = None):
    """
    Prépare un DataFrame unique pour la modélisation multi-zones à partir de données brutes (cerco + météo + INTRANT).

    Cette fonction réalise les étapes suivantes :
        1. Vérifie la présence des colonnes essentielles (zone, année, semaine, cible).
        2. Crée une colonne de date ('ds') à partir de l'année et de la semaine ISO.
        3. Crée une colonne identifiant unique de zone ('unique_id').
        4. Sélectionne et conserve uniquement les colonnes nécessaires à la modélisation :
            - Identifiants ('unique_id', 'ds', 'Annee', 'Semaine')
            - Cible (renommée en 'y')
            - Variables exogènes disponibles (parmi exog_vars)
        5. Trie les données par zone et date.
        6. Nettoie les zones invalides (aucune donnée ou toutes cibles manquantes).
        7. Affiche un résumé statistique par zone (nombre d'observations, taux de valeurs manquantes, période).
        8. Retourne le DataFrame final prêt pour la modélisation, le nom de la cible, et la liste des exogènes retenues.

    Args:
        df (pd.DataFrame): Données d'entrée contenant au moins les colonnes zone, année, semaine, cible et exogènes.
        target_col (str): Nom de la colonne cible à prédire (ex: 'Pjft_moyen').
        zone_col (str): Nom de la colonne identifiant la zone de traitement.
        exog_vars (list, optionnel): Liste des variables exogènes à inclure. Si None, utilise la liste par défaut.

    Returns:
        tuple:
            - df_processed (pd.DataFrame): DataFrame unique multi-zones, formaté et nettoyé, prêt pour la modélisation.
            - target_col_name (str): Nom de la colonne cible d'origine (avant renommage en 'y').
            - available_exog (list): Liste des variables exogènes effectivement présentes et conservées.

    Raises:
        ValueError: Si des colonnes essentielles sont manquantes dans le DataFrame d'entrée.
    """
    target_col_name = target_col

    if exog_vars is None:
        exog_vars = [
            'Tmoy_mean', 'Tmoy_d1', 'Tmoy_d2', 'Tmoy_d3', 'Tmoy_d4', 'Tmoy_d5', 'Tmoy_d6', 'Tmoy_d7', 'Tmoy_d8', 'Tmoy_d9',
            'Tmax_mean', 'Tmax_d1', 'Tmax_d2', 'Tmax_d3', 'Tmax_d4', 'Tmax_d5', 'Tmax_d6', 'Tmax_d7', 'Tmax_d8', 'Tmax_d9',
            'Tmin_mean', 'Tmin_d1', 'Tmin_d2', 'Tmin_d3', 'Tmin_d4', 'Tmin_d5', 'Tmin_d6', 'Tmin_d7', 'Tmin_d8', 'Tmin_d9',
            'Pluie_sum', 'Pluie_d1', 'Pluie_d2', 'Pluie_d3', 'Pluie_d4', 'Pluie_d5', 'Pluie_d6', 'Pluie_d7', 'Pluie_d8', 'Pluie_d9',
            'Humidite_max_mean', 'Humidite_max_d1', 'Humidite_max_d2', 'Humidite_max_d3', 'Humidite_max_d4', 'Humidite_max_d5',
            'Humidite_max_d6', 'Humidite_max_d7', 'Humidite_max_d8', 'Humidite_max_d9',
            'Humidite_min_mean', 'Humidite_min_d1', 'Humidite_min_d2', 'Humidite_min_d3', 'Humidite_min_d4', 'Humidite_min_d5',
            'Humidite_min_d6', 'Humidite_min_d7', 'Humidite_min_d8', 'Humidite_min_d9', 'Intensite_pluie','Intrant_revu', 'Quantite',
            'Quantite_huile', 'Nb_jours_entre_2_traitements', 'Nb_jours_2mm_tr','Categorie'
        ]

    print(" Début de la préparation des données...")

    # Vérification des colonnes requises
    required_cols = [zone_col, 'Annee', 'Semaine', target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le DataFrame: {missing_cols}")

    # Copie du DataFrame pour ne pas modifier l'original
    df_processed = df.copy()

    # Détecter toutes les zones uniques dans la base
    zones_uniques = df_processed[zone_col].unique()
    print(f" Détection automatique: {len(zones_uniques)} zones trouvées dans la base")
    print(f"Zones détectées: {list(zones_uniques)}")

    # 1. CRÉATION DE LA COLONNE DS pour toute la base
    print(" Création de la colonne 'ds' à partir d'Annee et Semaine...")
    df_processed['ds'] = pd.to_datetime(
        df_processed['Annee'].astype(str) + df_processed['Semaine'].astype(str).str.zfill(2) + '1', 
        format='%Y%W%w'
    )

    # 2. CRÉATION DE LA COLONNE UNIQUE_ID pour toute la base
    print(" Création de la colonne 'unique_id'...")
    df_processed['unique_id'] = df_processed[zone_col]

    # 3. SÉLECTION DES COLONNES DE BASE
    base_cols = ['unique_id', 'ds', target_col, 'Annee', 'Semaine']

    # 4. VÉRIFICATION DES VARIABLES EXOGÈNES DISPONIBLES
    print(" Vérification des variables exogènes...")
    available_exog = [col for col in exog_vars if col in df_processed.columns]
    missing_exog = [col for col in exog_vars if col not in df_processed.columns]

    print(f" Variables exogènes disponibles: {len(available_exog)}/{len(exog_vars)}")
    if missing_exog:
        print(f"  Variables exogènes manquantes: {len(missing_exog)} variables")
        print(f"   Manquantes: {missing_exog[:5]}{'...' if len(missing_exog) > 5 else ''}")

    # 5. COLONNES À CONSERVER
    cols_to_keep = base_cols + available_exog
    print(f" Colonnes à conserver: {len(cols_to_keep)} colonnes")

    # 6. FILTRAGE DES COLONNES
    df_processed = df_processed[cols_to_keep]

    # 7. TRI PAR ZONE ET DATE
    print(" Tri des données par zone et date...")
    df_processed = df_processed.sort_values(['unique_id', 'Annee', 'Semaine'])

    # 8. RENOMMAGE DE LA COLONNE CIBLE
    print(f" Renommage de la colonne cible '{target_col}' -> 'y'")
    df_processed = df_processed.rename(columns={target_col: 'y'})

    # 9. NETTOYAGE ET STATISTIQUES PAR ZONE
    print(" Calcul des statistiques par zone...")
    zones_valides = []
    zones_invalides = []

    for zone in zones_uniques:
        df_zone = df_processed[df_processed['unique_id'] == zone]

        # Vérifier si la zone a des données valides
        if df_zone.empty:
            zones_invalides.append(f"{zone} (aucune donnée)")
            continue

        if df_zone['y'].isna().all():
            zones_invalides.append(f"{zone} (toutes valeurs cibles manquantes)")
            continue

        # Statistiques de la zone
        n_obs = len(df_zone)
        pct_missing = (df_zone['y'].isna().sum() / n_obs) * 100
        date_debut = df_zone['ds'].min()
        date_fin = df_zone['ds'].max()

        zones_valides.append({
            'zone': zone,
            'observations': n_obs,
            'pct_missing': pct_missing,
            'debut': date_debut,
            'fin': date_fin
        })

    # 10. SUPPRESSION DES ZONES INVALIDES
    if zones_invalides:
        print(f"  Zones avec problèmes: {len(zones_invalides)}")
        for zone_invalide in zones_invalides:
            print(f"   • {zone_invalide}")

        # Supprimer les zones invalides du DataFrame
        zones_a_garder = [z['zone'] for z in zones_valides]
        df_processed = df_processed[df_processed['unique_id'].isin(zones_a_garder)]

    # 11. RESET DE L'INDEX
    df_processed = df_processed.reset_index(drop=True)

    # 12. RÉSUMÉ FINAL
    print(f"\n{'='*60}")
    print(f" RÉSUMÉ FINAL DE LA PRÉPARATION")
    print(f"{'='*60}")
    print(f" Zones valides: {len(zones_valides)}")
    print(f" Zones supprimées: {len(zones_invalides)}")
    print(f" Total observations: {len(df_processed)}")
    print(f" Période couverte: {df_processed['ds'].min().strftime('%Y-%m-%d')} à {df_processed['ds'].max().strftime('%Y-%m-%d')}")
    print(f" Colonnes finales: {len(df_processed.columns)}")
    print(f" Colonnes: {list(df_processed.columns)}")

    # Statistiques par zone (top 5)
    print(f"\n Aperçu des zones (top 5):")
    for i, zone_info in enumerate(zones_valides[:5]):
        print(f"   {i+1}. {zone_info['zone']}: {zone_info['observations']} obs, "
              f"{zone_info['pct_missing']:.1f}% manquant")

    if len(zones_valides) > 5:
        print(f"   ... et {len(zones_valides) - 5} autres zones")

    print(f"\n Données préparées avec succès!")
    print(f" Retour d'un DataFrame unique avec {len(df_processed)} lignes et {len(df_processed.columns)} colonnes")

    return df_processed, target_col_name, available_exog


def train_test_split_cercos_zones(df: pd.DataFrame, test_size: int = 12, 
                                  min_total_size: int = 15, min_train_size: int = 3, 
                                  verbose: bool = True):
    """
    Effectue un split train/test par zone pour les données CERCOS.
    Chaque zone a ses propres 'test_size' dernières semaines comme test.
    
    Args:
        df: DataFrame avec colonnes 'unique_id', 'ds', 'y' et variables exogènes
        test_size: Nombre de semaines à garder pour le test par zone (défaut: 12)
        min_total_size: Taille minimale totale pour inclure une zone (défaut: 15)
        min_train_size: Taille minimale pour le train par zone (défaut: 3)
        verbose: Afficher les détails du split (défaut: True)
    
    Returns:
        tuple: (df_train, df_test) - DataFrames d'entraînement et de test
    """
    
    if verbose:
        print(" Début du split train/test par zone...")
        print(f" Paramètres: test_size={test_size}, min_total={min_total_size}, min_train={min_train_size}")
    
    # Vérifications
    required_cols = ['unique_id', 'ds', 'y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes: {missing_cols}")
    
    # Listes pour stocker les DataFrames de chaque zone
    train_dfs = []
    test_dfs = []
    
    # Statistiques
    zones_traitees = 0
    zones_ignorees_taille = 0
    zones_ignorees_train = 0
    zones_ignorees_cible = 0
    
    # Traitement par zone
    zones_uniques = df['unique_id'].unique()
    
    if verbose:
        print(f" Traitement de {len(zones_uniques)} zones...")
    
    for i, zone in enumerate(zones_uniques):
        # Filtrer et trier les données de la zone
        df_zone = df[df['unique_id'] == zone].copy().sort_values(['Annee', 'Semaine'])
        
        if verbose and i < 5:  # Afficher les détails pour les 5 premières zones
            print(f"\n Zone {zone} ({i+1}/{len(zones_uniques)}):")
            print(f"    Total observations: {len(df_zone)}")
        
        # Vérification 1: Taille minimale
        if len(df_zone) < min_total_size:
            zones_ignorees_taille += 1
            if verbose and i < 5:
                print(f"    Ignorée: données insuffisantes ({len(df_zone)} < {min_total_size})")
            continue
        
        # Vérification 2: Données cibles valides
        if df_zone['y'].isna().all():
            zones_ignorees_cible += 1
            if verbose and i < 5:
                print(f"    Ignorée: toutes valeurs cibles manquantes")
            continue
        
        # Calcul de l'index de split
        split_idx = len(df_zone) - test_size
        
        # Vérification 3: Train suffisant
        if split_idx < min_train_size:
            zones_ignorees_train += 1
            if verbose and i < 5:
                print(f"    Ignorée: train trop petit ({split_idx} < {min_train_size})")
            continue
        
        # Effectuer le split
        df_train_zone = df_zone.iloc[:split_idx].copy()
        df_test_zone = df_zone.iloc[split_idx:].copy()
        
        # Vérification supplémentaire: au moins une valeur non-nulle dans train
        if df_train_zone['y'].notna().sum() == 0:
            zones_ignorees_cible += 1
            if verbose and i < 5:
                print(f"  Ignorée: aucune valeur cible valide dans train")
            continue
        
        # Ajouter aux listes
        train_dfs.append(df_train_zone)
        test_dfs.append(df_test_zone)
        zones_traitees += 1
        
        if verbose and i < 5:
            pct_missing_train = (df_train_zone['y'].isna().sum() / len(df_train_zone)) * 100
            pct_missing_test = (df_test_zone['y'].isna().sum() / len(df_test_zone)) * 100
            print(f"    Split réussi:")
            print(f"       Train: {len(df_train_zone)} semaines (manquant: {pct_missing_train:.1f}%)")
            print(f"       Test: {len(df_test_zone)} semaines (manquant: {pct_missing_test:.1f}%)")
            print(f"       Train: {df_train_zone['ds'].min().strftime('%Y-%m-%d')} à {df_train_zone['ds'].max().strftime('%Y-%m-%d')}")
            print(f"       Test: {df_test_zone['ds'].min().strftime('%Y-%m-%d')} à {df_test_zone['ds'].max().strftime('%Y-%m-%d')}")
    
    # Concaténation des résultats
    if train_dfs:
        df_train = pd.concat(train_dfs, ignore_index=True)
        df_test = pd.concat(test_dfs, ignore_index=True)
    else:
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        raise ValueError("Aucune zone n'a pu être traitée pour le split train/test")
    
    # Résumé final
    if verbose:
        print(f"\n{'='*60}")
        print(f" RÉSUMÉ DU SPLIT TRAIN/TEST")
        print(f"{'='*60}")
        print(f" Zones traitées avec succès: {zones_traitees}")
        print(f" Zones ignorées:")
        print(f"   • Taille insuffisante: {zones_ignorees_taille}")
        print(f"   • Train trop petit: {zones_ignorees_train}")
        print(f"   • Problème cible: {zones_ignorees_cible}")
        print(f" Taux de succès: {(zones_traitees/len(zones_uniques)*100):.1f}%")
        print(f"\n Données finales:")
        print(f"    Train: {len(df_train)} observations")
        print(f"    Test: {len(df_test)} observations")
        print(f"    Période train: {df_train['ds'].min().strftime('%Y-%m-%d')} à {df_train['ds'].max().strftime('%Y-%m-%d')}")
        print(f"    Période test: {df_test['ds'].min().strftime('%Y-%m-%d')} à {df_test['ds'].max().strftime('%Y-%m-%d')}")
        
        # Statistiques par zone (échantillon)
        if zones_traitees > 0:
            train_by_zone = df_train.groupby('unique_id').size()
            test_by_zone = df_test.groupby('unique_id').size()
            print(f"\n Aperçu train/test par zone:")
            print(f"    Train moyen par zone: {train_by_zone.mean():.1f} semaines")
            print(f"    Test moyen par zone: {test_by_zone.mean():.1f} semaines")
            print(f"    Train min/max: {train_by_zone.min()}/{train_by_zone.max()} semaines")
            print(f"    Test min/max: {test_by_zone.min()}/{test_by_zone.max()} semaines")
    
    print(f"\n Split terminé avec succès!")
    
    return df_train, df_test


def visualiser_zone(
    base_path,
    target_col_name,
    zone,
    pred_col='y_pred',  # colonne générique
    train_filename='train_data.csv',
    test_filename='predictions.csv'
):
    """
    Affiche l'évolution temporelle de la cible et des prédictions pour une zone donnée.

    Cette fonction :
        - Charge les données d'entraînement et de test pour une zone spécifique.
        - Affiche l'historique (train), les vraies valeurs de test et les prédictions.
        - Calcule et affiche les métriques RMSE et MAE sur le test.
        - Trace une ligne de séparation entre train et test.

    Args:
        base_path (str): Chemin de base où sont stockés les résultats.
        target_col_name (str): Nom de la variable cible (pour l'affichage).
        zone (str): Nom ou identifiant de la zone à visualiser.
        pred_col (str, optionnel): Nom de la colonne de prédiction dans le fichier test (défaut 'y_pred').
        train_filename (str, optionnel): Nom du fichier CSV contenant les données d'entraînement (défaut 'train_data.csv').
        test_filename (str, optionnel): Nom du fichier CSV contenant les prédictions de test (défaut 'predictions.csv').

    Returns:
        None. Affiche directement la figure matplotlib.
    """
    import numpy as np
    # Chemins
    save_dir = os.path.join(base_path, f"models_{target_col_name}_{zone}")
    train_path = os.path.join(save_dir, train_filename)
    test_path = os.path.join(save_dir, test_filename)

    if not os.path.exists(train_path):
        print(f"Train data not found: {train_path}")
        return
    if not os.path.exists(test_path):
        print(f"Predictions not found: {test_path}")
        return

    # Chargement et conversion des dates
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Conversion en datetime
    train_df['ds'] = pd.to_datetime(train_df['ds'])
    test_df['ds'] = pd.to_datetime(test_df['ds'])

    # Colonne 'y_pred'
    if pred_col not in test_df.columns:
        pred_cols = [c for c in test_df.columns if c not in ['unique_id', 'ds', 'y']]
        if len(pred_cols) == 1:
            pred_col = pred_cols[0]
        else:
            raise ValueError(f"Colonne de prédiction non trouvée dans {test_df.columns}")

    y_pred = test_df[pred_col]
    y_true = test_df['y']

    # Calcul des métriques
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    

    # Visualisation style PRO
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Historique (train)
    ax.plot(train_df['ds'], train_df['y'], 'gray', alpha=0.7,
            label='Historique (train)', linewidth=1.5)

    # Vrais test
    ax.plot(test_df['ds'], y_true, 'b-', label='Réel (test)', linewidth=2.5,
            marker='o', markersize=6)

    # Prédictions
    ax.plot(test_df['ds'], y_pred, 'r--', label='Prédiction', linewidth=2.5,
            marker='s', markersize=6)

    # Ligne séparation train/test
    ax.axvline(test_df['ds'].iloc[0], color='green', linestyle=':',
               label='Début test', alpha=0.8, linewidth=2)

    # Titre avec métriques
    ax.set_title(
        f'Zone {zone}\n'
        f'RMSE: {rmse:.3f} | MAE: {mae:.3f}',
        fontsize=16, fontweight='bold', pad=20
    )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(target_col_name, fontsize=12)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)

    # Rotation des dates et espacement
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    plt.show()



## Fonction utilitaire pour la préparation des données des modèles de prévision futur

def train_test_split_futur_zones(
    df: pd.DataFrame,
    split_annee: int,
    split_semaine: int,
    test_size: int = 4,
    min_total_size: int = 15,
    min_train_size: int = 3,
    verbose: bool = True
):
    """
    Effectue un découpage train/test temporel par zone pour la prévision future.

    Pour chaque zone, cette fonction :
        - Sélectionne les données d'entraînement jusqu'à la semaine (split_annee, split_semaine) incluse.
        - Sélectionne les données de test comme les `test_size` semaines suivantes immédiatement après la coupure.
        - Ignore le reste des données.
        - Exclut les zones avec trop peu de données totales, un train trop petit, ou des cibles manquantes.
        - Affiche un résumé détaillé du split si verbose=True.

    Args:
        df (pd.DataFrame): Données contenant au minimum 'unique_id', 'Annee', 'Semaine', 'y'.
        split_annee (int): Année de coupure pour le split train/test.
        split_semaine (int): Semaine de coupure pour le split train/test.
        test_size (int, optionnel): Nombre de semaines à inclure dans le test après la coupure (défaut: 4).
        min_total_size (int, optionnel): Nombre minimal d'observations pour inclure une zone (défaut: 15).
        min_train_size (int, optionnel): Nombre minimal d'observations dans le train (défaut: 3).
        verbose (bool, optionnel): Affiche les détails du split si True (défaut: True).

    Returns:
        tuple:
            - df_train (pd.DataFrame): Données d'entraînement concaténées pour toutes les zones.
            - df_test (pd.DataFrame): Données de test concaténées pour toutes les zones.

    Raises:
        ValueError: Si aucune zone n'a pu être traitée pour le split train/test.
    """
    required_cols = ['unique_id', 'Annee', 'Semaine', 'y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes: {missing_cols}")

    train_dfs = []
    test_dfs = []
    zones_traitees = 0
    zones_ignores_taille = 0
    zones_ignores_train = 0
    zones_ignores_cible = 0

    zones_uniques = df['unique_id'].unique()
    if verbose:
        print(f" Début du split train/test 'futur' par zone...")
        print(f"Split à partir de {split_annee} semaine {split_semaine} | test_size = {test_size}")
        print(f" Traitement de {len(zones_uniques)} zones...")

    for i, zone in enumerate(zones_uniques):
        df_zone = df[df['unique_id'] == zone].copy().sort_values(['Annee', 'Semaine'])
        if verbose and i < 5:
            print(f"\n Zone {zone} ({i+1}/{len(zones_uniques)}):")
            print(f"    Total observations: {len(df_zone)}")

        # 1. Vérification taille minimale totale
        if len(df_zone) < min_total_size:
            zones_ignores_taille += 1
            if verbose and i < 5:
                print(f"    Ignorée: données insuffisantes ({len(df_zone)} < {min_total_size})")
            continue

        # 2. Sélection train/test via coupure
        # Index de la coupure
        mask_train = ((df_zone['Annee'] < split_annee) |
                      ((df_zone['Annee'] == split_annee) & (df_zone['Semaine'] <= split_semaine)))
        df_train_zone = df_zone[mask_train].copy()

        # Le test commence juste après la coupure (année/semaine), pour test_size semaines
        mask_test = ((df_zone['Annee'] > split_annee) |
                     ((df_zone['Annee'] == split_annee) & (df_zone['Semaine'] > split_semaine)))
        df_test_candidats = df_zone[mask_test].copy().sort_values(['Annee', 'Semaine'])
        df_test_zone = df_test_candidats.head(test_size)

        # Contrôle tailles
        if len(df_train_zone) < min_train_size:
            zones_ignores_train += 1
            if verbose and i < 5:
                print(f"    Ignorée: train trop petit ({len(df_train_zone)} < {min_train_size})")
            continue
        if df_train_zone['y'].isna().all():
            zones_ignores_cible += 1
            if verbose and i < 5:
                print(f"    Ignorée: toutes valeurs cibles manquantes dans train")
            continue

        train_dfs.append(df_train_zone)
        test_dfs.append(df_test_zone)
        zones_traitees += 1

        if verbose and i < 5:
            print(f"    Split réussi:")
            print(f"       Train: {len(df_train_zone)} semaines")
            print(f"       Test: {len(df_test_zone)} semaines")
            if not df_train_zone.empty:
                print(f"       Train: {df_train_zone['Annee'].min()}-{df_train_zone['Semaine'].min()} à {df_train_zone['Annee'].max()}-{df_train_zone['Semaine'].max()}")
            if not df_test_zone.empty:
                print(f"       Test: {df_test_zone['Annee'].min()}-{df_test_zone['Semaine'].min()} à {df_test_zone['Annee'].max()}-{df_test_zone['Semaine'].max()}")

    if train_dfs:
        df_train = pd.concat(train_dfs, ignore_index=True)
        df_test = pd.concat(test_dfs, ignore_index=True)
    else:
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        raise ValueError("Aucune zone n'a pu être traitée pour le split train/test")

    if verbose:
        print(f"\n{'='*60}")
        print(f" RÉSUMÉ DU SPLIT TRAIN/TEST FUTUR")
        print(f"{'='*60}")
        print(f" Zones traitées avec succès: {zones_traitees}")
        print(f" Zones ignorées:")
        print(f"   • Taille insuffisante: {zones_ignores_taille}")
        print(f"   • Train trop petit: {zones_ignores_train}")
        print(f"   • Problème cible: {zones_ignores_cible}")
        print(f" Taux de succès: {(zones_traitees/len(zones_uniques)*100):.1f}%")
        print(f"\n Données finales:")
        print(f"    Train: {len(df_train)} observations")
        print(f"    Test: {len(df_test)} observations")
        if not df_train.empty:
            print(f"    Période train: {df_train['Annee'].min()}-{df_train['Semaine'].min()} à {df_train['Annee'].max()}-{df_train['Semaine'].max()}")
        if not df_test.empty:
            print(f"    Période test: {df_test['Annee'].min()}-{df_test['Semaine'].min()} à {df_test['Annee'].max()}-{df_test['Semaine'].max()}")

    print(f"\n Split terminé avec succès!")
    return df_train, df_test


def visualiser_futur_zone(
    base_path,
    target_col_name,
    zone,
    train_filename='train_futur.csv',
    futur_pred_filename='futur_predictions.csv',
    pred_col='y_pred',
    ytrue_col='y_true',  # optionnel, si tu as le vrai y dans df_test
):
    """
    Visualise l'historique (train utilisé pour la prédiction future)
    et les prédictions futures pour une zone de traitement donnée.
    """
    save_dir = os.path.join(base_path, f"models_{target_col_name}_{zone}")
    train_path = os.path.join(save_dir, train_filename)
    fut_pred_path = os.path.join(save_dir, futur_pred_filename)

    if not os.path.exists(train_path):
        print(f"Train data not found: {train_path}")
        return
    if not os.path.exists(fut_pred_path):
        print(f"Futur predictions not found: {fut_pred_path}")
        return

    # Chargement
    train_df = pd.read_csv(train_path)
    fut_pred_df = pd.read_csv(fut_pred_path)

    # Dates en datetime
    train_df['ds'] = pd.to_datetime(train_df['ds'])
    fut_pred_df['ds'] = pd.to_datetime(fut_pred_df['ds'])

    # Y true futur (optionnel)
    has_y_true = ytrue_col in fut_pred_df.columns
    if has_y_true:
        y_true = fut_pred_df[ytrue_col]
    else:
        y_true = None

    # Y préd
    if pred_col not in fut_pred_df.columns:
        pred_cols = [c for c in fut_pred_df.columns if c not in ['unique_id', 'ds', ytrue_col, 'type_prediction']]
        if len(pred_cols) == 1:
            pred_col = pred_cols[0]
        else:
            raise ValueError(f"Colonne de prédiction non trouvée dans {fut_pred_df.columns}")

    y_pred = fut_pred_df[pred_col]

    # Calcul métriques si vrai y dispo
    if has_y_true and y_true.notna().all():
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        metrics_str = f'RMSE: {rmse:.3f} | MAE: {mae:.3f} '
    else:
        metrics_str = "Prédictions futures (pas de vérité connue)"

    # Visualisation
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Historique (train utilisé pour le refit)
    ax.plot(train_df['ds'], train_df['y'], 'gray', alpha=0.7,
            label='Historique (train)', linewidth=1.5)

    # Vrais y futurs (si dispo)
    if has_y_true and y_true.notna().all():
        ax.plot(fut_pred_df['ds'], y_true, 'b-', label='Réel (futur)', linewidth=2.5,
                marker='o', markersize=6)

    # Prédictions futures
    ax.plot(fut_pred_df['ds'], y_pred, 'r--', label='Prédiction futur', linewidth=2.5,
            marker='s', markersize=6)

    # Ligne de séparation
    ax.axvline(fut_pred_df['ds'].iloc[0], color='green', linestyle=':',
               label='Début prévision future', alpha=0.8, linewidth=2)

    ax.set_title(
        f'Zone {zone}\n{metrics_str}',
        fontsize=16, fontweight='bold', pad=20
    )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(target_col_name, fontsize=12)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    plt.show()




def transform_exo(df):
    """
    Applique des transformations de type lags sur des variables exogènes sélectionnées d'un DataFrame.

    Cette fonction :
        - Vérifie la présence des colonnes exogènes à transformer.
        - Applique des lags (retards) sur ces colonnes via la fonction transform_exog de mlforecast.
        - Ne conserve que les nouvelles colonnes transformées (lags) en plus de 'unique_id' et 'ds'.
        - Fusionne ces colonnes transformées au DataFrame d'origine sur ['unique_id', 'ds'].

    Args:
        df (pd.DataFrame): DataFrame d'entrée contenant au moins 'unique_id', 'ds' et les colonnes exogènes à transformer.

    Returns:
        pd.DataFrame: DataFrame enrichi avec les colonnes de lags pour les variables exogènes sélectionnées.

    Raises:
        ValueError: Si une ou plusieurs colonnes exogènes à transformer sont absentes du DataFrame.
    """
    # Liste des colonnes exogènes à transformer
    exo_cols_to_transform = [
        'Tmoy_mean', 'Tmax_mean', 'Tmin_mean', 'Pluie_sum',
        'Humidite_max_mean', 'Humidite_min_mean'
    ]
    # Vérifie que ces colonnes sont bien présentes dans le DataFrame
    missing_cols = [col for col in exo_cols_to_transform if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le DataFrame pour la transformation : {missing_cols}")

    # On sélectionne uniquement les colonnes nécessaires pour la transformation
    exog_df = df[['unique_id', 'ds'] + exo_cols_to_transform].copy()

    # Définir les lags et transformations souhaitées
    lags = [1,2,3,4, 5, 6]
    lag_transforms = {} 

    # Appliquer la transformation
    transformed_exog = transform_exog(exog_df, lags=lags, lag_transforms=lag_transforms)
    
     # Filtrer pour ne garder que les colonnes de transformation (pas celles d'origine)
    cols_transfo = [
        col for col in transformed_exog.columns
        if col not in ['unique_id', 'ds'] + exo_cols_to_transform
    ]
    # On garde unique_id, ds et les colonnes transformées
    transformed_only = transformed_exog[['unique_id', 'ds'] + cols_transfo]
    # Fusionner les nouvelles colonnes dans le DataFrame de départ
    df_merged = df.merge(
        transformed_only,
        on=['unique_id', 'ds'],
        how='left'
    )

    return df_merged







def visualiser_zone_train_test(
    base_path,
    target_col_name,
    zone,
    pred_col='y_pred',
    train_filename='predictions_train.csv',
    test_filename='predictions.csv'
):
    """
    Visualise l'historique et les prédictions pour une zone et une cible, 
    en utilisant predictions_train.csv et predictions.csv au lieu du rolling.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # Chemins
    save_dir = os.path.join(base_path, f"models_{target_col_name}_{zone}")
    train_path = os.path.join(save_dir, train_filename)
    test_path = os.path.join(save_dir, test_filename)

    if not os.path.exists(train_path):
        print(f"Train predictions not found: {train_path}")
        return
    if not os.path.exists(test_path):
        print(f"Test predictions not found: {test_path}")
        return

    # Chargement et conversion des dates
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df['ds'] = pd.to_datetime(train_df['ds'])
    test_df['ds'] = pd.to_datetime(test_df['ds'])

    # Colonne 'y_pred'
    if pred_col not in test_df.columns:
        pred_cols = [c for c in test_df.columns if c not in ['unique_id', 'ds', 'y']]
        if len(pred_cols) == 1:
            pred_col = pred_cols[0]
        else:
            raise ValueError(f"Colonne de prédiction non trouvée dans {test_df.columns}")

    if pred_col not in train_df.columns:
        pred_cols_train = [c for c in train_df.columns if c not in ['unique_id', 'ds', 'y']]
        if len(pred_cols_train) == 1:
            pred_col_train = pred_cols_train[0]
        else:
            raise ValueError(f"Colonne de prédiction non trouvée dans {train_df.columns}")
    else:
        pred_col_train = pred_col

    # Calcul des métriques
    rmse_test = mean_squared_error(test_df['y'], test_df[pred_col], squared=False)
    mae_test = mean_absolute_error(test_df['y'], test_df[pred_col])

    rmse_train = mean_squared_error(train_df['y'], train_df[pred_col_train], squared=False)
    mae_train = mean_absolute_error(train_df['y'], train_df[pred_col_train])

    # Visualisation
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Historique (train)
    ax.plot(train_df['ds'], train_df['y'], color='gray', alpha=0.7, label='Réel (train)', linewidth=1.5, marker='o')
    ax.plot(train_df['ds'], train_df[pred_col_train], color='orange', linestyle='--', linewidth=2, marker='s',
            label=f"Prédiction (train)\n(RMSE={rmse_train:.3f}, MAE={mae_train:.3f})")

    # Test
    ax.plot(test_df['ds'], test_df['y'], 'b-', label='Réel (test)', linewidth=2.5, marker='o', markersize=6)
    ax.plot(test_df['ds'], test_df[pred_col], 'r--', label=f"Prédiction (test)\n(RMSE={rmse_test:.3f}, MAE={mae_test:.3f})",
            linewidth=2.5, marker='s', markersize=6)

    # Ligne séparation train/test
    ax.axvline(test_df['ds'].iloc[0], color='green', linestyle=':', label='Début test', alpha=0.8, linewidth=2)

    # Titre avec métriques
    ax.set_title(
        f'Zone {zone}\n'
        f'Test - RMSE: {rmse_test:.3f} | MAE: {mae_test:.3f}\n'
        f'Train - RMSE: {rmse_train:.3f} | MAE: {mae_train:.3f}',
        fontsize=16, fontweight='bold', pad=20
    )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(target_col_name, fontsize=12)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    plt.show()