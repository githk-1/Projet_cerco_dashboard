
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
from mlforecast import MLForecast
 

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from mlforecast.utils import PredictionIntervals

import os
import json
import joblib
import contextlib
import streamlit as st
import logging
import random
import optuna
import logging


from tqdm.notebook import tqdm
from datetime import datetime, timedelta
import importlib
import pkg_resources
import pickle 
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, Any, List
from mlforecast.lag_transforms import (
    ExponentiallyWeightedMean, 
    RollingMean, 
    RollingStd,
    RollingMax,      
    RollingMin,
    RollingQuantile,
    ExpandingStd,
    SeasonalRollingMean,
    RollingStd,
    ExpandingMean
)
from mlforecast.target_transforms import (
    Differences,
    LocalBoxCox,
    LocalStandardScaler,
    LocalRobustScaler,
    AutoDifferences,
    AutoSeasonalDifferences,
    AutoSeasonalityAndDifferences,
    GlobalSklearnTransformer
)
from sklearn.preprocessing import FunctionTransformer
from .utils_models import *  



# def optimize_model_zone_r(
#     df_train, df_test, zone, target_col, target_col_name, h, base_path, n_trials, verbose,
#     exog_cols=None
# ):
#     """
#     Sélectionne le meilleur modèle, fait le tuning Optuna,
#     et enregistre directement le modèle du meilleur trial Optuna (.pkl) + ses hyperparams (.json)
#     SANS refit final.
#     """
#     import numpy as np
#     import random
#     import os
#     import optuna
#     import joblib
#     import json
#     import logging

#     logging.basicConfig(
#     level=logging.INFO,  # Ou DEBUG si tu veux plus de détails
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

#     RANDOM_STATE = 42
#     np.random.seed(RANDOM_STATE)
#     random.seed(RANDOM_STATE)
#     sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

#     df_train = df_train[df_train['unique_id'] == zone].copy()
#     df_test = df_test[df_test['unique_id'] == zone].copy()

#     MODELS = {
#         'lgbm': lambda **kwargs: LGBMRegressor(**kwargs),
#         'xgb': lambda **kwargs: XGBRegressor(**kwargs),
#         'cat': lambda **kwargs: CatBoostRegressor(**kwargs)
#     }
#     lags = list(range(1, 25))
#     #sqrt_transform = FunctionTransformer(np.sqrt, validate=True)    
#     lag_transforms = {lag: [RollingMean(window_size=lag), ExponentiallyWeightedMean(alpha=0.9), RollingQuantile(window_size=lag, p=0.25), RollingQuantile(window_size=lag, p=0.75)] for lag in lags}
#     target_transforms = [
#         AutoDifferences(max_diffs=2),
#         #GlobalSklearnTransformer(sqrt_transform),
#         LocalStandardScaler()
#     ]

#     if exog_cols is None:
#         exog_cols = [col for col in df_train.columns if col not in ['unique_id', 'ds', 'y']]

#     df_train_proc, df_test_proc, final_exog_cols, ohe_dict, categorical_cols = preprocess_exog_and_target2(df_train, df_test, exog_cols)

#     preds_dict = {}
#     for model_name, model_fn in MODELS.items():
#         if verbose:
#             logging.info(f"\nPremier fit simple pour {model_name.upper()} (sans tuning Optuna)...")
#         if model_name == 'cat':
#             used_train = df_train.copy()
#             used_test = df_test.copy()
#             used_exog_cols = exog_cols
#         else:
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()
#             used_exog_cols = final_exog_cols

#         if model_name == 'cat':
#             cat_features = categorical_cols
#             model = model_fn(verbose=0, cat_features=cat_features)
#         elif model_name == 'lgbm':
#             model = model_fn(verbosity=-1)
#         elif model_name == 'xgb':
#             model = model_fn(verbosity=0)
#         else:
#             model = model_fn()

#         forecast = MLForecast(
#             models=[model],
#             freq='W-MON',
#             lags=lags,
#             lag_transforms=lag_transforms,
#             target_transforms=target_transforms,
#             date_features=['month', 'quarter']
#         )
#         forecast.fit(
#             used_train,
#             id_col='unique_id',
#             time_col='ds',
#             target_col='y',
#             static_features=[]
#         )
#         y_pred_scaled = forecast.predict(h=h, X_df=used_test)
#         pred_cols = [col for col in y_pred_scaled.columns if col not in ['unique_id', 'ds']]
#         if not pred_cols:
#             raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled.columns}")
#         y_pred_model_col = y_pred_scaled[pred_cols[0]].values[:h]
#         y_pred = y_pred_model_col
#         preds_dict[model_name] = y_pred

#     y_true = df_test['y'].values[:h]
#     best_model, best_model_metrics = select_best_model_by_vote(preds_dict, y_true, verbose=verbose)

#     if verbose:
#         logging.info(f"\nOptimisation Optuna sur {best_model.upper()} ...")

#     best_forecast = [None]  # mutable container to store best model
#     def objective(trial):
#         config = get_optimization_config(best_model, trial)
#         config['random_state'] = RANDOM_STATE
#         if best_model == 'cat':
#             config['verbose'] = 0
#             used_train = df_train.copy()
#             used_test = df_test.copy()
#         elif best_model == 'lgbm':
#             config['verbosity'] = -1
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()
#         elif best_model == 'xgb':
#             config['verbosity'] = 0
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()
#         else:
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()

#         if best_model == 'cat':
#             cat_features = categorical_cols
#             model = MODELS[best_model](**config, cat_features=cat_features)
#         else:
#             model = MODELS[best_model](**config)

#         forecast = MLForecast(
#             models=[model],
#             freq='W-MON',
#             lags=lags,
#             lag_transforms=lag_transforms,
#             target_transforms=target_transforms
#         )
#         try:
#             forecast.fit(
#                 used_train,
#                 id_col='unique_id',
#                 time_col='ds',
#                 target_col='y',
#                 static_features=[],
#                 prediction_intervals= PredictionIntervals(n_windows=4, h=h)
#             )
#             y_pred_scaled = forecast.predict(h=h, X_df=used_test)
#             pred_cols = [col for col in y_pred_scaled.columns if col not in ['unique_id', 'ds']]
#             if not pred_cols:
#                 raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled.columns}")
#             y_pred_model_col = y_pred_scaled[pred_cols[0]].values[:h]
#             y_pred = y_pred_model_col
#             score = mean_squared_error(y_true, y_pred, squared=False)
#             # si trial est le best trial, on sauvegarde le modèle
#             if not hasattr(objective, "best_score") or score < objective.best_score:
#                 objective.best_score = score
#                 best_forecast[0] = forecast
#             return score
#         except Exception as e:
#             logging.error(f"Erreur dans objective Optuna {best_model}: {e}")
#             return float('inf')

#     study = optuna.create_study(direction='minimize', sampler=sampler)
#     early_stop = PatienceEarlyStopper(patience=12)
#     study.optimize(objective, n_trials=n_trials, callbacks=[early_stop], show_progress_bar=False)
#     if verbose:
#         logging.info(f"Best RMSE {best_model}: {study.best_value:.4f}")

#     best_config = get_optimization_config(best_model, study.best_trial)
#     best_config['random_state'] = RANDOM_STATE

#     save_dir = f"{base_path}/models_{target_col_name}_{zone}/"
#     os.makedirs(save_dir, exist_ok=True)
#     if verbose:
#         logging.info(f"Sauvegarde: {save_dir}")

#     # Sauvegarde du modèle du meilleur trial Optuna 
#     best_forecast[0].save(save_dir)
#     #joblib.dump(best_forecast[0], os.path.join(save_dir, f"{best_model}_mlforecast.pkl"))
#     # Sauvegarde des meilleurs paramètres
#     with open(f"{save_dir}/best_params.json", "w") as f:
#         json.dump(best_config, f)

#     return {
#         'zone': zone,
#         'best_model': best_model,
#         'best_params': best_config,
#         'save_dir': save_dir
#     }


def optimize_model_zone_recursive(
    df_train, df_test, zone, target_col, target_col_name, h, base_path, n_trials, verbose,
    exog_cols=None
):
    """
    Optimise et sélectionne le meilleur modèle pour une zone donnée via Optuna, en utilisant une évaluation récursive sur le test.
    Sauvegarde le modèle du meilleur essai Optuna (sans refit final) ainsi que ses hyperparamètres.
    À chaque étape du test, les prédictions sont réinjectées dans l’historique pour simuler un vrai scénario auto-régressif.
    
    Paramètres
    ----------
    df_train : pandas.DataFrame
        Données d’apprentissage (pour la zone).
    df_test : pandas.DataFrame
        Données de test (pour la zone).
    zone : str
        Identifiant de la zone.
    target_col : str
        Nom de la colonne cible.
    target_col_name : str
        Nom de la cible pour le dossier modèle.
    h : int
        Horizon de prévision.
    base_path : str
        Chemin du dossier des modèles.
    n_trials : int
        Nombre d’essais Optuna.
    verbose : bool/int
        Affiche les logs détaillés.
    exog_cols : list, optionnel
        Variables exogènes à utiliser (None = auto).

    Retour
    ------
    dict
        Dictionnaire avec la zone, le meilleur modèle, les hyperparamètres et le chemin de sauvegarde.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

    df_train = df_train[df_train['unique_id'] == zone].copy()
    df_test = df_test[df_test['unique_id'] == zone].copy()

    MODELS = {
        'lgbm': lambda **kwargs: LGBMRegressor(**kwargs),
        'xgb': lambda **kwargs: XGBRegressor(**kwargs),
        'cat': lambda **kwargs: CatBoostRegressor(**kwargs)
    }
    lags = list(range(1, 25))
    lag_transforms = {
        lag: [
            RollingMean(window_size=lag),
            ExponentiallyWeightedMean(alpha=0.9),
            RollingQuantile(window_size=lag, p=0.25),
            RollingQuantile(window_size=lag, p=0.75)
        ] for lag in lags
    }
    target_transforms = [
        AutoDifferences(max_diffs=2),
        LocalStandardScaler()
    ]

    if exog_cols is None:
        exog_cols = [col for col in df_train.columns if col not in ['unique_id', 'ds', 'y']]

    df_train_proc, df_test_proc, final_exog_cols, ohe_dict, categorical_cols = preprocess_exog_and_target2(df_train, df_test, exog_cols)

    preds_dict = {}
    for model_name, model_fn in MODELS.items():
        if verbose:
            logging.info(f"\nPremier fit simple pour {model_name.upper()} (sans tuning Optuna)...")
        if model_name == 'cat':
            used_train = df_train.copy()
            used_test = df_test.copy()
            used_exog_cols = exog_cols
        else:
            used_train = df_train_proc.copy()
            used_test = df_test_proc.copy()
            used_exog_cols = final_exog_cols

        if model_name == 'cat':
            cat_features = categorical_cols
            model = model_fn(verbose=0, cat_features=cat_features)
        elif model_name == 'lgbm':
            model = model_fn(verbosity=-1)
        elif model_name == 'xgb':
            model = model_fn(verbosity=0)
        else:
            model = model_fn()

        forecast = MLForecast(
            models=[model],
            freq='W-MON',
            lags=lags,
            lag_transforms=lag_transforms,
            target_transforms=target_transforms,
            date_features=['month', 'quarter']
        )
        forecast.fit(
            used_train,
            id_col='unique_id',
            time_col='ds',
            target_col='y',
            static_features=[]
        )
        y_pred_scaled = forecast.predict(h=h, X_df=used_test)
        pred_cols = [col for col in y_pred_scaled.columns if col not in ['unique_id', 'ds']]
        if not pred_cols:
            raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled.columns}")
        y_pred_model_col = y_pred_scaled[pred_cols[0]].values[:h]
        y_pred = y_pred_model_col
        preds_dict[model_name] = y_pred

    y_true = df_test['y'].values[:h]
    best_model, best_model_metrics = select_best_model_by_vote(preds_dict, y_true, verbose=verbose)

    if verbose:
        logging.info(f"\nOptimisation Optuna sur {best_model.upper()} ...")

    best_forecast = [None]  # mutable container to store best model

    def objective(trial):
        """
        Fonction objectif pour l’optimisation d’hyperparamètres avec Optuna appliquée à différents modèles de
        prévision (CatBoost, LightGBM, XGBoost, etc.), utilisant une stratégie de prédiction récursive sur la fenêtre de test.        
        Cette fonction prépare la configuration du modèle selon le type de modèle choisi (`best_model`), instancie le modèle 
        avec les hyperparamètres proposés par Optuna (`trial`), puis entraîne ce modèle à l’aide de MLForecast sur les données d’entraînement. 
        L’évaluation se fait de manière récursive : à chaque pas sur la fenêtre de test, la prédiction précédente est réinjectée dans les données servant 
        d’historique pour la prédiction suivante, simulant ainsi un usage en production où seules les sorties du modèle sont disponibles pour les pas futurs.

        Le score retourné à Optuna correspond à la racine de l’erreur quadratique moyenne (RMSE) calculée sur l’ensemble des prédictions de la fenêtre de test.

        Si ce score est le meilleur observé jusqu’à présent, le modèle associé est sauvegardé dans la variable globale best_forecast (liste d’un élément).

        En cas d’erreur lors du processus, la fonction lève une exception (qui doit être gérée par Optuna ou le code appelant).

        Paramètres
        ----------
        trial : optuna.trial.Trial
            Objet trial Optuna, utilisé pour générer et enregistrer une configuration d’hyperparamètres à tester.

        Variables globales utilisées
        ----------------------------
        - best_model : str
            Nom du modèle à optimiser ('cat', 'lgbm', 'xgb', etc.)
        - df_train, df_test : pandas.DataFrame
            Jeux de données bruts (pour CatBoost)
        - df_train_proc, df_test_proc : pandas.DataFrame
            Jeux de données prétraités (pour LightGBM, XGBoost et autres)
        - exog_cols, final_exog_cols : list of str
            Liste des variables exogènes à inclure pour chaque type de modèle
        - categorical_cols : list of str
            Liste des colonnes catégorielles (pour CatBoost)
        - MODELS : dict
            Dictionnaire associant le nom du modèle à la classe correspondante (ex: {'cat': CatBoostRegressor, ...})
        - lags, lag_transforms, target_transforms : paramètres pour MLForecast
        - RANDOM_STATE : int
            Graine de reproductibilité
        - h : int
            Horizon de prévision
        - best_forecast : list
            Liste à un élément permettant de stocker le meilleur objet forecast trouvé

        Retour
        ------
        float
            Valeur du RMSE sur la fenêtre de test (plus la valeur est basse, meilleur est le modèle).

        Notes
        -----
        - La fonction suppose que toutes les variables globales nécessaires sont définies et accessibles.
        - L’évaluation récursive simule le contexte opérationnel réel où les prédictions sont réinjectées au fil de l’eau, contrairement à une simple évaluation statique.
        - La sauvegarde du meilleur modèle trouvé s’effectue via une variable globale mutable (best_forecast).
        - La fonction est conçue pour être utilisée comme objectif dans une étude Optuna (optimisation automatique d’hyperparamètres).
        - Toute erreur d’exécution non prévue sera propagée à l’appelant (pas de gestion d’exception locale).
    
        """
        # ================== Configuration modèle ==================
        config = get_optimization_config(best_model, trial)
        config['random_state'] = RANDOM_STATE
        if best_model == 'cat':
            config['verbose'] = 0
            used_train = df_train.copy()
            used_test = df_test.copy()
            used_exog_cols = exog_cols
        elif best_model == 'lgbm':
            config['verbosity'] = -1
            used_train = df_train_proc.copy()
            used_test = df_test_proc.copy()
            used_exog_cols = final_exog_cols
        elif best_model == 'xgb':
            config['verbosity'] = 0
            used_train = df_train_proc.copy()
            used_test = df_test_proc.copy()
            used_exog_cols = final_exog_cols
        else:
            used_train = df_train_proc.copy()
            used_test = df_test_proc.copy()
            used_exog_cols = final_exog_cols

        if best_model == 'cat':
            cat_features = categorical_cols
            model = MODELS[best_model](**config, cat_features=cat_features)
        else:
            model = MODELS[best_model](**config)

        forecast = MLForecast(
            models=[model],
            freq='W-MON',
            lags=lags,
            lag_transforms=lag_transforms,
            target_transforms=target_transforms
        )

        try:
            if h<= 11:
                n_windows_intervals = 2
                forecast.fit(
                    used_train,
                    id_col='unique_id',
                    time_col='ds',
                    target_col='y',
                    static_features=[],
                    prediction_intervals=PredictionIntervals(n_windows=n_windows_intervals, h=h)
                )
            else:
                forecast.fit(
                    used_train,
                    id_col='unique_id',
                    time_col='ds',
                    target_col='y',
                    static_features=[]
                )


            # ----------- Évaluation récursive -----------
            n_test = len(used_test)
            n_train = len(used_train)

            # train + test réel
            df_truth_all = pd.concat([used_train, used_test], ignore_index=True)

            # Copie de travail dont on remplacera y dans la partie test au fur et à mesure
            working_test = used_test.copy()  
            df_working_all = pd.concat([used_train, working_test], ignore_index=True)  

            errors = []

            for i in range(n_test):
                # Historique utilisé: train + test déjà prédit (avec y potentiellement remplacé)
                update_end = n_train + i
                if update_end > n_train:  # il y a au moins une ligne test déjà prédite
                    forecast.update(df_working_all.iloc[:update_end])  
                else:
                    forecast.update(df_working_all.iloc[:n_train])  # première itération

                # Ligne à prédire 
                X_pred = df_working_all.iloc[[n_train + i]][['unique_id', 'ds'] + used_exog_cols]
                y_pred_df = forecast.predict(h=1, X_df=X_pred)
                pred_cols = [col for col in y_pred_df.columns if col not in ['unique_id', 'ds']]
                if not pred_cols:
                    raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_df.columns}")
                y_pred_val = y_pred_df[pred_cols[0]].values[0]

                # Vraie valeur (depuis df_truth_all)
                y_true_val = df_truth_all.iloc[n_train + i]['y']
                errors.append((y_true_val - y_pred_val) ** 2) # pour calculer le rmse

                # Injection récursive: remplacer y dans df_working_all pour ce point avant la prochaine boucle
                df_working_all.at[n_train + i, 'y'] = y_pred_val  

            score = np.sqrt(np.mean(errors))
            # Sauvegarde du meilleur modèle (inchangé)
            if not hasattr(objective, "best_score") or score < objective.best_score:
                objective.best_score = score
                best_forecast[0] = forecast
            return score

        except Exception as e:
            logging.error(f"Erreur dans objective Optuna {best_model}: {e}")
            return float('inf')

    study = optuna.create_study(direction='minimize', sampler=sampler)
    early_stop = PatienceEarlyStopper(patience=12)
    study.optimize(objective, n_trials=n_trials, callbacks=[early_stop], show_progress_bar=False)
    if verbose:
        logging.info(f"Best RMSE {best_model}: {study.best_value:.4f}")

    best_config = get_optimization_config(best_model, study.best_trial)
    best_config['random_state'] = RANDOM_STATE

    save_dir = f"{base_path}_{h}/models_{target_col_name}_{zone}/"
    os.makedirs(save_dir, exist_ok=True)
    if verbose:
        logging.info(f"Sauvegarde: {save_dir}")

    # Sauvegarde du modèle du meilleur trial Optuna
    best_forecast[0].save(save_dir)

    with open(f"{save_dir}/best_params.json", "w") as f:
        json.dump(best_config, f)

    return {
        'zone': zone,
        'best_model': best_model,
        'best_params': best_config,
        'save_dir': save_dir
    }



def optimize_model_zone(
    df_train, df_test, zone, target_col, target_col_name, h, base_path, n_trials, verbose,
    exog_cols=None
):
    """
    Sélectionne et optimise le meilleur modèle de prévision pour une zone donnée à l'aide d'Optuna.
    Cette fonction :
            - Sélectionne le meilleur modèle parmi LGBM, XGBoost et CatBoost via un premier fit simple.
            - Effectue un tuning des hyperparamètres du meilleur modèle avec Optuna, sans refit final.
            - Évalue les modèles sur la fenêtre de test via une évaluation rolling.
            - Sauvegarde le modèle du meilleur trial Optuna ainsi que ses hyperparamètres.
            - Gère automatiquement le prétraitement des variables exogènes et des variables catégorielles.
            Paramètres
            ----------
            df_train : pd.DataFrame
                Données d'entraînement contenant au moins les colonnes 'unique_id', 'ds', 'y' et les exogènes.
            df_test : pd.DataFrame
                Données de test contenant la même structure que df_train.
            zone : str ou int
                Identifiant de la zone à traiter (valeur de 'unique_id').
            target_col : str
                Nom de la colonne cible (souvent 'y').
            target_col_name : str
                Nom utilisé pour sauvegarder les modèles/paramètres.
            h : int
                Horizon de prévision (nombre de périodes à prédire).
            base_path : str
                Chemin de base pour la sauvegarde des modèles et paramètres.
            n_trials : int
                Nombre d'itérations pour l'optimisation Optuna.
            verbose : bool
                Si True, affiche des logs détaillés.
            exog_cols : list, optional
                Liste des colonnes exogènes à utiliser. Si None, toutes les colonnes hors ['unique_id', 'ds', 'y'] sont utilisées.
            Retourne
            -------
            dict
                Un dictionnaire contenant :
                - 'zone' : identifiant de la zone traitée,
                - 'best_model' : nom du meilleur modèle sélectionné,
                - 'best_params' : dictionnaire des meilleurs hyperparamètres,
                - 'save_dir' : chemin du dossier de sauvegarde du modèle et des paramètres.
            Notes
            -----
            - Le modèle sauvegardé correspond au meilleur trial Optuna, sans refit final sur l'ensemble train+test.
            - Les paramètres sont sauvegardés au format JSON.
            - Les modèles sont évalués via une stratégie rolling sur la fenêtre de test.
    """


    logging.basicConfig(
        level=logging.INFO,  # Ou DEBUG si tu veux plus de détails
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

    df_train = df_train[df_train['unique_id'] == zone].copy()
    df_test = df_test[df_test['unique_id'] == zone].copy()

    MODELS = {
        'lgbm': lambda **kwargs: LGBMRegressor(**kwargs),
        'xgb': lambda **kwargs: XGBRegressor(**kwargs),
        'cat': lambda **kwargs: CatBoostRegressor(**kwargs)
    }
    lags = list(range(1, 25))   
    lag_transforms = {lag: [RollingMean(window_size=lag), ExponentiallyWeightedMean(alpha=0.9), RollingQuantile(window_size=lag, p=0.25), RollingQuantile(window_size=lag, p=0.75)] for lag in lags}
    target_transforms = [
        AutoDifferences(max_diffs=2),
        LocalStandardScaler()
    ]

    if exog_cols is None:
        exog_cols = [col for col in df_train.columns if col not in ['unique_id', 'ds', 'y']]

    df_train_proc, df_test_proc, final_exog_cols, ohe_dict, categorical_cols = preprocess_exog_and_target2(df_train, df_test, exog_cols)

    preds_dict = {}
    for model_name, model_fn in MODELS.items():
        if verbose:
            logging.info(f"\nPremier fit simple pour {model_name.upper()} (sans tuning Optuna)...")
        if model_name == 'cat':
            used_train = df_train.copy()
            used_test = df_test.copy()
            used_exog_cols = exog_cols
        else:
            used_train = df_train_proc.copy()
            used_test = df_test_proc.copy()
            used_exog_cols = final_exog_cols

        if model_name == 'cat':
            cat_features = categorical_cols
            model = model_fn(verbose=0, cat_features=cat_features)
        elif model_name == 'lgbm':
            model = model_fn(verbosity=-1)
        elif model_name == 'xgb':
            model = model_fn(verbosity=0)
        else:
            model = model_fn()

        forecast = MLForecast(
            models=[model],
            freq='W-MON',
            lags=lags,
            lag_transforms=lag_transforms,
            target_transforms=target_transforms,
            date_features=['month', 'quarter']
        )
        forecast.fit(
            used_train,
            id_col='unique_id',
            time_col='ds',
            target_col='y',
            static_features=[]
        )
        y_pred_scaled = forecast.predict(h=h, X_df=used_test)
        pred_cols = [col for col in y_pred_scaled.columns if col not in ['unique_id', 'ds']]
        if not pred_cols:
            raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled.columns}")
        y_pred_model_col = y_pred_scaled[pred_cols[0]].values[:h]
        y_pred = y_pred_model_col
        preds_dict[model_name] = y_pred

    y_true = df_test['y'].values[:h]
    best_model, best_model_metrics = select_best_model_by_vote(preds_dict, y_true, verbose=verbose)

    if verbose:
        logging.info(f"\nOptimisation Optuna sur {best_model.upper()} ...")

    best_forecast = [None] 

    def objective2(trial):
        """
        Fonction objectif pour l’optimisation d’hyperparamètres avec Optuna sur différents modèles de prévision (CatBoost, LightGBM, XGBoost).

        Cette fonction construit dynamiquement la configuration du modèle selon le type de modèle sélectionné (`best_model`), 
        instancie le modèle avec les hyperparamètres proposés par Optuna (`trial`), puis entraîne ce modèle à l’aide de MLForecast 
        sur les données d’entraînement fournies. Elle effectue une évaluation rolling (pas à pas en re intégrant la vrai valeur prise par la série dans le train) 
        sur l’ensemble de la fenêtre de test,en recalculant la prévision à chaque pas avec mise à jour incrémentale des données observées, ce qui permet d’estimer 
        la performance du modèle de façon réaliste (type walk-forward).

        Le score retourné à Optuna correspond à la racine de l’erreur quadratique moyenne (RMSE) calculée sur toutes les prévisions 
        de la fenêtre de test.

        Si ce score est le plus bas observé jusqu’ici, la fonction sauvegarde l’objet forecast dans la variable globale best_forecast 
        (type: liste d’un élément, pour mutabilité).

        En cas d’erreur lors du processus, la fonction journalise le problème et retourne une valeur de pénalité infinie 
        (float('inf')) pour signaler l’échec au moteur d’optimisation Optuna.

        Paramètres
        ----------
        trial : optuna.trial.Trial
            Objet trial Optuna, utilisé pour générer et enregistrer une configuration d’hyperparamètres à tester.

        Variables globales utilisées
        ----------------------------
        - best_model : str
            Nom du modèle à optimiser ('cat', 'lgbm', 'xgb', etc.)
        - df_train, df_test : pandas.DataFrame
            Jeux de données bruts (pour CatBoost)
        - df_train_proc, df_test_proc : pandas.DataFrame
            Jeux de données prétraités (pour LightGBM, XGBoost et autres)
        - exog_cols, final_exog_cols : list of str
            Liste des variables exogènes à inclure pour chaque type de modèle
        - categorical_cols : list of str
            Liste des colonnes catégorielles (pour CatBoost)
        - MODELS : dict
            Dictionnaire associant le nom du modèle à la classe correspondante (ex: {'cat': CatBoostRegressor, ...})
        - lags, lag_transforms, target_transforms : paramètres pour MLForecast
        - RANDOM_STATE : int
            Graine de reproductibilité
        - h : int
            Horizon de prévision
        - best_forecast : list
            Liste à un élément permettant de stocker le meilleur objet forecast trouvé

        Retour
        ------
        float
            Valeur du RMSE sur la fenêtre de test (plus la valeur est basse, meilleur est le modèle).
            Si une erreur survient, retourne float('inf').

        Exceptions
        ----------
        Toute erreur d’exécution pendant l’entraînement ou la prédiction est interceptée, journalisée, 
        et la fonction retourne float('inf') pour signaler l’échec à Optuna.

        Notes
        -----
        - La fonction suppose que toutes les variables globales nécessaires sont définies et accessibles.
        - Le rolling update sur la fenêtre de test simule un contexte de prévision incrémentale.
        - La sauvegarde du meilleur modèle trouvé s’effectue via une variable globale mutable (best_forecast).
        - La fonction est conçue pour être utilisée comme objectif dans une étude Optuna (optimisation automatique d’hyperparamètres).
    """
        config = get_optimization_config(best_model, trial)
        config['random_state'] = RANDOM_STATE
        if best_model == 'cat':
            config['verbose'] = 0
            used_train = df_train.copy()
            used_test = df_test.copy()
            used_exog_cols = exog_cols
        elif best_model == 'lgbm':
            config['verbosity'] = -1
            used_train = df_train_proc.copy()
            used_test = df_test_proc.copy()
            used_exog_cols = final_exog_cols
        elif best_model == 'xgb':
            config['verbosity'] = 0
            used_train = df_train_proc.copy()
            used_test = df_test_proc.copy()
            used_exog_cols = final_exog_cols
        else:
            used_train = df_train_proc.copy()
            used_test = df_test_proc.copy()
            used_exog_cols = final_exog_cols

        if best_model == 'cat':
            cat_features = categorical_cols
            model = MODELS[best_model](**config, cat_features=cat_features)
        else:
            model = MODELS[best_model](**config)

        forecast = MLForecast(
            models=[model],
            freq='W-MON',
            lags=lags,
            lag_transforms=lag_transforms,
            target_transforms=target_transforms
        )
        try:
            forecast.fit(
                used_train,
                id_col='unique_id',
                time_col='ds',
                target_col='y',
                static_features=[],
                prediction_intervals= PredictionIntervals(n_windows=4, h=h)
            )

            # ----------- Rolling evaluation sur toute la fenêtre de test -----------
            n_test = len(used_test)
            errors = []
            # Concatène train + test pour faciliter l'accès
            df_all = pd.concat([used_train, used_test], ignore_index=True)
            n_train = len(used_train)
            for i in range(n_test):
                idx_update = list(range(n_train + i))  # train + ce qu'on a déjà vu du test
                forecast.update(df_all.iloc[idx_update])

                X_pred = df_all.iloc[[n_train + i]][['unique_id', 'ds'] + used_exog_cols]
                y_pred_df = forecast.predict(h=1, X_df=X_pred)
                pred_cols = [col for col in y_pred_df.columns if col not in ['unique_id', 'ds']]
                if not pred_cols:
                    raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_df.columns}")
                y_pred = y_pred_df[pred_cols[0]].values[0]
                y_true = df_all.iloc[n_train + i]['y']

                errors.append((y_true - y_pred) ** 2)  # RMSE

            score = np.sqrt(np.mean(errors))
            # Si trial est le best trial, on sauvegarde le modèle
            if not hasattr(objective2, "best_score") or score < objective2.best_score:
                objective2.best_score = score
                best_forecast[0] = forecast
            return score
        except Exception as e:
            logging.error(f"Erreur dans objective Optuna {best_model}: {e}")
            return float('inf')

    study = optuna.create_study(direction='minimize', sampler=sampler)
    early_stop = PatienceEarlyStopper(patience=12)
    study.optimize(objective2, n_trials=n_trials, callbacks=[early_stop], show_progress_bar=False)
    if verbose:
        logging.info(f"Best RMSE {best_model}: {study.best_value:.4f}")

    best_config = get_optimization_config(best_model, study.best_trial)
    best_config['random_state'] = RANDOM_STATE

    save_dir = f"{base_path}/models_{target_col_name}_{zone}/"
    os.makedirs(save_dir, exist_ok=True)
    if verbose:
        logging.info(f"Sauvegarde: {save_dir}")

    # Sauvegarde du modèle du meilleur trial Optuna 
    best_forecast[0].save(save_dir)

    # Sauvegarde des meilleurs paramètres
    with open(f"{save_dir}/best_params.json", "w") as f:
        json.dump(best_config, f)

    return {
        'zone': zone,
        'best_model': best_model,
        'best_params': best_config,
        'save_dir': save_dir
    }


def optimize_models_for_all_zones(
    df_train, df_test, target_col_name , target_col='y',  h=12, 
    base_path="./models", n_trials=20, verbose=True
):
    """
    Optimise et sauvegarde un modèle de prévision pour chaque zone distincte identifiée dans les données d'entraînement. 
    Pour chaque zone (champ 'unique_id'), la fonction effectue une optimisation des hyperparamètres à l'aide de la fonction optimize_model_zone_recursive, 
    puis sauvegarde le meilleur modèle obtenu ainsi que sa configuration (généralement au format .pkl et .json) dans le dossier précisé.

    Un suivi du déroulement (succès/échecs par zone) est affiché si verbose=True. 
    Les résultats détaillés de l’optimisation pour chaque zone sont retournés dans un dictionnaire, 
    où la clé correspond à l’identifiant de la zone et la valeur au résultat de l’optimisation.

    Paramètres
    ----------
    df_train : pandas.DataFrame
        Données d’entraînement contenant au moins les colonnes 'unique_id' et la colonne cible.
    df_test : pandas.DataFrame
        Données de test structurées comme df_train, utilisées pour l’évaluation.
    target_col_name : str
        Nom lisible de la variable cible (utilisé pour la journalisation ou la sauvegarde).
    target_col : str, optionnel (défaut='y')
        Nom de la colonne cible à prédire.
    h : int, optionnel (défaut=12)
        Horizon de prévision (nombre de pas à prédire).
    base_path : str, optionnel (défaut="./models")
        Répertoire de sauvegarde des modèles et fichiers associés.
    n_trials : int, optionnel (défaut=20)
        Nombre d’itérations d’optimisation (essais d’hyperparamètres) par zone.
    verbose : bool, optionnel (défaut=True)
        Active l’affichage détaillé du suivi de l’optimisation.

    Retour
    ------
    dict
        Dictionnaire associant chaque zone à son résultat d’optimisation (meilleur modèle, 
        hyperparamètres, métriques, chemins des fichiers sauvegardés, etc.).

    Notes
    -----
    Les erreurs rencontrées lors de l’optimisation d’une zone sont interceptées, 
    journalisées et n’interrompent pas le traitement des autres zones. 
    Les zones échouées ne figurent pas dans le dictionnaire de sortie.
    La fonction suppose que optimize_model_zone_recursive est déjà implémentée et accessible.
    
    """
    import logging
    logging.basicConfig(
    level=logging.INFO,  # Ou DEBUG si tu veux plus de détails
    format="%(asctime)s - %(levelname)s - %(message)s"
)
    
    if verbose:
        logging.info("==== OPTIMISATION DES MODÈLES MLFORECAST PAR ZONE ====")
    
    zones_uniques = df_train['unique_id'].unique()
    resultats_toutes_zones = {}
    zones_reussies, zones_echecs = 0, 0
    
    for i, zone in enumerate(zones_uniques):
        if verbose:
            logging.info(f"==== Optimisation pour la zone : {zone} ====")
            
        try:
            df_train_zone = df_train[df_train['unique_id'] == zone].copy()
            df_test_zone = df_test[df_test['unique_id'] == zone].copy()
            
            resultats_zone = optimize_model_zone_recursive(
                df_train=df_train_zone,
                df_test=df_test_zone,
                zone=zone, # "Banasud"
                target_col=target_col,
                target_col_name=target_col_name,
                h=h,
                base_path=base_path,
                n_trials=n_trials,
                verbose=verbose
            )
            resultats_toutes_zones[zone] = resultats_zone
            zones_reussies += 1
            if verbose:
                logging.info(f"SUCCÈS - Meilleur: {resultats_zone['best_model']}")
        except Exception as e:
            zones_echecs += 1
            if verbose:
                logging.error(f"ÉCHEC: {e}")
            continue
    
    if verbose:
        logging.info(f"\n{'='*60}\nOptimisation terminée !\n{'='*60}")
        logging.info(f"Zones traitées: {len(zones_uniques)} | Succès: {zones_reussies} | Échecs: {zones_echecs}")
    return resultats_toutes_zones



def train_zone_pred(
    df_train,
    zone,
    base_path,
    target_col_name,
    h,
    exog_cols=None,
    window_size=4,
    verbose=1,
    is_future=False
):
    """
    Effectue une prédiction rolling (fenêtre glissante en ajoutant les vrais valeurs de la série) sur les données d'entraînement pour une zone donnée, 
    en utilisant le modèle sauvegardé lors de l'optimisation.Respecte l'encodage et la logique initiale du modèle.
    Le paramètre is_future permet de différencier et sauvegarder les prédictions pour des périodes futures sans écraser les fichiers existants.

    Paramètres
    ----------
    df_train : pandas.DataFrame
        Données d’apprentissage complètes (toutes zones).
    zone : str
        Nom ou identifiant de la zone à traiter.
    base_path : str
        Chemin vers le dossier de sauvegarde du modèle.
    target_col_name : str
        Nom de la variable cible (pour retrouver le dossier du modèle).
    exog_cols : list, optionnel
        Variables exogènes à utiliser (None = auto-détection).
    window_size : int, optionnel
        Taille de la fenêtre pour l’update et la prédiction (défaut = 4).
    verbose : int, optionnel
        Niveau d’affichage (0 = silencieux, 1 = détails).
    is_future : bool, optionnel
        Si True, sauvegarde dans un fichier séparé pour les prédictions futures.

    Retour
    ------
    pandas.DataFrame
        Résultats contenant les vraies valeurs et les prédictions pour chaque fenêtre.
   
    """
    # 1. Chargement du modèle optimisé et de ses paramètres
    save_dir = os.path.join(
        f"{base_path}_{h}", f"models_{target_col_name}_{zone}"
    )
    if verbose:
        print(f"Chargement du modèle depuis {save_dir}")
    with open(os.path.join(save_dir, "best_params.json"), "r") as f:
        best_params = json.load(f)
    
    # On charge le MLForecast sauvegardé (ceci inclut le modèle entraîné)
    fcst = MLForecast.load(save_dir)
    
    # 2. Extraction des infos sur le modèle
    model_key = list(fcst.models.keys())[0]
    model = fcst.models[model_key]
    model_name = model_key.lower()

    if "catboost" in model_name:
        model_type = "cat"
    elif "lgbm" in model_name:
        model_type = "lgbm"
    elif "xgb" in model_name:
        model_type = "xgb"
    else:
        raise ValueError(f"Modèle non reconnu : {model_name}")

    # 3. Sélection des données de la zone
    df = df_train[df_train['unique_id'] == zone].copy()

    ds_to_add = pd.Timestamp("2024-12-30")
    if not (df["ds"] == ds_to_add).any():
        prev_row = df[df["ds"] == pd.Timestamp("2024-12-23")]
        if not prev_row.empty:
            # Crée une nouvelle ligne avec toutes les colonnes d'origine (y compris exogènes)
            new_row = pd.DataFrame({col: prev_row.iloc[0][col] if col in prev_row.columns else np.nan for col in df.columns}, index=[0])
            new_row["ds"] = ds_to_add
            iso = ds_to_add.isocalendar()
            if "Annee" in df.columns:
                new_row["Annee"] = iso.year
            if "Semaine" in df.columns:
                new_row["Semaine"] = iso.week
            df = pd.concat(
                [
                    df[df["ds"] <= pd.Timestamp("2024-12-23")],
                    new_row,
                    df[df["ds"] > pd.Timestamp("2024-12-23")]
                ]
            ).reset_index(drop=True)

    # 4. Préparation des colonnes exogènes
    # On doit utiliser exactement le même encodage qu'au fit initial
    if exog_cols is None:
        exog_cols = [col for col in df.columns if col not in ['unique_id', 'ds', 'y']]

    # Le preprocess renvoie : train_proc, test_proc, final_exog_cols, ohe_dict, categorical_cols
    # Ici on veut appliquer l'encodage sur tout le train (test inutile)
    df_proc, _, final_exog_cols, ohe_dict, categorical_cols = preprocess_exog_and_target2(
        df, df, exog_cols
    )

    # Pour CatBoost, on garde les colonnes brutes, sinon on prend les colonnes encodées
    if model_type == "cat":
        df_used = df.copy()
        used_exog_cols = exog_cols
    else:
        df_used = df_proc.copy()
        used_exog_cols = final_exog_cols
    #print("CatBoost: Colonnes du df_used :", df_used.columns.tolist())

    # 5. Rolling update & prédiction
    start = 0
    end = window_size
    results = []
    n = len(df_used)

    while end + window_size <= n:
        df_update = df_used.iloc[start:end]
        df_pred_true = df_used.iloc[end:end+window_size]


        # update du modèle sur la fenêtre courante
        fcst.update(df_update)

        # Pour la prédiction, on doit passer explicitement les exogènes (cas XGB/LGBM)
        if model_type == "cat":
            exog_pred = df_pred_true[['unique_id', 'ds'] + used_exog_cols]
            pred_df = fcst.predict(h=window_size, X_df=exog_pred)

        else:
            exog_pred = df_pred_true[['unique_id', 'ds'] + used_exog_cols]
            pred_df = fcst.predict(h=window_size, X_df=exog_pred)

         # Repère la colonne de prédiction à chaque itération
        pred_cols = [col for col in pred_df.columns if col not in ['unique_id', 'ds']]
        if not pred_cols:
            raise ValueError("Aucune colonne de prédiction trouvée dans pred_df.columns")
        pred_colname = pred_cols[0]

        # Fusion vraies valeurs / prédiction
        merged = df_pred_true.merge(pred_df, on=["unique_id", "ds"])
        results.append(merged)

        # Slide la fenêtre
        start += window_size
        end += window_size

    # 6. Résultat final
    if not results:
        raise ValueError("Aucune fenêtre rolling n'a été générée, vérifier la taille de vos données/train.")

    df_results = pd.concat(results, ignore_index=True)
    # Renommage des colonnes pour clarté
    df_results = df_results.rename(columns={pred_colname: 'y_pred'})
    df_results = df_results[["unique_id", "ds","Annee","Semaine", "y", "y_pred"]]
    df_results['y_pred'] = np.clip(df_results['y_pred'], 0, None)
    

    # Sauvegarde du DataFrame résultat

    try:
        # Script classique
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    except NameError:
        # Notebook ou environnement interactif
        project_root = os.getcwd()
    data_save_dir = os.path.join(project_root, "data", f"models_data_{h}", f"models_{target_col_name}_{zone}")
    os.makedirs(data_save_dir, exist_ok=True)

    if is_future:
        train_filename = "train_futur.csv"
    else:
        train_filename = "train_data.csv"

    df_used.to_csv(os.path.join(data_save_dir, train_filename), index=False)

    # On enlève l'index pour avoir le meme format que le fichier predictions.csv
    df_results_to_save = df_results.reset_index(drop=True)
    
    if is_future:
        filename = "predictions_train_futur.csv"
    else:
        filename = "predictions_train.csv"
    # On sauvegarde le fichier CSV
    df_results_to_save.to_csv(os.path.join(data_save_dir, filename), index=False)

    if verbose:
        print(f"Prédictions sauvegardées dans {os.path.join(data_save_dir, filename)}")

    return df_results



def train_all_zones_pred(
    df_train,
    base_path,
    target_col_name,
    h,
    exog_cols=None,
    window_size=4,
    verbose=True,
    is_future= False
):
    """
    Applique la fonction train_zone_pred à chaque zone unique du DataFrame.  
    Pour chaque zone, effectue un rolling update et sauvegarde les prédictions dans le bon dossier, sans écraser les fichiers existants si is_future=True.

    Paramètres
    ----------
    df_train : pandas.DataFrame
        Données d’apprentissage complètes (toutes zones).
    base_path : str
        Chemin du dossier où sont sauvegardés les modèles et résultats.
    target_col_name : str
        Nom de la cible (pour retrouver le dossier du modèle).
    exog_cols : list, optionnel
        Variables exogènes à utiliser (None = auto).
    window_size : int, optionnel
        Taille de la fenêtre rolling (défaut = 4).
    verbose : bool, optionnel
        Affiche les logs étape par étape (défaut = True).
    is_future : bool, optionnel
        Si True, sauvegarde dans des fichiers séparés pour les prédictions futures.

    Retour
    ------
    dict
        Dictionnaire {zone: df_results} avec les prédictions pour chaque zone.
    
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    zones_uniques = df_train['unique_id'].unique()
    resultats_train = {}
    for i, zone in enumerate(zones_uniques):
        try:
            if verbose:
                logging.info(f"--- Rolling train pour la zone {zone} ---")
            df_results = train_zone_pred(
                df_train=df_train,
                zone=zone,
                base_path=base_path,
                target_col_name=target_col_name,
                h=h, 
                exog_cols=exog_cols,
                window_size=window_size,
                verbose=verbose,
                is_future=is_future
            )
            resultats_train[zone] = df_results
            if verbose:
                logging.info(f"Succès train {zone} | Nb lignes={len(df_results)}")
        except Exception as e:
            logging.error(f"ÉCHEC train {zone}: {e}")
            continue
    return resultats_train










def test_model_zone_ML2_rolling_test(
    df_train, df_test, zone, target_col, target_col_name, window_size, base_path, verbose,
    exog_cols=None,
    ohe_dict=None,
    categorical_cols=None,
    h=None
):
    """
    Effectue une évaluation rolling (semaine par semaine en actualisant le train à chaque étape avec la valeur y réel prise par la série pour la semaine de test prédite) 
    sur le jeu de test d’une zone avec le modèle MLForecast sauvegardé lors de l'optimisation. 
    À chaque fenêtre, le modèle est mis à jour avec toutes les vraies valeurs connues du test, puis prédit la fenêtre suivante. 
    Ne fait aucune prédiction ni mise à jour sur le train.

    Sauvegarde les prédictions, les intervalles de confiance (si disponibles) et les métriques (RMSE, MAE) dans le dossier correspondant à la zone et la cible.

    Paramètres
    ----------
    df_train : pandas.DataFrame
        Données d’apprentissage (pour l’update initiale).
    df_test : pandas.DataFrame
        Données de test à prédire (pour la zone).
    zone : str
        Identifiant de la zone à traiter.
    target_col : str
        Nom de la colonne cible.
    target_col_name : str
        Nom de la cible pour retrouver le dossier modèle.
    window_size : int
        Taille de la fenêtre rolling.
    base_path : str
        Chemin du dossier des modèles.
    verbose : bool/int
        Affiche les logs détaillés si True/1.
    exog_cols : list, optionnel
        Variables exogènes à utiliser (None = auto).
    ohe_dict : dict, optionnel
        Dictionnaire d’encodage OneHot (pour XGB/LGBM).
    categorical_cols : list, optionnel
        Colonnes catégorielles (pour CatBoost).

    Retour
    ------
    dict ou None
        Dictionnaire contenant la zone, les meilleurs hyperparamètres, les métriques, 
        le DataFrame des prédictions et le chemin de sauvegarde. 
        Retourne None en cas d’erreur.
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        save_dir = f"{base_path}_{h}/models_{target_col_name}_{zone}/"
        if not os.path.exists(save_dir):
            raise ValueError(f"Dossier {save_dir} introuvable, lance d'abord optimize_model_zone")

        # Charger le modèle MLForecast déjà entraîné
        fcst = MLForecast.load(save_dir)

        # Identifier le type de modèle
        first_model = list(fcst.models.values())[0]
        model_name = first_model.__class__.__name__.lower()
        if 'xgb' in model_name:
            main_model = 'xgb'
        elif 'lgbm' in model_name:
            main_model = 'lgbm'
        elif 'cat' in model_name:
            main_model = 'cat'
        else:
            main_model = 'other'

        # Préparation des données train/test (par zone)
        df_train_zone = df_train[df_train['unique_id'] == zone].copy()
        df_test_zone = df_test[df_test['unique_id'] == zone].copy()
        n_train = len(df_train_zone)
        n_test = len(df_test_zone)

        # Préparation des exogènes et encodage si nécessaire
        if exog_cols is None:
            exog_cols = [col for col in df_train_zone.columns if col not in ['unique_id', 'ds', 'y']]
        df_full = pd.concat([df_train_zone, df_test_zone], ignore_index=True)
        if main_model in ['xgb', 'lgbm']:
            df_full_proc, _, final_exog_cols, ohe_dict_out, categorical_cols_out = preprocess_exog_and_target2(
                df_full, df_full, exog_cols
            )
            df_used = df_full_proc
            used_exog_cols = final_exog_cols
        elif main_model == 'cat':
            df_used = df_full
            used_exog_cols = exog_cols
            if categorical_cols is None:
                _, _, _, _, categorical_cols = preprocess_exog_and_target2(df_full, df_full, exog_cols)
            cat_features = categorical_cols
        else:
            df_used = df_full
            used_exog_cols = exog_cols

        # Rolling update uniquement sur le test
        start = 0
        end = window_size
        results = []

        # On avance dans le test, fenêtre par fenêtre
        while end <= n_test:
            # Update sur TOUT le train + la partie du test déjà observée
            df_update = pd.concat([df_used.iloc[:n_train], df_used.iloc[n_train:n_train+start]], ignore_index=True)
            df_pred_true = df_used.iloc[n_train+start:n_train+end]

            fcst.update(df_update)

            exog_pred = df_pred_true[['unique_id', 'ds'] + used_exog_cols]

            if h is not None and h <= 11:
                pred_df = fcst.predict(h=window_size, X_df=exog_pred,level=[90,95])
            else:
                pred_df = fcst.predict(h=window_size, X_df=exog_pred)

            pred_cols = [col for col in pred_df.columns if col not in ['unique_id', 'ds']]
            if not pred_cols:
                raise ValueError("Aucune colonne de prédiction trouvée dans pred_df.columns")
            pred_colname = pred_cols[0]

            merged = df_pred_true.merge(pred_df, on=["unique_id", "ds"])
            results.append(merged)

            start += window_size
            end += window_size

        if not results:
            raise ValueError("Aucune fenêtre rolling n'a été générée, vérifier la taille de vos données/test.")

        df_results = pd.concat(results, ignore_index=True)
        df_results = df_results.rename(columns={pred_colname: 'y_pred'})
        df_results['y_pred'] = df_results['y_pred'].clip(lower=0)

         # ======== AJOUT INTERVALLES DE CONFIANCE IC  ==========
        # On va chercher les colonnes d'IC si elles existent, selon la classe du modèle utilisé
        model_class_name = first_model.__class__.__name__
        for lvl in [90, 95]:
            lo_col = f'{model_class_name}-lo-{lvl}'
            hi_col = f'{model_class_name}-hi-{lvl}'
            if lo_col in df_results.columns:
                df_results[lo_col] = df_results[lo_col].clip(lower=0)
            if hi_col in df_results.columns:
                df_results[hi_col] = df_results[hi_col].clip(lower=0)
            if lo_col in df_results.columns and hi_col in df_results.columns:
                df_results[f'ic_lo_{lvl}'] = df_results[lo_col]
                df_results[f'ic_hi_{lvl}'] = df_results[hi_col]

        colonnes = ["unique_id", "ds", "Annee", "Semaine", "y", "y_pred",
            "ic_lo_90", "ic_hi_90", "ic_lo_95", "ic_hi_95"]
        colonnes = [c for c in colonnes if c in df_results.columns]
        df_results = df_results[colonnes]

        # Sauvegarde
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        except NameError:
            project_root = os.getcwd()
        data_save_dir = os.path.join(project_root, "data", f"models_data_{h}", f"models_{target_col_name}_{zone}")
        os.makedirs(data_save_dir, exist_ok=True)

        df_results_to_save = df_results.reset_index(drop=True)
        df_results_to_save['model_name'] = model_class_name
        df_results_to_save.to_csv(os.path.join(data_save_dir, "predictions.csv"), index=False)

        y_true = df_results['y'].values
        y_pred = df_results['y_pred'].values
        rmse_final = mean_squared_error(y_true, y_pred,squared = False)
        mae_final = mean_absolute_error(y_true, y_pred)
        best_model_metrics = {'RMSE': rmse_final, 'MAE': mae_final}
        
        pd.DataFrame([best_model_metrics]).to_csv(os.path.join(data_save_dir, "best_metrics_rolling.csv"), index=False)

        if verbose:
            logging.info(f"Test rolling (test only) et sauvegardes terminés pour {zone} - {target_col_name}")

        best_params_path = os.path.join(save_dir, "best_params.json")
        best_config = {}
        if os.path.exists(best_params_path):
            with open(best_params_path) as f:
                best_config = json.load(f)
        
        # ==== IMPORTANCE DES VARIABLES ====
        importance_df = None
        model_obj = first_model
        if main_model == 'xgb':
            # Pour XGBRegressor
            feature_names = model_obj.get_booster().feature_names
            importances = model_obj.feature_importances_
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        elif main_model == 'lgbm':
            # Pour LGBMRegressor
            feature_names = model_obj.feature_name_
            importances = model_obj.feature_importances_
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        elif main_model == 'cat':
            # Pour CatBoostRegressor
            feature_names = model_obj.feature_names_
            importances = model_obj.get_feature_importance()
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        if importance_df is not None:
            importance_df = importance_df.sort_values(by='importance', ascending=False)
            importance_df.to_csv(os.path.join(data_save_dir, "importance.csv"), index=False)

        return {
            'zone': zone,
            'best_params': best_config,
            'best_metrics': best_model_metrics,
            'predictions': df_results,
            'mlforecast': fcst,
            'save_dir': save_dir
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"Erreur dans test_model_zone_ML2_rolling_test_only pour la zone {zone} : {repr(e)}")
        return None
    
def test_model_zone_ML2_recursive_test(
    df_train, df_test, zone, target_col, target_col_name, window_size, base_path, verbose,
    exog_cols=None,
    ohe_dict=None,
    categorical_cols=None
):
    """
    Effectue une évaluation rolling récursive sur le jeu de test d’une zone avec un modèle MLForecast sauvegardé.
    Après chaque prédiction d’une fenêtre, les valeurs prédites remplacent les vraies valeurs dans l’historique pour la suite du test (mode récursif).
    L'historique de la série est actualisé grâce à l'update qui met a jours les lags et les variables transformées.
    Les vraies valeurs sont conservées pour le calcul des métriques.

    Les résultats (prédictions, intervalles de confiance, métriques RMSE/MAE) sont sauvegardés dans le dossier de la zone et de la cible.

    Paramètres
    ----------
    df_train : pandas.DataFrame
        Données d’apprentissage (pour l’update initiale).
    df_test : pandas.DataFrame
        Données de test à prédire.
    zone : str
        Identifiant de la zone.
    target_col : str
        Nom de la colonne cible.
    target_col_name : str
        Nom de la cible (pour retrouver le dossier du modèle).
    window_size : int
        Taille de la fenêtre rolling.
    base_path : str
        Chemin du dossier des modèles.
    verbose : bool/int
        Affiche les logs si True/1.
    exog_cols : list, optionnel
        Variables exogènes à utiliser (None = auto).
    ohe_dict : dict, optionnel
        Dictionnaire d’encodage OneHot (pour XGB/LGBM).
    categorical_cols : list, optionnel
        Colonnes catégorielles (pour CatBoost).

    Retour
    ------
    dict ou None
        Dictionnaire avec la zone, les meilleurs hyperparamètres, les métriques, les prédictions et le chemin de sauvegarde.
        Retourne None en cas d’erreur.
    
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        save_dir = f"{base_path}/models_{target_col_name}_{zone}/"
        if not os.path.exists(save_dir):
            raise ValueError(f"Dossier {save_dir} introuvable, lance d'abord optimize_model_zone")

        # Charger le modèle MLForecast déjà entraîné
        fcst = MLForecast.load(save_dir)

        # Identifier le type de modèle
        first_model = list(fcst.models.values())[0]
        model_name = first_model.__class__.__name__.lower()
        if 'xgb' in model_name:
            main_model = 'xgb'
        elif 'lgbm' in model_name:
            main_model = 'lgbm'
        elif 'cat' in model_name:
            main_model = 'cat'
        else:
            main_model = 'other'

        # Préparation des données train/test (par zone)
        df_train_zone = df_train[df_train['unique_id'] == zone].copy()
        df_test_zone = df_test[df_test['unique_id'] == zone].copy()
        n_train = len(df_train_zone)
        n_test = len(df_test_zone)

        # Préparation des exogènes et encodage si nécessaire
        if exog_cols is None:
            exog_cols = [col for col in df_train_zone.columns if col not in ['unique_id', 'ds', 'y']]
        df_full = pd.concat([df_train_zone, df_test_zone], ignore_index=True)
        if main_model in ['xgb', 'lgbm']:
            df_full_proc, _, final_exog_cols, ohe_dict_out, categorical_cols_out = preprocess_exog_and_target2(
                df_full, df_full, exog_cols
            )
            df_used = df_full_proc
            used_exog_cols = final_exog_cols
        elif main_model == 'cat':
            df_used = df_full
            used_exog_cols = exog_cols
            if categorical_cols is None:
                _, _, _, _, categorical_cols = preprocess_exog_and_target2(df_full, df_full, exog_cols)
            cat_features = categorical_cols
        else:
            df_used = df_full
            used_exog_cols = exog_cols

        # Rolling update uniquement sur le test
        start = 0
        end = window_size
        results = []

        while end <= n_test:
            # Jeu d'update = train + test déjà "révélé" (avec y potentiellement remplacés par y_pred des blocs précédents)
            df_update = pd.concat(
                [df_used.iloc[:n_train], df_used.iloc[n_train:n_train+start]],
                ignore_index=True
            )

            # Fenêtre courante du test (on fait une COPIE pour garder les vraies valeurs y)
            df_pred_true = df_used.iloc[n_train+start:n_train+end].copy() 

            # Update du modèle avec l'historique (train + blocs test précédents déjà transformés)
            fcst.update(df_update)

            # Exogènes pour la prédiction du bloc courant
            exog_pred = df_pred_true[['unique_id', 'ds'] + used_exog_cols]

            # Prédiction multi-step sur la fenêtre courante
            pred_df = fcst.predict(h=window_size, X_df=exog_pred, level=[90, 95])

            pred_cols = [col for col in pred_df.columns if col not in ['unique_id', 'ds']]
            if not pred_cols:
                raise ValueError("Aucune colonne de prédiction trouvée dans pred_df.columns")
            pred_colname = pred_cols[0]

            # Fusion pour conserver y réel (depuis df_pred_true copie) + y_pred
            merged = df_pred_true.merge(pred_df, on=["unique_id", "ds"])

            # Ajout aux résultats
            results.append(merged)

            # REMPLACEMENT récursif : on injecte les y_pred dans df_used pour cette fenêtre
            # (pour que la prochaine itération n'utilise plus les y réels de ce bloc)
            df_used.loc[
                df_used.index[n_train+start:n_train+end], 'y'
            ] = merged[pred_colname].values  

            # Avance à la fenêtre suivante
            start += window_size
            end += window_size

        if not results:
            raise ValueError("Aucune fenêtre rolling n'a été générée, vérifier la taille de vos données/test.")

        df_results = pd.concat(results, ignore_index=True)
        df_results = df_results.rename(columns={pred_colname: 'y_pred'})
        df_results['y_pred'] = df_results['y_pred'].clip(lower=0)

        # ======== INTERVALLES DE CONFIANCE ==========
        model_class_name = first_model.__class__.__name__
        for lvl in [90, 95]:
            lo_col = f'{model_class_name}-lo-{lvl}'
            hi_col = f'{model_class_name}-hi-{lvl}'
            if lo_col in df_results.columns:
                df_results[lo_col] = df_results[lo_col].clip(lower=0)
            if hi_col in df_results.columns:
                df_results[hi_col] = df_results[hi_col].clip(lower=0)
            if lo_col in df_results.columns and hi_col in df_results.columns:
                df_results[f'ic_lo_{lvl}'] = df_results[lo_col]
                df_results[f'ic_hi_{lvl}'] = df_results[hi_col]

        colonnes = ["unique_id", "ds", "Annee", "Semaine", "y", "y_pred",
                    "ic_lo_90", "ic_hi_90", "ic_lo_95", "ic_hi_95"]
        colonnes = [c for c in colonnes if c in df_results.columns]
        df_results = df_results[colonnes]

        # Sauvegarde
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        except NameError:
            project_root = os.getcwd()
        data_save_dir = os.path.join(project_root, "data", "models_data", f"models_{target_col_name}_{zone}")
        os.makedirs(data_save_dir, exist_ok=True)

        df_results_to_save = df_results.reset_index(drop=True)
        df_results_to_save['model_name'] = model_class_name
        df_results_to_save.to_csv(os.path.join(data_save_dir, "predictions.csv"), index=False)

        y_true = df_results['y'].values
        y_pred = df_results['y_pred'].values
        rmse_final = mean_squared_error(y_true, y_pred, squared=False)
        mae_final = mean_absolute_error(y_true, y_pred)
        best_model_metrics = {'RMSE': rmse_final, 'MAE': mae_final}

        
        pd.DataFrame([best_model_metrics]).to_csv(os.path.join(data_save_dir, "best_metrics_recursif.csv"), index=False)

        if verbose:
            logging.info(f"Test récursif et sauvegardes terminés pour {zone} - {target_col_name}")

        best_params_path = os.path.join(save_dir, "best_params.json")
        best_config = {}
        if os.path.exists(best_params_path):
            with open(best_params_path) as f:
                best_config = json.load(f)
        
        # ==== IMPORTANCE DES VARIABLES ====
        importance_df = None
        model_obj = first_model
        if main_model == 'xgb':
            # Pour XGBRegressor
            feature_names = model_obj.get_booster().feature_names
            importances = model_obj.feature_importances_
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        elif main_model == 'lgbm':
            # Pour LGBMRegressor
            feature_names = model_obj.feature_name_
            importances = model_obj.feature_importances_
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        elif main_model == 'cat':
            # Pour CatBoostRegressor
            feature_names = model_obj.feature_names_
            importances = model_obj.get_feature_importance()
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        if importance_df is not None:
            importance_df = importance_df.sort_values(by='importance', ascending=False)
            importance_df.to_csv(os.path.join(data_save_dir, "importance.csv"), index=False)

        return {
            'zone': zone,
            'best_params': best_config,
            'best_metrics': best_model_metrics,
            'predictions': df_results,
            'mlforecast': fcst,
            'save_dir': save_dir
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"Erreur dans test_model_zone_ML2_recursif_test_only pour la zone {zone} : {repr(e)}")
        return None

def test_all_zones(
    df_train,
    df_test,
    base_path,
    target_col,
    target_col_name,
    window_size,
    h,
    verbose=True,
    exog_cols=None,
    ohe_dict=None,
    categorical_cols=None
):
    """
    Applique la fonction test_model_zone_ML2_recursive_test à chaque zone unique du DataFrame de test.
    Pour chaque zone, effectue un test rolling récursif et sauvegarde les résultats dans le bon dossier.

    Paramètres
    ----------
    df_train : pandas.DataFrame
        Données d’apprentissage complètes.
    df_test : pandas.DataFrame
        Données de test complètes.
    base_path : str
        Chemin du dossier où sont sauvegardés les modèles et résultats.
    target_col : str
        Nom de la colonne cible.
    target_col_name : str
        Nom de la cible (pour retrouver le dossier du modèle).
    window_size : int
        Taille de la fenêtre rolling.
    verbose : bool, optionnel
        Affiche les logs étape par étape (défaut = True).
    exog_cols : list, optionnel
        Variables exogènes à utiliser (None = auto).
    ohe_dict : dict, optionnel
        Dictionnaire d’encodage OneHot.
    categorical_cols : list, optionnel
        Colonnes catégorielles (pour CatBoost).

    Retour
    ------
    dict
        Dictionnaire {zone: result_dict} avec les résultats de test pour chaque zone.
    
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    zones_uniques = df_test['unique_id'].unique()
    resultats_test = {}
    for i, zone in enumerate(zones_uniques):
        try:
            if verbose:
                logging.info(f"--- Prédictions récursives sur le test pour la zone {zone} ---")
            result_dict = test_model_zone_ML2_rolling_test(
                df_train=df_train,
                df_test=df_test,
                zone=zone,
                target_col=target_col,
                target_col_name=target_col_name,
                window_size=window_size,
                base_path=base_path,
                verbose=verbose,
                exog_cols=exog_cols,
                ohe_dict=ohe_dict,
                categorical_cols=categorical_cols,
                h=h
            )
            resultats_test[zone] = result_dict
            if verbose and result_dict is not None:
                logging.info(f"Succès prédictions récursives sur le test {zone} | Nb lignes={len(result_dict['predictions'])}")
        except Exception as e:
            logging.error(f"ÉCHEC prédictions récursives sur le test {zone}: {e}")
            continue
    return resultats_test





def predict_futur_for_all_zones_ML2(
    df_train, df_test, target_col_name, base_path="./models", window_size=1, verbose=True,
    exog_cols=None, ohe_dict=None, categorical_cols=None, h=None
):
    """
    Prédit les valeurs futures pour chaque zone avec un modèle MLForecast déjà optimisé et sauvegardé.
    Utilise un rolling window : chaque prédiction est réinjectée comme nouvelle valeur de 'y' pour la suite (auto-régressif),
     afin de simuler un vrai scénario de prévision sans connaître le vrai 'y' du futur.
    Les résultats (prédictions et intervalles de confiance) sont sauvegardés dans un fichier par zone.

    Paramètres
    ----------
    df_train : pandas.DataFrame
        Données d’entraînement par zone.
    df_test : pandas.DataFrame
        Données futures à prédire par zone.
    target_col_name : str
        Nom de la cible (pour retrouver le dossier modèle).
    base_path : str, optionnel
        Chemin du dossier des modèles (défaut = "./models").
    window_size : int, optionnel
        Taille de la fenêtre rolling pour les prédictions (défaut = 1).
    verbose : bool, optionnel
        Affiche les logs détaillés (défaut = True).
    exog_cols : list, optionnel
        Variables exogènes à utiliser (None = auto).
    ohe_dict : dict, optionnel
        Dictionnaire d’encodage OneHot.
    categorical_cols : list, optionnel
        Colonnes catégorielles (pour CatBoost).

    Retour
    ------
    dict
        Dictionnaire {zone: DataFrame} contenant les prédictions futures pour chaque zone.
   """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    zones_uniques = df_train['unique_id'].unique()
    all_preds = {}

    for zone in zones_uniques:
        if verbose:
            print(f"\n=== Zone {zone} ===")
        save_dir = os.path.join(f"{base_path}_{h}", f"models_{target_col_name}_{zone}")
        if not os.path.exists(save_dir):
            logging.error(f"Dossier {save_dir} introuvable, lance d'abord optimize_model_zone pour la zone {zone}")
            continue

        try:
            forecast = MLForecast.load(save_dir)
            first_model = list(forecast.models.values())[0]
            model_name = first_model.__class__.__name__.lower()
            if 'xgb' in model_name:
                main_model = 'xgb'
            elif 'lgbm' in model_name:
                main_model = 'lgbm'
            elif 'cat' in model_name:
                main_model = 'cat'
            else:
                main_model = 'other'

            df_train_zone = df_train[df_train['unique_id'] == zone].copy()
            df_test_zone = df_test[df_test['unique_id'] == zone].copy()
            n_train = len(df_train_zone)
            n_test = len(df_test_zone)

            # Exogènes
            if exog_cols is None:
                exog_cols_zone = [col for col in df_train_zone.columns if col not in ['unique_id', 'ds', 'y']]
            else:
                exog_cols_zone = exog_cols

            # Préparation pour rolling
            df_full = pd.concat([df_train_zone, df_test_zone], ignore_index=True)
            if main_model in ['xgb', 'lgbm']:
                df_full_proc, _, final_exog_cols, ohe_dict_out, categorical_cols_out = preprocess_exog_and_target2(
                    df_full, df_full, exog_cols_zone
                )
                df_used = df_full_proc
                used_exog_cols = final_exog_cols
            elif main_model == 'cat':
                df_used = df_full
                used_exog_cols = exog_cols_zone
                if categorical_cols is None:
                    _, _, _, _, categorical_cols = preprocess_exog_and_target2(df_full, df_full, exog_cols_zone)
                cat_features = categorical_cols
            else:
                df_used = df_full
                used_exog_cols = exog_cols_zone
            
            if h is not None and h <= 16:
                forecast.fit(
                    df_used.iloc[:n_train],  # les lignes du train uniquement sont utilisé ici , donc on refit le model sur tout le train
                    id_col='unique_id',
                    time_col='ds',
                    target_col='y',
                    static_features=[],
                    prediction_intervals= PredictionIntervals(n_windows=2, h=2)
                )
            else:
                forecast.fit(
                    df_used.iloc[:n_train],
                    id_col='unique_id',
                    time_col='ds',
                    target_col='y',
                    static_features=[]
                )
 

            # Début des prédictions récursives
            start = 0
            end = window_size
            results = []

            # Initial update avec tout le train connu
            forecast.update(df_used.iloc[:n_train])

            # Rolling sur le test (futur)
            while end <= n_test:
                # Fenêtre courante du test à prédire
                idx_pred = np.arange(n_train + start, n_train + end)
                df_pred_input = df_used.iloc[idx_pred][['unique_id', 'ds'] + used_exog_cols].copy()

                # Prédire la/les semaines de la fenêtre
                if h is not None and h <= 16:
                    pred_df = forecast.predict(h=window_size, X_df=df_pred_input, level=[90, 95])
                else: 
                    pred_df = forecast.predict(h=window_size, X_df=df_pred_input)


                pred_cols = [col for col in pred_df.columns if col not in ['unique_id', 'ds']+
                             [f'lo-90', f'hi-90', f'lo-95', f'hi-95']]
                if not pred_cols:
                    logging.error(f"Aucune colonne de prédiction trouvée dans pred_df.columns pour {zone}")
                    break
                pred_colname = pred_cols[0]

                # Sauvegarder la prédiction
                # -- on remplace la valeur manquante de y dans df_used par la prédiction
                y_pred = pred_df[pred_colname].values
                y_pred = np.clip(y_pred, 0, None)
                df_used.loc[idx_pred, 'y_pred'] = y_pred
                df_used.loc[idx_pred, 'y'] = y_pred  # Injection pour update

                 # On construit le résultat de la fenêtre avec IC si présents
                base_cols = ['unique_id', 'ds'] + used_exog_cols + ['y_pred']
                
                # ==== AJOUT IC ====
                model_class_name = first_model.__class__.__name__
                ic_cols = []
                for lvl in [90, 95]:
                    lo_col = f'{model_class_name}-lo-{lvl}'
                    hi_col = f'{model_class_name}-hi-{lvl}'
                    if lo_col in pred_df.columns and hi_col in pred_df.columns:
                        df_used.loc[idx_pred, f'ic_lo_{lvl}'] = pred_df[lo_col].values
                        df_used.loc[idx_pred, f'ic_hi_{lvl}'] = pred_df[hi_col].values
                        ic_cols.extend([f'ic_lo_{lvl}', f'ic_hi_{lvl}'])

                # On conserve le résultat de la fenêtre
                results.append(
                    df_used.iloc[idx_pred][base_cols + ic_cols].copy()
                )

                # Update le modèle avec la fenêtre courante (y prédit)
                forecast.update(df_used.iloc[idx_pred])

                start += window_size
                end += window_size

            if not results:
                logging.error(f"Aucune fenêtre rolling n'a été générée pour {zone}")
                continue

            df_results = pd.concat(results, ignore_index=True)
            df_results['y_pred'] = df_results['y_pred'].clip(lower=0)
            for lvl in [90, 95]:
                lo_col = f'ic_lo_{lvl}'
                hi_col = f'ic_hi_{lvl}'
                if lo_col in df_results.columns:
                    df_results[lo_col] = df_results[lo_col].clip(lower=0)
                if hi_col in df_results.columns:
                    df_results[hi_col] = df_results[hi_col].clip(lower=0)

            # On récupére le nom du model pour l'enregistrer avec les prédicitons
            df_results['model_name'] = model_class_name
            colonnes = ["unique_id", "ds", "Annee", "Semaine", "y_pred",
                        "ic_lo_90", "ic_hi_90", "ic_lo_95", "ic_hi_95", "model_name"]
            colonnes = [c for c in colonnes if c in df_results.columns]

            # Sauvegarde
            try:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            except NameError:
                project_root = os.getcwd()
            data_save_dir = os.path.join(project_root, "data", f"models_data_{h}", f"models_{target_col_name}_{zone}")
            os.makedirs(data_save_dir, exist_ok=True)
            df_results_to_save = df_results.reset_index(drop=True)
            df_results_to_save[colonnes].to_csv(os.path.join(data_save_dir, "predictions_futur.csv"), index=False)

             # ==== IMPORTANCE DES VARIABLES FUTURES ====
            importance_futur_df = None
            model_obj = first_model
            if main_model == 'xgb':
                feature_names = model_obj.get_booster().feature_names
                importances = model_obj.feature_importances_
                importance_futur_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            elif main_model == 'lgbm':
                feature_names = model_obj.feature_name_
                importances = model_obj.feature_importances_
                importance_futur_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            elif main_model == 'cat':
                feature_names = model_obj.feature_names_
                importances = model_obj.get_feature_importance()
                importance_futur_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            if importance_futur_df is not None:
                importance_futur_df = importance_futur_df.sort_values(by='importance', ascending=False)
                importance_futur_df.to_csv(os.path.join(data_save_dir, "importance_futur.csv"), index=False)

            all_preds[zone] = df_results_to_save

            if verbose:
                logging.info(f"Prédictions futures rolling faites pour {zone} et sauvegardées dans {os.path.join(data_save_dir, 'predictions_futur.csv')}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            logging.error(f"Erreur dans predict_futur_for_all_zones_ML2 pour la zone {zone} : {repr(e)}")
            continue

    return all_preds


# def optimize_single_poste_mlforecast_no_standard(
#     df_train, df_test, poste, target_col, target_col_name, h, base_path, n_trials, verbose,
#     exog_cols=None
# ):

#     # Fix all random seeds for reproducibility
#     RANDOM_STATE = 42
#     np.random.seed(RANDOM_STATE)
#     random.seed(RANDOM_STATE)
#     try:
#         import torch
#         torch.manual_seed(RANDOM_STATE)
#         torch.cuda.manual_seed(RANDOM_STATE)
#         torch.cuda.manual_seed_all(RANDOM_STATE)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#     except ImportError:
#         pass

#     # Optuna seed for reproducibility
#     sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

#     df_train = df_train[df_train['unique_id'] == poste].copy()
#     df_test = df_test[df_test['unique_id'] == poste].copy()

#     # Always pass random_state to all models
#     MODELS = {
#         'lgbm': lambda **kwargs: LGBMRegressor(**kwargs),
#         'xgb': lambda **kwargs: XGBRegressor(**kwargs),
#         'cat': lambda **kwargs: CatBoostRegressor(**kwargs)
#     }
#     lags = list(range(1, 13))
#     lag_transforms = {lag: [RollingMean(window_size=lag), ExpandingMean(), RollingQuantile(window_size=lag, p=0.5), ExponentiallyWeightedMean(alpha=0.35)] for lag in lags}

#     if exog_cols is None:
#         exog_cols = [col for col in df_train.columns if col not in ['unique_id', 'ds', 'y']]

#     # Only encoding, no standardization
#     df_train_proc, df_test_proc, final_exog_cols, ohe_dict, categorical_cols = preprocess_exog_and_target2(df_train, df_test, exog_cols)

#     preds_dict = {}
#     print("df_test_proc shape:", df_test_proc.shape)
#     print("h:", h)

#     for model_name, model_fn in MODELS.items():
#         if verbose:
#             print(f"\nPremier fit simple pour {model_name.upper()} (sans tuning Optuna)...")
#         if model_name == 'cat':
#             model = model_fn(verbose=0)
#         elif model_name == 'lgbm':
#             model = model_fn(verbose=-1)
#         else:
#             model = model_fn()

#         forecast = MLForecast(
#             models=[model],
#             freq='W-MON',
#             lags=lags,
#             lag_transforms=lag_transforms,
#             target_transforms=[],
#             date_features=['month', 'quarter']
#         )
#         forecast.fit(
#             df_train_proc,
#             id_col='unique_id',
#             time_col='ds',
#             target_col='y',
#             static_features=[]
#         )
#         y_pred_scaled = forecast.predict(h=h, X_df=df_test_proc)
#         pred_cols = [col for col in y_pred_scaled.columns if col not in ['unique_id', 'ds']]
#         if not pred_cols:
#             raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled.columns}")
#         y_pred_model_col = y_pred_scaled[pred_cols[0]].values[:h]
#         y_pred = y_pred_model_col
#         preds_dict[model_name] = y_pred

#     y_true = df_test['y'].values[:h]
#     best_model, best_model_metrics = select_best_model_by_vote(preds_dict, y_true, verbose=verbose)

#     if verbose:
#         print(f"\nOptimisation Optuna sur {best_model.upper()} ...")

#     def objective(trial):
#         config = get_optimization_config(best_model, trial)
#         # Always force random_state in config if possible (can be overwritten)
#         config['random_state'] = RANDOM_STATE
#         if best_model == 'cat':
#             config['verbose'] = 0
#         elif best_model == 'lgbm':
#             config['verbose'] = -1

#         model = MODELS[best_model](**config)
#         forecast = MLForecast(
#             models=[model],
#             freq='W-MON',
#             lags=lags,
#             lag_transforms=lag_transforms,
#             target_transforms=[]
#         )
#         try:
#             forecast.fit(
#                 df_train_proc,
#                 id_col='unique_id',
#                 time_col='ds',
#                 target_col='y',
#                 static_features=[]
#             )
#             y_pred_scaled = forecast.predict(h=h, X_df=df_test_proc)
#             pred_cols = [col for col in y_pred_scaled.columns if col not in ['unique_id', 'ds']]
#             if not pred_cols:
#                 raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled.columns}")
#             y_pred_model_col = y_pred_scaled[pred_cols[0]].values[:h]
#             y_pred = y_pred_model_col
#             score = mean_squared_error(y_true, y_pred, squared=False)
#             return score
#         except Exception as e:
#             print(f"Erreur dans objective Optuna {best_model}: {e}")
#             return float('inf')

#     study = optuna.create_study(direction='minimize', sampler=sampler)
#     early_stop = PatienceEarlyStopper(patience=12)
#     study.optimize(objective, n_trials=n_trials, callbacks=[early_stop], show_progress_bar=False)
#     if verbose:
#         print(f"Best RMSE {best_model}: {study.best_value:.4f}")

#     best_config = get_optimization_config(best_model, study.best_trial)
#     # Always force random_state in config
#     best_config['random_state'] = RANDOM_STATE
#     if best_model == 'cat':
#         best_config['verbose'] = 0
#     elif best_model == 'lgbm':
#         best_config['verbose'] = -1

#     model = MODELS[best_model](**best_config)
#     forecast = MLForecast(
#         models=[model],
#         freq='W-MON',
#         lags=lags,
#         lag_transforms=lag_transforms,
#         target_transforms=[]
#     )
#     forecast.fit(
#         df_train_proc,
#         id_col='unique_id',
#         time_col='ds',
#         target_col='y',
#         static_features=[]
#     )

#     y_pred_scaled_final = forecast.predict(h=h, X_df=df_test_proc)

#     save_dir = f"{base_path}/models_{target_col_name}_{poste}/"
#     os.makedirs(save_dir, exist_ok=True)
#     if verbose:
#         print(f"Sauvegarde: {save_dir}")

#     df_train.to_csv(os.path.join(save_dir, "train_data.csv"), index=False)
#     joblib.dump(forecast, os.path.join(save_dir, f"{best_model}_mlforecast.pkl"))
#     with open(f"{save_dir}/best_params.json", "w") as f:
#         json.dump(best_config, f)

#     preds_df = df_test.reset_index(drop=True).copy()
#     preds_df['y_true'] = y_true
#     pred_cols_final = [col for col in y_pred_scaled_final.columns if col not in ['unique_id', 'ds']]
#     if not pred_cols_final:
#         raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled_final.columns}")

#     y_pred_model_col_final = y_pred_scaled_final[pred_cols_final[0]].values[:h]
#     y_pred_final = y_pred_model_col_final

#     preds_df['y_pred'] = y_pred_final

#     from scipy.stats import norm

#     residuals = y_true - y_pred_final
#     std_res = np.std(residuals, ddof=1)
#     z_95 = norm.ppf(0.975)
#     z_90 = norm.ppf(0.95)

#     preds_df['lower_95'] = y_pred_final - z_95 * std_res
#     preds_df['upper_95'] = y_pred_final + z_95 * std_res
#     preds_df['lower_90'] = y_pred_final - z_90 * std_res
#     preds_df['upper_90'] = y_pred_final + z_90 * std_res
#     preds_df.to_csv(f"{save_dir}/predictions.csv", index=False)

#     rmse_final = mean_squared_error(y_true, y_pred_final, squared=False)
#     mae_final = mean_absolute_error(y_true, y_pred_final)
#     mape_final = mape(y_true, y_pred_final)

#     best_model_metrics = {'RMSE': rmse_final, 'MAE': mae_final, 'MAPE': mape_final}
#     pd.DataFrame([best_model_metrics]).to_csv(f"{save_dir}/best_metrics.csv", index=False)

#     return {
#         'poste': poste,
#         'best_model': best_model,
#         'best_params': best_config,
#         'best_metrics': best_model_metrics,
#         'predictions': preds_df,
#         'mlforecast': forecast,
#         'save_dir': save_dir,
#         'ohe_dict': ohe_dict,
#         'categorical_cols': categorical_cols,
#         'final_exog_cols': final_exog_cols
#     }



# def optimize_models_for_all_postes(
#     df_train, df_test, target_col_name , target_col='y',  h=12, 
#     base_path="./models", n_trials=20, verbose=True
# ):
#     """
#     Optimise un modèle MLForecast pour chaque poste d'observation avec Optuna.
#     """
#     if verbose:
#         print("="*60)
#         print("OPTIMISATION DES MODÈLES MLFORECAST PAR POSTE")
#         print("="*60)
    
#     postes_uniques = df_train['unique_id'].unique()
#     resultats_tous_postes = {}
#     postes_reussis, postes_echecs = 0, 0
    
#     if verbose:
#         print(f"Postes détectés: {len(postes_uniques)}")
#         print(f"Données: Train={len(df_train)} obs, Test={len(df_test)} obs")
#         print(f"Paramètres: h={h}, n_trials={n_trials}")
    
#     for i, poste in enumerate(postes_uniques):
#         if verbose:
#             print(f"\n[{i+1}/{len(postes_uniques)}] POSTE: {poste}")
#             print("-"*40)
#         try:
#             df_train_poste = df_train[df_train['unique_id'] == poste].copy()
#             df_test_poste = df_test[df_test['unique_id'] == poste].copy()
#             if verbose:
#                 print(f"Données: Train={len(df_train_poste)}, Test={len(df_test_poste)}")
            
#             resultats_poste = optimize_single_poste_mlforecast_no_standard(
#                 df_train=df_train_poste,
#                 df_test=df_test_poste,
#                 poste=poste,
#                 target_col=target_col,
#                 target_col_name= target_col_name,
#                 h=h,
#                 base_path=base_path,
#                 n_trials=n_trials,
#                 verbose=verbose
#             )
#             resultats_tous_postes[poste] = resultats_poste
#             postes_reussis += 1
            
#             if verbose:
#                 print(f"SUCCÈS - Meilleur: {resultats_poste['best_model']}")
#                 metrics = resultats_poste['best_metrics']
#                 print(f"Métriques: RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}, MAPE={metrics['MAPE']:.2f}")
#         except Exception as e:
#             postes_echecs += 1
#             if verbose:
#                 print(f"ÉCHEC: {e}")
#             continue
    
#     if verbose:
#         print(f"\n{'='*60}")
#         print("RÉSUMÉ FINAL")
#         print(f"{'='*60}")
#         print(f"Postes traités: {len(postes_uniques)}")
#         print(f"Succès: {postes_reussis}")
#         print(f"Échecs: {postes_echecs}")
#         print(f"Taux réussite: {(postes_reussis/len(postes_uniques)*100):.1f}%")
#         if postes_reussis > 0:
#             modeles_count = {}
#             for result in resultats_tous_postes.values():
#                 model = result['best_model']
#                 modeles_count[model] = modeles_count.get(model, 0) + 1
#             print(f"\nModèles gagnants:")
#             for model, count in sorted(modeles_count.items(), key=lambda x: x[1], reverse=True):
#                 print(f"  {model}: {count} postes ({count/postes_reussis*100:.1f}%)")
#     print(f"\nOptimisation terminée !")
#     return resultats_tous_postes

# #Visualisation des résultats

# def visualiser_poste(
#     base_path,
#     target_col_name,
#     poste,
#     pred_col='y_pred',  # colonne générique
#     train_filename='train_data.csv',
#     test_filename='predictions.csv'
# ):
#     import numpy as np
#     # Chemins
#     save_dir = os.path.join(base_path, f"models_{target_col_name}_{poste}")
#     train_path = os.path.join(save_dir, train_filename)
#     test_path = os.path.join(save_dir, test_filename)

#     if not os.path.exists(train_path):
#         print(f"Train data not found: {train_path}")
#         return
#     if not os.path.exists(test_path):
#         print(f"Predictions not found: {test_path}")
#         return

#     # Chargement et conversion des dates
#     train_df = pd.read_csv(train_path)
#     test_df = pd.read_csv(test_path)

#     # Conversion en datetime
#     train_df['ds'] = pd.to_datetime(train_df['ds'])
#     test_df['ds'] = pd.to_datetime(test_df['ds'])

#     # Colonne 'y_pred'
#     if pred_col not in test_df.columns:
#         pred_cols = [c for c in test_df.columns if c not in ['unique_id', 'ds', 'y_true']]
#         if len(pred_cols) == 1:
#             pred_col = pred_cols[0]
#         else:
#             raise ValueError(f"Colonne de prédiction non trouvée dans {test_df.columns}")

#     y_pred = test_df[pred_col]
#     y_true = test_df['y_true']

#     # Calcul des métriques
#     rmse = mean_squared_error(y_true, y_pred, squared=False)
#     mae = mean_absolute_error(y_true, y_pred)
#     mape_score = mape(y_true, y_pred)

#     # Visualisation style PRO
#     fig, ax = plt.subplots(1, 1, figsize=(14, 8))

#     # Historique (train)
#     ax.plot(train_df['ds'], train_df['y'], 'gray', alpha=0.7,
#             label='Historique (train)', linewidth=1.5)

#     # Vrais test
#     ax.plot(test_df['ds'], y_true, 'b-', label='Réel (test)', linewidth=2.5,
#             marker='o', markersize=6)

#     # Prédictions
#     ax.plot(test_df['ds'], y_pred, 'r--', label='Prédiction', linewidth=2.5,
#             marker='s', markersize=6)

#     # Ligne séparation train/test
#     ax.axvline(test_df['ds'].iloc[0], color='green', linestyle=':',
#                label='Début test', alpha=0.8, linewidth=2)

#     # Titre avec métriques
#     ax.set_title(
#         f'Poste {poste}\n'
#         f'RMSE: {rmse:.3f} | MAE: {mae:.3f} | MAPE: {mape_score:.2f}%',
#         fontsize=16, fontweight='bold', pad=20
#     )

#     ax.set_xlabel('Date', fontsize=12)
#     ax.set_ylabel(target_col_name, fontsize=12)
#     ax.legend(fontsize=12, loc='best')
#     ax.grid(True, alpha=0.3)

#     # Rotation des dates et espacement
#     ax.tick_params(axis='x', rotation=45, labelsize=10)
#     ax.tick_params(axis='y', labelsize=10)
#     plt.tight_layout()
#     plt.show()




# ################################################ période future ################################################

# def train_test_split_futur(
#     df: pd.DataFrame,
#     split_annee: int,
#     split_semaine: int,
#     test_size: int = 4,
#     min_total_size: int = 15,
#     min_train_size: int = 3,
#     verbose: bool = True
# ):
#     """
#     Pour chaque poste, splitte le train jusqu'à (split_annee, split_semaine) inclus,
#     puis le test prend les `test_size` semaines suivantes.
#     Le reste est ignoré.
#     """
#     required_cols = ['unique_id', 'Annee', 'Semaine', 'y']
#     missing_cols = [col for col in required_cols if col not in df.columns]
#     if missing_cols:
#         raise ValueError(f"Colonnes manquantes: {missing_cols}")

#     train_dfs = []
#     test_dfs = []
#     postes_traites = 0
#     postes_ignores_taille = 0
#     postes_ignores_train = 0
#     postes_ignores_cible = 0

#     postes_uniques = df['unique_id'].unique()
#     if verbose:
#         print(f"🔄 Début du split train/test 'futur' par poste...")
#         print(f"Split à partir de {split_annee} semaine {split_semaine} | test_size = {test_size}")
#         print(f"🎯 Traitement de {len(postes_uniques)} postes...")

#     for i, poste in enumerate(postes_uniques):
#         df_poste = df[df['unique_id'] == poste].copy().sort_values(['Annee', 'Semaine'])
#         if verbose and i < 5:
#             print(f"\n Poste {poste} ({i+1}/{len(postes_uniques)}):")
#             print(f"    Total observations: {len(df_poste)}")

#         # 1. Vérification taille minimale totale
#         if len(df_poste) < min_total_size:
#             postes_ignores_taille += 1
#             if verbose and i < 5:
#                 print(f"    Ignoré: données insuffisantes ({len(df_poste)} < {min_total_size})")
#             continue

#         # 2. Sélection train/test via coupure
#         # Index de la coupure
#         mask_train = ((df_poste['Annee'] < split_annee) |
#                       ((df_poste['Annee'] == split_annee) & (df_poste['Semaine'] <= split_semaine)))
#         df_train_poste = df_poste[mask_train].copy()

#         # Le test commence juste après la coupure (année/semaine), pour test_size semaines
#         mask_test = ((df_poste['Annee'] > split_annee) |
#                      ((df_poste['Annee'] == split_annee) & (df_poste['Semaine'] > split_semaine)))
#         df_test_candidats = df_poste[mask_test].copy().sort_values(['Annee', 'Semaine'])
#         df_test_poste = df_test_candidats.head(test_size)

#         # Contrôle tailles
#         if len(df_train_poste) < min_train_size:
#             postes_ignores_train += 1
#             if verbose and i < 5:
#                 print(f"    Ignoré: train trop petit ({len(df_train_poste)} < {min_train_size})")
#             continue
#         if df_train_poste['y'].isna().all():
#             postes_ignores_cible += 1
#             if verbose and i < 5:
#                 print(f"    Ignoré: toutes valeurs cibles manquantes dans train")
#             continue

#         train_dfs.append(df_train_poste)
#         test_dfs.append(df_test_poste)
#         postes_traites += 1

#         if verbose and i < 5:
#             print(f"   ✅ Split réussi:")
#             print(f"      📚 Train: {len(df_train_poste)} semaines")
#             print(f"      🧪 Test: {len(df_test_poste)} semaines")
#             if not df_train_poste.empty:
#                 print(f"      📅 Train: {df_train_poste['Annee'].min()}-{df_train_poste['Semaine'].min()} à {df_train_poste['Annee'].max()}-{df_train_poste['Semaine'].max()}")
#             if not df_test_poste.empty:
#                 print(f"      📅 Test: {df_test_poste['Annee'].min()}-{df_test_poste['Semaine'].min()} à {df_test_poste['Annee'].max()}-{df_test_poste['Semaine'].max()}")

#     if train_dfs:
#         df_train = pd.concat(train_dfs, ignore_index=True)
#         df_test = pd.concat(test_dfs, ignore_index=True)
#     else:
#         df_train = pd.DataFrame()
#         df_test = pd.DataFrame()
#         raise ValueError("Aucun poste n'a pu être traité pour le split train/test")

#     if verbose:
#         print(f"\n{'='*60}")
#         print(f" RÉSUMÉ DU SPLIT TRAIN/TEST FUTUR")
#         print(f"{'='*60}")
#         print(f" Postes traités avec succès: {postes_traites}")
#         print(f" Postes ignorés:")
#         print(f"   • Taille insuffisante: {postes_ignores_taille}")
#         print(f"   • Train trop petit: {postes_ignores_train}")
#         print(f"   • Problème cible: {postes_ignores_cible}")
#         print(f" Taux de succès: {(postes_traites/len(postes_uniques)*100):.1f}%")
#         print(f"\n Données finales:")
#         print(f"    Train: {len(df_train)} observations")
#         print(f"    Test: {len(df_test)} observations")
#         if not df_train.empty:
#             print(f"    Période train: {df_train['Annee'].min()}-{df_train['Semaine'].min()} à {df_train['Annee'].max()}-{df_train['Semaine'].max()}")
#         if not df_test.empty:
#             print(f"    Période test: {df_test['Annee'].min()}-{df_test['Semaine'].min()} à {df_test['Annee'].max()}-{df_test['Semaine'].max()}")

#     print(f"\n Split terminé avec succès!")
#     return df_train, df_test


# def visualiser_futur_poste(
#     base_path,
#     target_col_name,
#     poste,
#     train_filename='train_futur.csv',
#     futur_pred_filename='futur_predictions.csv',
#     pred_col='y_pred',
#     ytrue_col='y_true',  # optionnel, si tu as le vrai y dans df_test
# ):
#     """
#     Visualise l'historique (train utilisé pour la prédiction future)
#     et les prédictions futures pour un poste donné.
#     """
#     save_dir = os.path.join(base_path, f"models_{target_col_name}_{poste}")
#     train_path = os.path.join(save_dir, train_filename)
#     fut_pred_path = os.path.join(save_dir, futur_pred_filename)

#     if not os.path.exists(train_path):
#         print(f"Train data not found: {train_path}")
#         return
#     if not os.path.exists(fut_pred_path):
#         print(f"Futur predictions not found: {fut_pred_path}")
#         return

#     # Chargement
#     train_df = pd.read_csv(train_path)
#     fut_pred_df = pd.read_csv(fut_pred_path)

#     # Dates en datetime
#     train_df['ds'] = pd.to_datetime(train_df['ds'])
#     fut_pred_df['ds'] = pd.to_datetime(fut_pred_df['ds'])

#     # Y true futur (optionnel)
#     has_y_true = ytrue_col in fut_pred_df.columns
#     if has_y_true:
#         y_true = fut_pred_df[ytrue_col]
#     else:
#         y_true = None

#     # Y préd
#     if pred_col not in fut_pred_df.columns:
#         pred_cols = [c for c in fut_pred_df.columns if c not in ['unique_id', 'ds', ytrue_col, 'type_prediction']]
#         if len(pred_cols) == 1:
#             pred_col = pred_cols[0]
#         else:
#             raise ValueError(f"Colonne de prédiction non trouvée dans {fut_pred_df.columns}")

#     y_pred = fut_pred_df[pred_col]

#     # Calcul métriques si vrai y dispo
#     if has_y_true and y_true.notna().all():
#         rmse = mean_squared_error(y_true, y_pred, squared=False)
#         mae = mean_absolute_error(y_true, y_pred)
#         mape_score = mape(y_true, y_pred)
#         metrics_str = f'RMSE: {rmse:.3f} | MAE: {mae:.3f} | MAPE: {mape_score:.2f}%'
#     else:
#         metrics_str = "Prédictions futures (pas de vérité connue)"

#     # Visualisation
#     fig, ax = plt.subplots(1, 1, figsize=(14, 8))

#     # Historique (train utilisé pour le refit)
#     ax.plot(train_df['ds'], train_df['y'], 'gray', alpha=0.7,
#             label='Historique (train)', linewidth=1.5)

#     # Vrais y futurs (si dispo)
#     if has_y_true and y_true.notna().all():
#         ax.plot(fut_pred_df['ds'], y_true, 'b-', label='Réel (futur)', linewidth=2.5,
#                 marker='o', markersize=6)

#     # Prédictions futures
#     ax.plot(fut_pred_df['ds'], y_pred, 'r--', label='Prédiction futur', linewidth=2.5,
#             marker='s', markersize=6)

#     # Ligne de séparation
#     ax.axvline(fut_pred_df['ds'].iloc[0], color='green', linestyle=':',
#                label='Début prévision future', alpha=0.8, linewidth=2)

#     ax.set_title(
#         f'Poste {poste}\n{metrics_str}',
#         fontsize=16, fontweight='bold', pad=20
#     )

#     ax.set_xlabel('Date', fontsize=12)
#     ax.set_ylabel(target_col_name, fontsize=12)
#     ax.legend(fontsize=12, loc='best')
#     ax.grid(True, alpha=0.3)
#     ax.tick_params(axis='x', rotation=45, labelsize=10)
#     ax.tick_params(axis='y', labelsize=10)
#     plt.tight_layout()
#     plt.show()




# def predict_futur_for_all_postes(
#     df_train, df_test, target_col_name, base_path="./models", h=4, verbose=True, exog_cols=None
# ):
#     MODELS = {
#         'lgbm': LGBMRegressor,
#         'xgb': XGBRegressor,
#         'cat': CatBoostRegressor,
#     }

#     postes_uniques = df_train['unique_id'].unique()
#     all_preds = {}

#     for poste in postes_uniques:
#         if verbose:
#             print(f"\n=== Poste {poste} ===")

#         save_dir = f"{base_path}/models_{target_col_name}_{poste}/"

#         # 1. Identifier le modèle gagnant
#         best_model_name = None
#         for model_name in ["xgb", "lgbm", "cat"]:
#             model_file = os.path.join(save_dir, f"{model_name}_mlforecast.pkl")
#             if os.path.exists(model_file):
#                 best_model_name = model_name
#                 break
#         if best_model_name is None:
#             print(f"⚠️ Aucun modèle trouvé pour {poste}")
#             continue

#         # 2. Charger les meilleurs hyperparams
#         with open(os.path.join(save_dir, "best_params.json")) as f:
#             best_params = json.load(f)

#         # 3. (Re)fit preprocessing sur toutes les données train disponibles
#         df_train_poste = df_train[df_train['unique_id'] == poste].copy()
#         df_test_poste = df_test[df_test['unique_id'] == poste].copy()

#         cols = exog_cols if exog_cols is not None else [
#             col for col in df_train_poste.columns if col not in ['unique_id', 'ds', 'y']
#         ]
#         exog_cols = [col for col in df_train_poste.columns if col not in ['unique_id', 'ds', 'y']]

#         df_train_proc, df_test_proc, final_exog_cols, ohe_dict, scaler_exog, scaler_y, _, _ = preprocess_exog_and_target(
#             df_train_poste, df_test_poste, cols
#         )

#         # === Vérifications exogènes ===
#         print("\n=== [EXOGENES DEBUG] ===")
#         print("Exogènes finales utilisées:", final_exog_cols)
#         missing_cols_train = [col for col in final_exog_cols if col not in df_train_proc.columns]
#         missing_cols_test = [col for col in final_exog_cols if col not in df_test_proc.columns]
#         if missing_cols_train:
#             print(f"❌ Colonnes exogènes manquantes dans TRAIN: {missing_cols_train}")
#         if missing_cols_test:
#             print(f"❌ Colonnes exogènes manquantes dans TEST: {missing_cols_test}")
#         print("df_train_proc shape:", df_train_proc.shape)
#         print("df_test_proc shape:", df_test_proc.shape)
#         # Vérifie les NaN dans les exogènes
#         nans_train = df_train_proc[final_exog_cols].isna().sum().sum()
#         nans_test = df_test_proc[final_exog_cols].isna().sum().sum()
#         print(f"NaN dans exogènes TRAIN: {nans_train}, TEST: {nans_test}")

#         # === DEBUG SAISONNALITE, ROLLING, IMPORTANCES ===
#         print("\n=== [SAISONNALITE SEMAINE] ===")
#         if 'weekofyear' not in df_train_poste.columns:
#             df_train_poste['weekofyear'] = pd.to_datetime(df_train_poste['ds']).dt.isocalendar().week
#         print(df_train_poste.groupby('weekofyear')['y'].mean().sort_index())

#         print("\n=== [IMPORTANCES MODELE] ===")
#         model_cls = MODELS[best_model_name]
#         model = model_cls(**best_params)

#         lags = list(range(1, 13))
#         lag_transforms = {lag: [RollingMean(window_size=lag), ExpandingMean(), RollingQuantile(window_size=lag, p= 0.5), ExponentiallyWeightedMean(alpha = 0.35)] for lag in lags}
#         forecast = MLForecast(
#             models=[model],
#             freq='W-MON',
#             lags=lags,
#             lag_transforms=lag_transforms,
#             target_transforms=[],
#             date_features=['month', 'quarter']
#         )
#         forecast.fit(
#             df_train_proc,
#             id_col='unique_id',
#             time_col='ds',
#             target_col='y',
#             static_features=[]
#         )

#         try:
#             if hasattr(model, "feature_importances_"):
#                 print("Feature importances :")
#                 feature_names = [c for c in df_train_proc.columns if c not in ['unique_id', 'ds', 'y' ]]
#                 print(
#                     sorted(
#                         zip(feature_names, model.feature_importances_),
#                         key=lambda x: -x[1]
#                     )
#                 )
#             else:
#                 print("Le modèle ne fournit pas de feature_importances_.")
#         except Exception as e:
#             print("Erreur lors de l'affichage des importance features :", e)

#         df_test_h = df_test_proc.groupby('unique_id').head(h)
#         print( "==============================Colonne test ", df_test_h.columns)
#         # Vérification exogènes sur df_test_h
#         print("\n=== [EXOGENES FUTUR DEBUG] ===")
#         for col in final_exog_cols:
#             if col in df_test_h.columns:
#                 unique_vals = df_test_h[col].unique()
#                 print(f"Colonne {col}: valeurs uniques dans df_test_h: {unique_vals[:5]} (total uniques: {len(unique_vals)})")
#             else:
#                 print(f"Colonne {col} ABSENTE de df_test_h")

#         y_pred_scaled = forecast.predict(h=h, X_df=df_test_h)
#         print("\n=== [DEBUG MLForecast LAGS] ===")
#         try:
#             print("Train features MLForecast:", forecast.current_features_)
#         except Exception as e:
#             print("Impossible d'afficher les features générées par MLForecast:", e)
#         pred_cols = [col for col in y_pred_scaled.columns if col not in ['unique_id', 'ds']]
#         y_pred_model_col = y_pred_scaled[pred_cols[0]].values

        
#         try:
#             y_pred = scaler_y.inverse_transform(y_pred_model_col.reshape(-1, 1)).flatten()
            
#         except Exception as e:
#             print(" Erreur lors de l'inverse_transform du scaler_y:", e)
#             y_pred = np.full_like(y_pred_model_col, np.nan)

#         df_preds = df_test_h.reset_index(drop=True).copy()
#         df_preds['y_pred'] = y_pred
#         df_preds['type_prediction'] = 'futur'
#         # Liste exacte des colonnes à inverse_transform (copiée de ton message)
#         cols_to_inverse = [
#             'Tmoy_mean', 'Tmoy_d1', 'Tmoy_d2', 'Tmoy_d3', 'Tmoy_d4', 'Tmoy_d5', 'Tmoy_d6', 'Tmoy_d7', 'Tmoy_d8', 'Tmoy_d9',
#             'Tmax_mean', 'Tmax_d1', 'Tmax_d2', 'Tmax_d3', 'Tmax_d4', 'Tmax_d5', 'Tmax_d6', 'Tmax_d7', 'Tmax_d8', 'Tmax_d9',
#             'Tmin_mean', 'Tmin_d1', 'Tmin_d2', 'Tmin_d3', 'Tmin_d4', 'Tmin_d5', 'Tmin_d6', 'Tmin_d7', 'Tmin_d8', 'Tmin_d9',
#             'Pluie_sum', 'Pluie_d1', 'Pluie_d2', 'Pluie_d3', 'Pluie_d4', 'Pluie_d5', 'Pluie_d6', 'Pluie_d7', 'Pluie_d8', 'Pluie_d9',
#             'Humidite_max_mean', 'Humidite_max_d1', 'Humidite_max_d2', 'Humidite_max_d3', 'Humidite_max_d4', 'Humidite_max_d5',
#             'Humidite_max_d6', 'Humidite_max_d7', 'Humidite_max_d8', 'Humidite_max_d9',
#             'Humidite_min_mean', 'Humidite_min_d1', 'Humidite_min_d2', 'Humidite_min_d3', 'Humidite_min_d4', 'Humidite_min_d5',
#             'Humidite_min_d6', 'Humidite_min_d7', 'Humidite_min_d8', 'Humidite_min_d9',
#             'Intensite_pluie_Alerte', 'Intensite_pluie_Calme', 'Intensite_pluie_Faible', 'Intensite_pluie_Forte', 'Intensite_pluie_Très forte'
#         ]

#         # Vérifie que toutes les colonnes sont présentes
#         cols_to_inverse = [col for col in cols_to_inverse if col in df_preds.columns]

#         try:
#             # Applique le inverse_transform uniquement sur ces colonnes
#             df_preds_exog = scaler_exog.inverse_transform(df_preds[cols_to_inverse])
#             # Remplace les colonnes standardisées par leur version inversée
#             for i, col in enumerate(cols_to_inverse):
#                 df_preds[col] = df_preds_exog[:, i]
#         except Exception as e:
#             print("Erreur lors de l'inverse_transform des exogènes:", e)

#         all_preds[poste] = df_preds

#         df_preds.to_csv(os.path.join(save_dir, "futur_predictions.csv"), index=False)
#         df_train_poste.reset_index(drop=True).to_csv(os.path.join(save_dir, "train_futur.csv"), index=False)
#         if verbose:
#             print(f"✅ Prédictions faites sur {len(df_preds)} semaines pour {poste} et train futur sauvegardé.")

#     return all_preds


################################################################ Niveau zone de traitement #############################################################################







# def optimize_single_zone_mlforecast_no_standard(
#     df_train, df_test, zone, target_col, target_col_name, h, base_path, n_trials, verbose,
#     exog_cols=None
# ):
#     # Fix all random seeds for reproducibility
#     RANDOM_STATE = 42
#     np.random.seed(RANDOM_STATE)
#     random.seed(RANDOM_STATE)

#     # Optuna seed for reproducibility
#     sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

#     df_train = df_train[df_train['unique_id'] == zone].copy()
#     df_test = df_test[df_test['unique_id'] == zone].copy()

#     # Always pass random_state to all models
#     MODELS = {
#         'lgbm': lambda **kwargs: LGBMRegressor(**kwargs),
#         'xgb': lambda **kwargs: XGBRegressor(**kwargs),
#         'cat': lambda **kwargs: CatBoostRegressor(**kwargs)
#     }
#     lags = list(range(1, 25))
#     lag_transforms = {lag: [RollingMean(window_size=lag), ExpandingMean(), RollingQuantile(window_size=lag, p=0.25), RollingQuantile(window_size=lag, p=0.75)] for lag in lags}
#     target_transforms = [
#         AutoDifferences(max_diffs=2),
#         LocalStandardScaler()
#     ]

#     if exog_cols is None:
#         exog_cols = [col for col in df_train.columns if col not in ['unique_id', 'ds', 'y']]

#     # Only encoding, no standardization
#     df_train_proc, df_test_proc, final_exog_cols, ohe_dict, categorical_cols = preprocess_exog_and_target2(df_train, df_test, exog_cols)

#     preds_dict = {}
#     print("df_test_proc shape:", df_test_proc.shape)
#     print("h:", h)

#     for model_name, model_fn in MODELS.items():
#         if verbose:
#             print(f"\nPremier fit simple pour {model_name.upper()} (sans tuning Optuna)...")
#         # Sélectionne les bons jeux de données et exogènes selon le modèle
#         if model_name == 'cat':
#             used_train = df_train.copy()
#             used_test = df_test.copy()
#             used_exog_cols = exog_cols  # colonnes originales, non encodées
#         else:
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()
#             used_exog_cols = final_exog_cols  # colonnes encodées
        
#         if model_name == 'cat':
#             cat_features = categorical_cols
#             model = model_fn(verbose=0, cat_features=cat_features)
#         elif model_name == 'lgbm':
#             model = model_fn(verbosity=-1)
#         elif model_name == 'xgb':
#             model = model_fn(verbosity=0)
#         else:
#             model = model_fn()

#         forecast = MLForecast(
#             models=[model],
#             freq='W-MON',
#             lags=lags,
#             lag_transforms=lag_transforms,
#             target_transforms=target_transforms,
#             date_features=['month', 'quarter']
#         )
#         forecast.fit(
#             used_train,
#             id_col='unique_id',
#             time_col='ds',
#             target_col='y',
#             static_features=[]
#         )
#         y_pred_scaled = forecast.predict(h=h, X_df=used_test)
#         pred_cols = [col for col in y_pred_scaled.columns if col not in ['unique_id', 'ds']]
#         if not pred_cols:
#             raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled.columns}")
#         y_pred_model_col = y_pred_scaled[pred_cols[0]].values[:h]
#         y_pred = y_pred_model_col
#         preds_dict[model_name] = y_pred

#     y_true = df_test['y'].values[:h]
#     best_model, best_model_metrics = select_best_model_by_vote(preds_dict, y_true, verbose=verbose)

#     if verbose:
#         print(f"\nOptimisation Optuna sur {best_model.upper()} ...")

#     def objective(trial):
#         config = get_optimization_config(best_model, trial)
#         config['random_state'] = RANDOM_STATE
#         if best_model == 'cat':
#             config['verbose'] = 0
#             used_train = df_train.copy()
#             used_test = df_test.copy()
#             used_exog_cols = exog_cols
#         elif best_model == 'lgbm':
#             config['verbosity'] = -1
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()
#             used_exog_cols = final_exog_cols
#         elif best_model == 'xgb':
#             config['verbosity'] = 0
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()
#             used_exog_cols = final_exog_cols
#         else:
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()
#             used_exog_cols = final_exog_cols

#         if best_model == 'cat':
#             cat_features = categorical_cols
#             model = MODELS[best_model](**config, cat_features=cat_features)
#         else:
#             model = MODELS[best_model](**config)

#         forecast = MLForecast(
#             models=[model],
#             freq='W-MON',
#             lags=lags,
#             lag_transforms=lag_transforms,
#             target_transforms=target_transforms
#         )
#         try:
#             forecast.fit(
#                 used_train,
#                 id_col='unique_id',
#                 time_col='ds',
#                 target_col='y',
#                 static_features=[]
#             )
#             y_pred_scaled = forecast.predict(h=h, X_df=used_test)
#             pred_cols = [col for col in y_pred_scaled.columns if col not in ['unique_id', 'ds']]
#             if not pred_cols:
#                 raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled.columns}")
#             y_pred_model_col = y_pred_scaled[pred_cols[0]].values[:h]
#             y_pred = y_pred_model_col
#             score = mean_squared_error(y_true, y_pred, squared=False)
#             return score
#         except Exception as e:
#             print(f"Erreur dans objective Optuna {best_model}: {e}")
#             return float('inf')

#     study = optuna.create_study(direction='minimize', sampler=sampler)
#     early_stop = PatienceEarlyStopper(patience=12)
#     study.optimize(objective, n_trials=n_trials, callbacks=[early_stop], show_progress_bar=False)
#     if verbose:
#         print(f"Best RMSE {best_model}: {study.best_value:.4f}")

#     best_config = get_optimization_config(best_model, study.best_trial)
#     best_config['random_state'] = RANDOM_STATE
#     if best_model == 'cat':
#         best_config['verbose'] = 0
#         used_train = df_train.copy()
#         used_test = df_test.copy()
#         used_exog_cols = exog_cols
#     elif best_model == 'lgbm':
#         best_config['verbosity'] = -1
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#     elif best_model == 'xgb':
#         best_config['verbosity'] = 0
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#     else:
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols

#     if best_model == 'cat':
#         cat_features = categorical_cols
#         model = MODELS[best_model](**best_config, cat_features=cat_features)
#     else:
#         model = MODELS[best_model](**best_config)
#     forecast = MLForecast(
#         models=[model],
#         freq='W-MON',
#         lags=lags,
#         lag_transforms=lag_transforms,
#         target_transforms=target_transforms
#     )
#     forecast.fit(
#         used_train,
#         id_col='unique_id',
#         time_col='ds',
#         target_col='y',
#         static_features=[]
#     )

#     y_pred_scaled_final = forecast.predict(h=h, X_df=used_test)

#     save_dir = f"{base_path}/models_{target_col_name}_{zone}/"
#     os.makedirs(save_dir, exist_ok=True)
#     if verbose:
#         print(f"Sauvegarde: {save_dir}")

#     df_train.to_csv(os.path.join(save_dir, "train_data.csv"), index=False)
#     joblib.dump(forecast, os.path.join(save_dir, f"{best_model}_mlforecast.pkl"))
#     with open(f"{save_dir}/best_params.json", "w") as f:
#         json.dump(best_config, f)

#     preds_df = df_test.reset_index(drop=True).copy()
#     preds_df['y_true'] = y_true
#     pred_cols_final = [col for col in y_pred_scaled_final.columns if col not in ['unique_id', 'ds']]
#     if not pred_cols_final:
#         raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled_final.columns}")

#     y_pred_model_col_final = y_pred_scaled_final[pred_cols_final[0]].values[:h]
#     y_pred_final = y_pred_model_col_final

#     preds_df['y_pred'] = y_pred_final

#     from scipy.stats import norm

#     # residuals = y_true - y_pred_final
#     # std_res = np.std(residuals, ddof=1)
#     # z_95 = norm.ppf(0.975)
#     # z_90 = norm.ppf(0.95)

#     # preds_df['lower_95'] = y_pred_final - z_95 * std_res
#     # preds_df['upper_95'] = y_pred_final + z_95 * std_res
#     # preds_df['lower_90'] = y_pred_final - z_90 * std_res
#     # preds_df['upper_90'] = y_pred_final + z_90 * std_res
#     preds_df.to_csv(f"{save_dir}/predictions.csv", index=False)

#     rmse_final = mean_squared_error(y_true, y_pred_final, squared=False)
#     mae_final = mean_absolute_error(y_true, y_pred_final)

#     best_model_metrics = {'RMSE': rmse_final, 'MAE': mae_final}
#     pd.DataFrame([best_model_metrics]).to_csv(f"{save_dir}/best_metrics.csv", index=False)

#     return {
#         'zone': zone,
#         'best_model': best_model,
#         'best_params': best_config,
#         'best_metrics': best_model_metrics,
#         'predictions': preds_df,
#         'mlforecast': forecast,
#         'save_dir': save_dir,
#         'ohe_dict': ohe_dict,
#         'categorical_cols': categorical_cols,
#         'final_exog_cols': final_exog_cols
#     }


# def optimize_models_for_all_zones_pas_split(
#     df_train, df_test, target_col_name , target_col='y',  h=12, 
#     base_path="./models", n_trials=20, verbose=True
# ):
#     """
#     Optimise un modèle MLForecast pour chaque zone de traitement avec Optuna.
#     """
#     if verbose:
#         print("="*60)
#         print("OPTIMISATION DES MODÈLES MLFORECAST PAR ZONE")
#         print("="*60)
    
#     zones_uniques = df_train['unique_id'].unique()
#     resultats_toutes_zones = {}
#     zones_reussies, zones_echecs = 0, 0
    
#     if verbose:
#         print(f"Zones détectées: {len(zones_uniques)}")
#         print(f"Données: Train={len(df_train)} obs, Test={len(df_test)} obs")
#         print(f"Paramètres: h={h}, n_trials={n_trials}")
    
#     for i, zone in enumerate(zones_uniques):
#         if verbose:
#             print(f"\n[{i+1}/{len(zones_uniques)}] ZONE: {zone}")
#             print("-"*40)
#         try:
#             df_train_zone = df_train[df_train['unique_id'] == zone].copy()
#             df_test_zone = df_test[df_test['unique_id'] == zone].copy()
#             if verbose:
#                 print(f"Données: Train={len(df_train_zone)}, Test={len(df_test_zone)}")
            
#             resultats_zone = optimize_single_zone_mlforecast_no_standard(
#                 df_train=df_train_zone,
#                 df_test=df_test_zone,
#                 zone=zone,
#                 target_col=target_col,
#                 target_col_name=target_col_name,
#                 h=h,
#                 base_path=base_path,
#                 n_trials=n_trials,
#                 verbose=verbose
#             )
#             resultats_toutes_zones[zone] = resultats_zone
#             zones_reussies += 1
            
#             if verbose:
#                 print(f"SUCCÈS - Meilleur: {resultats_zone['best_model']}")
#                 metrics = resultats_zone['best_metrics']
#                 print(f"Métriques: RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}")
#         except Exception as e:
#             zones_echecs += 1
#             if verbose:
#                 print(f"ÉCHEC: {e}")
#             continue
    
#     if verbose:
#         print(f"\n{'='*60}")
#         print("RÉSUMÉ FINAL")
#         print(f"{'='*60}")
#         print(f"Zones traitées: {len(zones_uniques)}")
#         print(f"Succès: {zones_reussies}")
#         print(f"Échecs: {zones_echecs}")
#         print(f"Taux réussite: {(zones_reussies/len(zones_uniques)*100):.1f}%")
#         if zones_reussies > 0:
#             modeles_count = {}
#             for result in resultats_toutes_zones.values():
#                 model = result['best_model']
#                 modeles_count[model] = modeles_count.get(model, 0) + 1
#             print(f"\nModèles gagnants:")
#             for model, count in sorted(modeles_count.items(), key=lambda x: x[1], reverse=True):
#                 print(f"  {model}: {count} zones ({count/zones_reussies*100:.1f}%)")
#     print(f"\nOptimisation terminée !")
#     return resultats_toutes_zones




# ########################################## Modèle futur ###########################################




# def predict_futur_for_all_zones1(
#     df_train, df_test, target_col_name, base_path="./models", h=4, verbose=True, exog_cols=None
# ):
#     import os
#     import json
#     # Fixe une seed pour la reproducité des résultats.
#     RANDOM_STATE = 42
    
#     np.random.seed(RANDOM_STATE)
#     random.seed(RANDOM_STATE)

#     MODELS = {
#         'lgbm': LGBMRegressor,
#         'xgb': XGBRegressor,
#         'cat': CatBoostRegressor,
#     }

#     zones_uniques = df_train['unique_id'].unique()
#     all_preds = {}

#     for zone in zones_uniques:
#         if verbose:
#             print(f"\n=== Zone {zone} ===")

#         save_dir = f"{base_path}/models_{target_col_name}_{zone}/"

#         # 1. Identifier le modèle gagnant
#         best_model_name = None
#         for model_name in ["xgb", "lgbm", "cat"]:
#             model_file = os.path.join(save_dir, f"{model_name}_mlforecast.pkl")
#             if os.path.exists(model_file):
#                 best_model_name = model_name
#                 break
#         if best_model_name is None:
#             print(f" Aucun modèle trouvé pour {zone}")
#             continue

#         # 2. Charger les meilleurs hyperparams
#         import json, os
#         with open(os.path.join(save_dir, "best_params.json")) as f:
#             best_params = json.load(f)

#         # Always add random_state to model config if possible
#         best_params['random_state'] = RANDOM_STATE
#         # Set verbosity if not already set (for CatBoost/LGBM)
#         if best_model_name == 'cat':
#             best_params.setdefault('verbose', 0)
#         elif best_model_name == 'lgbm':
#             best_params.setdefault('verbosity', -1) 
#         elif best_model_name == 'xgb':
#             best_params.setdefault('verbosity', 0) 

#         # 3. (Re)fit preprocessing sur toutes les données train disponibles
#         df_train_zone = df_train[df_train['unique_id'] == zone].copy()
#         df_test_zone = df_test[df_test['unique_id'] == zone].copy()

#         cols = exog_cols if exog_cols is not None else [
#             col for col in df_train_zone.columns if col not in ['unique_id', 'ds', 'y']
#         ]
#         exog_cols = [col for col in df_train_zone.columns if col not in ['unique_id', 'ds', 'y']]

#         # Utilisation de la nouvelle fonction d'encodage (pas de scaling)
#         df_train_proc, df_test_proc, final_exog_cols, ohe_dict, categorical_cols = preprocess_exog_and_target2(
#             df_train_zone, df_test_zone, cols
#         )

#         # === Vérifications exogènes ===
#         print("\n=== [EXOGENES DEBUG] ===")
#         print("Exogènes finales utilisées:", final_exog_cols)
#         missing_cols_train = [col for col in final_exog_cols if col not in df_train_proc.columns]
#         missing_cols_test = [col for col in final_exog_cols if col not in df_test_proc.columns]
#         if missing_cols_train:
#             print(f"❌ Colonnes exogènes manquantes dans TRAIN: {missing_cols_train}")
#         if missing_cols_test:
#             print(f"❌ Colonnes exogènes manquantes dans TEST: {missing_cols_test}")
#         print("df_train_proc shape:", df_train_proc.shape)
#         print("df_test_proc shape:", df_test_proc.shape)
#         # Vérifie les NaN dans les exogènes
#         nans_train = df_train_proc[final_exog_cols].isna().sum().sum()
#         nans_test = df_test_proc[final_exog_cols].isna().sum().sum()
#         print(f"NaN dans exogènes TRAIN: {nans_train}, TEST: {nans_test}")

#         # === DEBUG SAISONNALITE, ROLLING, IMPORTANCES ===
#         print("\n=== [IMPORTANCES MODELE] ===")
#         model_cls = MODELS[best_model_name]
#         if best_model_name == 'cat':
#             used_train = df_train_zone.copy()
#             used_test = df_test_zone.copy()
#             cat_features = categorical_cols
#             model = model_cls(**best_params, cat_features=cat_features)
#         else:
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()
#             model = model_cls(**best_params)


#         lags = list(range(1, 13))
#         lag_transforms = {lag: [RollingMean(window_size=lag), ExpandingMean(), RollingQuantile(window_size=lag, p= 0.5), ExponentiallyWeightedMean(alpha = 0.35)] for lag in lags}
#         target_transforms=[
#             AutoDifferences(max_diffs= 5),
#             LocalStandardScaler()
#             ]
#         forecast = MLForecast(
#             models=[model],
#             freq='W-MON',
#             lags=lags,
#             lag_transforms=lag_transforms,
#             target_transforms=target_transforms,
#             date_features=['month', 'quarter']
#         )
#         forecast.fit(
#             used_train,
#             id_col='unique_id',
#             time_col='ds',
#             target_col='y',
#             static_features=[]
#         )
#         try:
#             if hasattr(model, "feature_importances_"):
#                 print("Feature importances :")
#                 feature_names = [c for c in df_train_proc.columns if c not in ['unique_id', 'ds', 'y' ]]
#                 print(
#                     sorted(
#                         zip(feature_names, model.feature_importances_),
#                         key=lambda x: -x[1]
#                     )
#                 )
#             else:
#                 print("Le modèle ne fournit pas de feature_importances_.")
#         except Exception as e:
#             print("Erreur lors de l'affichage des importance features :", e)

#         df_test_h = used_test.groupby('unique_id').head(h)
#         print( "Colonne test ", df_test_h.columns)

#         y_pred_scaled = forecast.predict(h=h, X_df=df_test_h)
#         print("\n=== [DEBUG MLForecast LAGS] ===")
#         try:
#             print("Train features MLForecast:", forecast.current_features_)
#         except Exception as e:
#             print("Impossible d'afficher les features générées par MLForecast:", e)
#         pred_cols = [col for col in y_pred_scaled.columns if col not in ['unique_id', 'ds']]
#         y_pred_model_col = y_pred_scaled[pred_cols[0]].values

#         # Pas d'inverse transform, ici on garde y_pred tel quel
#         y_pred = y_pred_model_col

#         df_preds = df_test_h.reset_index(drop=True).copy()
#         df_preds['y_pred'] = y_pred
#         df_preds['type_prediction'] = 'futur'

#         # Pas de bloc d'inverse_transform des exogènes ici !

#         df_preds.to_csv(os.path.join(save_dir, "futur_predictions.csv"), index=False)
#         df_train_zone.reset_index(drop=True).to_csv(os.path.join(save_dir, "train_futur.csv"), index=False)
#         all_preds[zone] = df_preds

#         if verbose:
#             print(f"✅ Prédictions faites sur {len(df_preds)} semaines pour {zone} et train futur sauvegardé.")

#     return all_preds







############### Modularisation des fonctions

# def optimize_model_zone(
#     df_train, df_test, zone, target_col, target_col_name, h, base_path, n_trials, verbose,
#     exog_cols=None
# ):
#     """
#     Sélectionne le meilleur modèle, fait le tuning Optuna,
#     et enregistre directement le modèle du meilleur trial Optuna (.pkl) + ses hyperparams (.json)
#     SANS refit final.
#     """
#     import numpy as np
#     import random
#     import os
#     import optuna
#     import joblib
#     import json

#     RANDOM_STATE = 42
#     np.random.seed(RANDOM_STATE)
#     random.seed(RANDOM_STATE)
#     sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

#     df_train = df_train[df_train['unique_id'] == zone].copy()
#     df_test = df_test[df_test['unique_id'] == zone].copy()

#     MODELS = {
#         'lgbm': lambda **kwargs: LGBMRegressor(**kwargs),
#         'xgb': lambda **kwargs: XGBRegressor(**kwargs),
#         'cat': lambda **kwargs: CatBoostRegressor(**kwargs)
#     }
#     lags = list(range(1, 25))
#     lag_transforms = {lag: [RollingMean(window_size=lag), ExpandingMean(), RollingQuantile(window_size=lag, p=0.25), RollingQuantile(window_size=lag, p=0.75)] for lag in lags}
#     target_transforms = [
#         AutoDifferences(max_diffs=2),
#         LocalStandardScaler()
#     ]

#     if exog_cols is None:
#         exog_cols = [col for col in df_train.columns if col not in ['unique_id', 'ds', 'y']]

#     df_train_proc, df_test_proc, final_exog_cols, ohe_dict, categorical_cols = preprocess_exog_and_target2(df_train, df_test, exog_cols)

#     preds_dict = {}
#     for model_name, model_fn in MODELS.items():
#         if verbose:
#             print(f"\nPremier fit simple pour {model_name.upper()} (sans tuning Optuna)...")
#         if model_name == 'cat':
#             used_train = df_train.copy()
#             used_test = df_test.copy()
#             used_exog_cols = exog_cols
#         else:
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()
#             used_exog_cols = final_exog_cols

#         if model_name == 'cat':
#             cat_features = categorical_cols
#             model = model_fn(verbose=0, cat_features=cat_features)
#         elif model_name == 'lgbm':
#             model = model_fn(verbosity=-1)
#         elif model_name == 'xgb':
#             model = model_fn(verbosity=0)
#         else:
#             model = model_fn()

#         forecast = MLForecast(
#             models=[model],
#             freq='W-MON',
#             lags=lags,
#             lag_transforms=lag_transforms,
#             target_transforms=target_transforms,
#             date_features=['month', 'quarter']
#         )
#         forecast.fit(
#             used_train,
#             id_col='unique_id',
#             time_col='ds',
#             target_col='y',
#             static_features=[]
#         )
#         y_pred_scaled = forecast.predict(h=h, X_df=used_test)
#         pred_cols = [col for col in y_pred_scaled.columns if col not in ['unique_id', 'ds']]
#         if not pred_cols:
#             raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled.columns}")
#         y_pred_model_col = y_pred_scaled[pred_cols[0]].values[:h]
#         y_pred = y_pred_model_col
#         preds_dict[model_name] = y_pred

#     y_true = df_test['y'].values[:h]
#     best_model, best_model_metrics = select_best_model_by_vote(preds_dict, y_true, verbose=verbose)

#     if verbose:
#         print(f"\nOptimisation Optuna sur {best_model.upper()} ...")

#     best_forecast = [None]  # mutable container to store best model
#     def objective(trial):
#         config = get_optimization_config(best_model, trial)
#         config['random_state'] = RANDOM_STATE
#         if best_model == 'cat':
#             config['verbose'] = 0
#             used_train = df_train.copy()
#             used_test = df_test.copy()
#         elif best_model == 'lgbm':
#             config['verbosity'] = -1
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()
#         elif best_model == 'xgb':
#             config['verbosity'] = 0
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()
#         else:
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()

#         if best_model == 'cat':
#             cat_features = categorical_cols
#             model = MODELS[best_model](**config, cat_features=cat_features)
#         else:
#             model = MODELS[best_model](**config)

#         forecast = MLForecast(
#             models=[model],
#             freq='W-MON',
#             lags=lags,
#             lag_transforms=lag_transforms,
#             target_transforms=target_transforms
#         )
#         try:
#             forecast.fit(
#                 used_train,
#                 id_col='unique_id',
#                 time_col='ds',
#                 target_col='y',
#                 static_features=[]
#             )
#             y_pred_scaled = forecast.predict(h=h, X_df=used_test)
#             pred_cols = [col for col in y_pred_scaled.columns if col not in ['unique_id', 'ds']]
#             if not pred_cols:
#                 raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled.columns}")
#             y_pred_model_col = y_pred_scaled[pred_cols[0]].values[:h]
#             y_pred = y_pred_model_col
#             score = mean_squared_error(y_true, y_pred, squared=False)
#             # si trial est le best trial, on sauvegarde le modèle
#             if not hasattr(objective, "best_score") or score < objective.best_score:
#                 objective.best_score = score
#                 best_forecast[0] = forecast
#             return score
#         except Exception as e:
#             print(f"Erreur dans objective Optuna {best_model}: {e}")
#             return float('inf')

#     study = optuna.create_study(direction='minimize', sampler=sampler)
#     early_stop = PatienceEarlyStopper(patience=12)
#     study.optimize(objective, n_trials=n_trials, callbacks=[early_stop], show_progress_bar=False)
#     if verbose:
#         print(f"Best RMSE {best_model}: {study.best_value:.4f}")

#     best_config = get_optimization_config(best_model, study.best_trial)
#     best_config['random_state'] = RANDOM_STATE

#     save_dir = f"{base_path}/models_{target_col_name}_{zone}/"
#     os.makedirs(save_dir, exist_ok=True)
#     if verbose:
#         print(f"Sauvegarde: {save_dir}")

#     # Sauvegarde du modèle du meilleur trial Optuna (pas de refit !)
#     joblib.dump(best_forecast[0], os.path.join(save_dir, f"{best_model}_mlforecast.pkl"))
#     # Sauvegarde des meilleurs paramètres
#     with open(f"{save_dir}/best_params.json", "w") as f:
#         json.dump(best_config, f)

#     return {
#         'zone': zone,
#         'best_model': best_model,
#         'best_params': best_config,
#         'save_dir': save_dir
#     }

# def optimize_models_for_all_zones(
#     df_train, df_test, target_col_name , target_col='y',  h=12, 
#     base_path="./models", n_trials=20, verbose=True
# ):
#     """
#     OPTIMISE ET SAUVEGARDE UN MODÈLE MLFORECAST PAR ZONE (inclut .pkl et .json)
#     """
#     if verbose:
#         print("="*60)
#         print("OPTIMISATION DES MODÈLES MLFORECAST PAR ZONE")
#         print("="*60)
    
#     zones_uniques = df_train['unique_id'].unique()
#     resultats_toutes_zones = {}
#     zones_reussies, zones_echecs = 0, 0
    
#     for i, zone in enumerate(zones_uniques):
#         if verbose:
#             print("\n" + "="*50)
#             print(f"Optimisation pour la zone : {zone}")
#             print("="*50)
            
#         try:
#             df_train_zone = df_train[df_train['unique_id'] == zone].copy()
#             df_test_zone = df_test[df_test['unique_id'] == zone].copy()
            
#             resultats_zone = optimize_model_zone(
#                 df_train=df_train_zone,
#                 df_test=df_test_zone,
#                 zone=zone,
#                 target_col=target_col,
#                 target_col_name=target_col_name,
#                 h=h,
#                 base_path=base_path,
#                 n_trials=n_trials,
#                 verbose=verbose
#             )
#             resultats_toutes_zones[zone] = resultats_zone
#             zones_reussies += 1
#             if verbose:
#                 print(f"SUCCÈS - Meilleur: {resultats_zone['best_model']}")
#         except Exception as e:
#             zones_echecs += 1
#             if verbose:
#                 print(f"ÉCHEC: {e}")
#             continue
    
#     if verbose:
#         print(f"\n{'='*60}\nOptimisation terminée !\n{'='*60}")
#         print(f"Zones traitées: {len(zones_uniques)} | Succès: {zones_reussies} | Échecs: {zones_echecs}")
#     return resultats_toutes_zones


# def test_model_zone(
#     df_train, df_test, zone, target_col, target_col_name, h, base_path, verbose,
#     exog_cols=None
# ):
#     """
#     Charge best_params.json, ré-entraine le modèle optimisé avec les bons hyperparams,
#     prédit sur le test, sauvegarde (en écrasant) le modèle .pkl, predictions.csv, train_data.csv, best_metrics.csv.
#     """
#     import os
#     import joblib
#     import json
#     import pandas as pd
#     from sklearn.metrics import mean_squared_error, mean_absolute_error

#     # Chemin du dossier zone
#     save_dir = f"{base_path}/models_{target_col_name}_{zone}/"
#     if not os.path.exists(save_dir):
#         raise ValueError(f"Dossier {save_dir} introuvable, lance d'abord optimize_model_zone")

#     # 1. Charger le nom du meilleur modèle (par le nom du fichier .pkl existant)
#     best_model_name = None
#     for model_name in ["xgb", "lgbm", "cat"]:
#         model_file = os.path.join(save_dir, f"{model_name}_mlforecast.pkl")
#         if os.path.exists(model_file):
#             best_model_name = model_name
#             model_pkl_path = model_file
#             break
#     if best_model_name is None:
#         raise FileNotFoundError(f"Aucun modèle trouvé pour {zone} dans {save_dir}")

#     # 2. Charger les meilleurs hyperparams
#     with open(os.path.join(save_dir, "best_params.json")) as f:
#         best_config = json.load(f)

#     # 3. Préparation des données
#     df_train = df_train[df_train['unique_id'] == zone].copy()
#     df_test = df_test[df_test['unique_id'] == zone].copy()

#     # 3.1 Préparation des exogènes
#     if exog_cols is None:
#         exog_cols = [col for col in df_train.columns if col not in ['unique_id', 'ds', 'y']]

#     df_train_proc, df_test_proc, final_exog_cols, ohe_dict, categorical_cols = preprocess_exog_and_target2(
#         df_train, df_test, exog_cols
#     )

#     # 4. Reconstruction et fit du modèle avec les bons hyperparams
#     # Définir ici tes objets MODELS, lags, lag_transforms, target_transforms comme dans optimize_model_zone
#     MODELS = {
#         'lgbm': lambda **kwargs: LGBMRegressor(**kwargs),
#         'xgb': lambda **kwargs: XGBRegressor(**kwargs),
#         'cat': lambda **kwargs: CatBoostRegressor(**kwargs)
#     }
#     lags = list(range(1, 25))
#     lag_transforms = {lag: [RollingMean(window_size=lag), ExpandingMean(), RollingQuantile(window_size=lag, p=0.25), RollingQuantile(window_size=lag, p=0.75)] for lag in lags}
#     target_transforms = [
#         AutoDifferences(max_diffs=2),
#         LocalStandardScaler()
#     ]

#     if best_model_name == 'cat':
#         best_config['verbose'] = 0
#         used_train = df_train.copy()
#         used_test = df_test.copy()
#         used_exog_cols = exog_cols
#         cat_features = categorical_cols
#         model = MODELS[best_model_name](**best_config, cat_features=cat_features)
#     elif best_model_name == 'lgbm':
#         best_config['verbosity'] = -1
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)
#     elif best_model_name == 'xgb':
#         best_config['verbosity'] = 0
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)
#     else:
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)

#     forecast = MLForecast(
#         models=[model],
#         freq='W-MON',
#         lags=lags,
#         lag_transforms=lag_transforms,
#         target_transforms=target_transforms,
#         date_features=['month', 'quarter']
#     )
#     forecast.fit(
#         used_train,
#         id_col='unique_id',
#         time_col='ds',
#         target_col='y',
#         static_features=[], 
#         prediction_intervals= PredictionIntervals(n_windows=4, h=h)
#     )

#     # 5. Prédiction sur le test
#     y_pred_scaled_final = forecast.predict(h=h, X_df=used_test, level= [90, 95])
#     print(y_pred_scaled_final.columns)
#     y_true = used_test['y'].values[:h]
#     pred_cols_final = [col for col in y_pred_scaled_final.columns if col not in ['unique_id', 'ds']+ [f'lo-{lvl}' for lvl in [90,95]] + [f'hi-{lvl}' for lvl in [90,95]]]
#     if not pred_cols_final:
#         raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled_final.columns}")

#     y_pred_final = y_pred_scaled_final[pred_cols_final[0]].values[:h]

#     # 6. Sauvegarde du modèle refitté (écrase l'ancien)
#     joblib.dump(forecast, model_pkl_path)
#     if verbose:
#         print(f"Le modèle {best_model_name} (zone {zone}) a été ré-entraîné et sauvegardé dans {model_pkl_path} (ancien modèle écrasé)")

#     # 7. Sauvegardes des résultats
#     preds_df = df_test.reset_index(drop=True).copy()
#     preds_df['y_true'] = y_true
#     preds_df['y_pred'] = y_pred_final

#     # Ajoute les intervalles pour chaque niveau
#     for lvl in [90, 95]:
#         lo_col = f'{model.__class__.__name__}-lo-{lvl}'
#         hi_col = f'{model.__class__.__name__}-hi-{lvl}'
#         if lo_col in y_pred_scaled_final.columns and hi_col in y_pred_scaled_final.columns:
#             preds_df[f'ic_lo_{lvl}'] = y_pred_scaled_final[lo_col].values[:h]
#             preds_df[f'ic_hi_{lvl}'] = y_pred_scaled_final[hi_col].values[:h]


#     preds_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
#     df_train.to_csv(os.path.join(save_dir, "train_data.csv"), index=False)

#     rmse_final = mean_squared_error(y_true, y_pred_final, squared=False)
#     mae_final = mean_absolute_error(y_true, y_pred_final)
#     best_model_metrics = {'RMSE': rmse_final, 'MAE': mae_final}
#     pd.DataFrame([best_model_metrics]).to_csv(os.path.join(save_dir, "best_metrics.csv"), index=False)

#     if verbose:
#         print(f"Test et sauvegardes terminés pour {zone} - {target_col_name}")

#     return {
#         'zone': zone,
#         'best_model': best_model_name,
#         'best_params': best_config,
#         'best_metrics': best_model_metrics,
#         'predictions': preds_df,
#         'mlforecast': forecast,
#         'save_dir': save_dir
#     }

# def test_model_zone(
#     df_train, df_test, zone, target_col, target_col_name, h, base_path, verbose,
#     exog_cols=None
# ):
#     """
#     Charge best_params.json, ré-entraine le modèle optimisé avec les bons hyperparams,
#     prédit sur le test, sauvegarde (en écrasant) le modèle .pkl, predictions.csv, train_data.csv, best_metrics.csv.
#     """
    

#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#     # Chemin du dossier zone
#     save_dir = f"{base_path}/models_{target_col_name}_{zone}/"
#     if not os.path.exists(save_dir):
#         raise ValueError(f"Dossier {save_dir} introuvable, lance d'abord optimize_model_zone")

#     # 1. Charger le nom du meilleur modèle (par le nom du fichier .pkl existant)
#     best_model_name = None
#     for model_name in ["xgb", "lgbm", "cat"]:
#         model_file = os.path.join(save_dir, f"{model_name}_mlforecast.pkl")
#         if os.path.exists(model_file):
#             best_model_name = model_name
#             model_pkl_path = model_file
#             break
#     if best_model_name is None:
#         raise FileNotFoundError(f"Aucun modèle trouvé pour {zone} dans {save_dir}")

#     # 2. Charger les meilleurs hyperparams
#     with open(os.path.join(save_dir, "best_params.json")) as f:
#         best_config = json.load(f)

#     # 3. Préparation des données
#     df_train = df_train[df_train['unique_id'] == zone].copy()
#     df_test = df_test[df_test['unique_id'] == zone].copy()

#     # 3.1 Préparation des exogènes
#     if exog_cols is None:
#         exog_cols = [col for col in df_train.columns if col not in ['unique_id', 'ds', 'y']]

#     df_train_proc, df_test_proc, final_exog_cols, ohe_dict, categorical_cols = preprocess_exog_and_target2(
#         df_train, df_test, exog_cols
#     )

#     # 4. Reconstruction et fit du modèle avec les bons hyperparams
#     # Définir ici tes objets MODELS, lags, lag_transforms, target_transforms comme dans optimize_model_zone
#     MODELS = {
#         'lgbm': lambda **kwargs: LGBMRegressor(**kwargs),
#         'xgb': lambda **kwargs: XGBRegressor(**kwargs),
#         'cat': lambda **kwargs: CatBoostRegressor(**kwargs)
#     }
#     lags = list(range(1, 25))
#     lag_transforms = {lag: [RollingMean(window_size=lag), ExponentiallyWeightedMean(alpha=0.9), RollingQuantile(window_size=lag, p=0.25), RollingQuantile(window_size=lag, p=0.75)] for lag in lags}
#     target_transforms = [
#         AutoDifferences(max_diffs=2),
#         LocalStandardScaler()
#     ]

#     if best_model_name == 'cat':
#         best_config['verbose'] = 0
#         used_train = df_train.copy()
#         used_test = df_test.copy()
#         used_exog_cols = exog_cols
#         cat_features = categorical_cols
#         model = MODELS[best_model_name](**best_config, cat_features=cat_features)
#     elif best_model_name == 'lgbm':
#         best_config['verbosity'] = -1
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)
#     elif best_model_name == 'xgb':
#         best_config['verbosity'] = 0
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)
#     else:
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)

#     forecast = MLForecast(
#         models=[model],
#         freq='W-MON',
#         lags=lags,
#         lag_transforms=lag_transforms,
#         target_transforms=target_transforms,
#         date_features=['month', 'quarter']
#     )
#     forecast.fit(
#         used_train,
#         id_col='unique_id',
#         time_col='ds',
#         target_col='y',
#         static_features=[], 
#         prediction_intervals= PredictionIntervals(n_windows=4, h=h)
#     )
    

#     # 5. Prédiction sur le test
#     y_pred_scaled_final = forecast.predict(h=h, X_df=used_test, level= [90, 95])
#     y_true = used_test['y'].values[:h]
#     pred_cols_final = [col for col in y_pred_scaled_final.columns if col not in ['unique_id', 'ds']+ [f'lo-{lvl}' for lvl in [90,95]] + [f'hi-{lvl}' for lvl in [90,95]]]
#     if not pred_cols_final:
#         raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled_final.columns}")

#     y_pred_final = y_pred_scaled_final[pred_cols_final[0]].values[:h]

#     # 6. Sauvegarde du modèle refitté (écrase l'ancien)
#     joblib.dump(forecast, model_pkl_path)
#     if verbose:
#         logging.info(f"Le modèle {best_model_name} (zone {zone}) a été ré-entraîné et sauvegardé dans {model_pkl_path} (ancien modèle écrasé)")

#     # 7. Sauvegardes des résultats
#     preds_df = df_test.reset_index(drop=True).copy()
#     preds_df['y_true'] = y_true
#     preds_df['y_pred'] = y_pred_final

#     # Ajoute les intervalles pour chaque niveau
#     for lvl in [90, 95]:
#         lo_col = f'{model.__class__.__name__}-lo-{lvl}'
#         hi_col = f'{model.__class__.__name__}-hi-{lvl}'
#         if lo_col in y_pred_scaled_final.columns and hi_col in y_pred_scaled_final.columns:
#             preds_df[f'ic_lo_{lvl}'] = y_pred_scaled_final[lo_col].values[:h]
#             preds_df[f'ic_hi_{lvl}'] = y_pred_scaled_final[hi_col].values[:h]


#     # Sélectionne uniquement les colonnes demandées (en gardant celles existantes)
#     colonnes = ["unique_id", "ds", "Annee", "Semaine","y", "y_pred", "ic_lo_90", "ic_hi_90", "ic_lo_95", "ic_hi_95"]
#     # Ne garde que celles qui existent déjà dans preds_df
#     colonnes = [c for c in colonnes if c in preds_df.columns]
    

#     rmse_final = mean_squared_error(y_true, y_pred_final, squared=False)
#     mae_final = mean_absolute_error(y_true, y_pred_final)
#     best_model_metrics = {'RMSE': rmse_final, 'MAE': mae_final}

#     # NOUVEAU DOSSIER data/models_data/models_target_colname_zone/
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#     data_save_dir = os.path.join(project_root, "data", "models_data", f"models_{target_col_name}_{zone}")
#     os.makedirs(data_save_dir, exist_ok=True)

#     # Sauvegarder dans le NOUVEAU dossier data
#     preds_df[colonnes].to_csv(os.path.join(data_save_dir, "predictions.csv"), index=False)
#     df_train.to_csv(os.path.join(data_save_dir, "train_data.csv"), index=False)
#     pd.DataFrame([best_model_metrics]).to_csv(os.path.join(data_save_dir, "best_metrics.csv"), index=False)


#     # pd.DataFrame([best_model_metrics]).to_csv(os.path.join(save_dir, "best_metrics.csv"), index=False)
#     # preds_df[colonnes].to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
#     # df_train.to_csv(os.path.join(save_dir, "train_data.csv"), index=False)

#     if verbose:
#         logging.info(f"Test et sauvegardes terminés pour {zone} - {target_col_name}")

#     return {
#         'zone': zone,
#         'best_model': best_model_name,
#         'best_params': best_config,
#         'best_metrics': best_model_metrics,
#         'predictions': preds_df,
#         'mlforecast': forecast,
#         'save_dir': save_dir
#     }
# def test_model_zone_ML(
#     df_train, df_test, zone, target_col, target_col_name, h, base_path, verbose,
#     exog_cols=None,
#     ohe_dict=None,  # À passer si déjà appris lors de l'optimisation, sinon None
#     categorical_cols=None  # Idem pour CatBoost
# ):
#     """
#     Teste et met à jour un modèle MLForecast sauvegardé pour une zone.
#     Gère l'encodage des exogènes selon le type de modèle (xgb/lgbm/cat).
#     Sauvegarde le modèle mis à jour et les résultats.
#     """
#     import os
#     import json
#     import logging
#     import pandas as pd
#     from mlforecast import MLForecast
#     from sklearn.metrics import mean_squared_error, mean_absolute_error

#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#     save_dir = f"{base_path}/models_{target_col_name}_{zone}/"
#     if not os.path.exists(save_dir):
#         raise ValueError(f"Dossier {save_dir} introuvable, lance d'abord optimize_model_zone")

#     # 1. Charger le modèle MLForecast déjà entraîné
#     forecast = MLForecast.load(save_dir)

#     # 2. Identifier le type de modèle (xgb/lgbm/cat)
#     model_name = forecast.models[0].__class__.__name__.lower()
#     if 'xgb' in model_name:
#         main_model = 'xgb'
#     elif 'lgbm' in model_name:
#         main_model = 'lgbm'
#     elif 'cat' in model_name:
#         main_model = 'cat'
#     else:
#         main_model = 'other'

#     # 3. Préparation des données train/test (par zone)
#     df_train_zone = df_train[df_train['unique_id'] == zone].copy()
#     df_test_zone = df_test[df_test['unique_id'] == zone].copy()

#     # 4. Préparation des exogènes et encodage si nécessaire
#     if exog_cols is None:
#         exog_cols = [col for col in df_train_zone.columns if col not in ['unique_id', 'ds', 'y']]
#     if main_model in ['xgb', 'lgbm']:
#         # Encodage identique à l'optimisation (en passant ohe_dict si besoin pour consistance)
#         df_train_proc, df_test_proc, final_exog_cols, ohe_dict_out, categorical_cols_out = preprocess_exog_and_target2(
#             df_train_zone, df_test_zone, exog_cols, ohe_dict=ohe_dict
#         )
#         used_train = df_train_proc
#         used_test = df_test_proc
#     elif main_model == 'cat':
#         # Pas d'encodage, CatBoost gère les colonnes catégorielles
#         used_train = df_train_zone
#         used_test = df_test_zone
#         # Récupérer la liste des colonnes catégorielles si non fournie
#         if categorical_cols is None:
#             _, _, _, _, categorical_cols = preprocess_exog_and_target2(df_train_zone, df_test_zone, exog_cols)
#         cat_features = categorical_cols
#         # S'assurer que le modèle CatBoost connaît les colonnes catégorielles
#         forecast.models[0].set_params(cat_features=cat_features)
#     else:
#         used_train = df_train_zone
#         used_test = df_test_zone

#     # 5. Update du modèle avec l'historique encodé ou non
#     forecast.update(used_train)
#     forecast.save(save_dir)

#     # 6. Prédiction sur le test (avec encodage si besoin)
#     y_pred_scaled_final = forecast.predict(h=h, X_df=used_test, level=[90, 95])
#     y_true = used_test['y'].values[:h]
#     pred_cols_final = [
#         col for col in y_pred_scaled_final.columns
#         if col not in ['unique_id', 'ds'] +
#         [f'lo-{lvl}' for lvl in [90, 95]] +
#         [f'hi-{lvl}' for lvl in [90, 95]]
#     ]
#     if not pred_cols_final:
#         raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled_final.columns}")

#     y_pred_final = y_pred_scaled_final[pred_cols_final[0]].values[:h]

#     # 7. Sauvegardes des résultats
#     preds_df = used_test.reset_index(drop=True).copy()
#     preds_df['y_true'] = y_true
#     preds_df['y_pred'] = y_pred_final

#     for lvl in [90, 95]:
#         lo_col = f'{forecast.models[0].__class__.__name__}-lo-{lvl}'
#         hi_col = f'{forecast.models[0].__class__.__name__}-hi-{lvl}'
#         if lo_col in y_pred_scaled_final.columns and hi_col in y_pred_scaled_final.columns:
#             preds_df[f'ic_lo_{lvl}'] = y_pred_scaled_final[lo_col].values[:h]
#             preds_df[f'ic_hi_{lvl}'] = y_pred_scaled_final[hi_col].values[:h]

#     colonnes = [
#         "unique_id", "ds", "Annee", "Semaine", "y", "y_pred",
#         "ic_lo_90", "ic_hi_90", "ic_lo_95", "ic_hi_95"
#     ]
#     colonnes = [c for c in colonnes if c in preds_df.columns]

#     rmse_final = mean_squared_error(y_true, y_pred_final, squared=False)
#     mae_final = mean_absolute_error(y_true, y_pred_final)
#     best_model_metrics = {'RMSE': rmse_final, 'MAE': mae_final}

#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#     data_save_dir = os.path.join(project_root, "data", "models_data", f"models_{target_col_name}_{zone}")
#     os.makedirs(data_save_dir, exist_ok=True)

#     preds_df[colonnes].to_csv(os.path.join(data_save_dir, "predictions.csv"), index=False)
#     used_train.to_csv(os.path.join(data_save_dir, "train_data.csv"), index=False)
#     pd.DataFrame([best_model_metrics]).to_csv(os.path.join(data_save_dir, "best_metrics.csv"), index=False)

#     if verbose:
#         logging.info(f"Test et sauvegardes terminés pour {zone} - {target_col_name}")

#     best_params_path = os.path.join(save_dir, "best_params.json")
#     best_config = {}
#     if os.path.exists(best_params_path):
#         with open(best_params_path) as f:
#             best_config = json.load(f)

#     return {
#         'zone': zone,
#         'best_params': best_config,
#         'best_metrics': best_model_metrics,
#         'predictions': preds_df,
#         'mlforecast': forecast,
#         'save_dir': save_dir
#     }
# def test_model_zone_ML2(
#     df_train, df_test, zone, target_col, target_col_name, h, base_path, verbose,
#     exog_cols=None,
#     ohe_dict=None,  # À passer si déjà appris lors de l'optimisation, sinon None
#     categorical_cols=None  # Idem pour CatBoost
# ):
#     """
#     Teste et met à jour un modèle MLForecast sauvegardé pour une zone.
#     Gère l'encodage des exogènes selon le type de modèle (xgb/lgbm/cat).
#     Sauvegarde le modèle mis à jour et les résultats.
#     """
#     import os
#     import json
#     import logging
#     import pandas as pd
#     import traceback
#     from mlforecast import MLForecast
#     from sklearn.metrics import mean_squared_error, mean_absolute_error

#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#     try:
#         save_dir = f"{base_path}/models_{target_col_name}_{zone}/"
#         if not os.path.exists(save_dir):
#             raise ValueError(f"Dossier {save_dir} introuvable, lance d'abord optimize_model_zone")

#         # 1. Charger le modèle MLForecast déjà entraîné
#         forecast = MLForecast.load(save_dir)

#         # 2. Identifier le type de modèle (xgb/lgbm/cat)
#         first_model = list(forecast.models.values())[0]
#         model_name = first_model.__class__.__name__.lower()

#         if 'xgb' in model_name:
#             main_model = 'xgb'
#         elif 'lgbm' in model_name:
#             main_model = 'lgbm'
#         elif 'cat' in model_name:
#             main_model = 'cat'
#         else:
#             main_model = 'other'

#         # 3. Préparation des données train/test (par zone)
#         df_train_zone = df_train[df_train['unique_id'] == zone].copy()
#         df_test_zone = df_test[df_test['unique_id'] == zone].copy()
        

#         # 4. Préparation des exogènes et encodage si nécessaire
#         if exog_cols is None:
#             exog_cols = [col for col in df_train_zone.columns if col not in ['unique_id', 'ds', 'y']]
#         if main_model in ['xgb', 'lgbm']:
#             # Encodage identique à l'optimisation (en passant ohe_dict si besoin pour consistance)
#             df_train_proc, df_test_proc, final_exog_cols, ohe_dict_out, categorical_cols_out = preprocess_exog_and_target2(
#                 df_train_zone, df_test_zone, exog_cols
#             )
#             used_train = df_train_proc
#             used_test = df_test_proc
#         elif main_model == 'cat':
#             # Pas d'encodage, CatBoost gère les colonnes catégorielles
#             used_train = df_train_zone
#             used_test = df_test_zone
#             # Récupérer la liste des colonnes catégorielles si non fournie
#             if categorical_cols is None:
#                 _, _, _, _, categorical_cols = preprocess_exog_and_target2(df_train_zone, df_test_zone, exog_cols)
#             cat_features = categorical_cols
#             # S'assurer que le modèle CatBoost connaît les colonnes catégorielles
#             #first_model.set_params(cat_features=cat_features)
#         else:
#             used_train = df_train_zone
#             used_test = df_test_zone

#         # 5. Update du modèle avec l'historique encodé ou non
#         forecast.update(used_train)
#         forecast.save(save_dir)

#         # 6. Prédiction sur le test (avec encodage si besoin)
#         y_pred_scaled_final = forecast.predict(h=h, X_df=used_test, level=[90, 95])
#         y_true = used_test['y'].values[:h]
#         pred_cols_final = [
#             col for col in y_pred_scaled_final.columns
#             if col not in ['unique_id', 'ds'] +
#             [f'lo-{lvl}' for lvl in [90, 95]] +
#             [f'hi-{lvl}' for lvl in [90, 95]]
#         ]
#         if not pred_cols_final:
#             raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled_final.columns}")

#         y_pred_final = y_pred_scaled_final[pred_cols_final[0]].values[:h]
#         #y_pred_final = np.clip(y_pred_final, 0, None)
        

#         # 7. Sauvegardes des résultats
#         preds_df = used_test.reset_index(drop=True).copy()
#         preds_df['y_true'] = y_true
#         preds_df['y_pred'] = y_pred_final

#         model_class_name = first_model.__class__.__name__
#         for lvl in [90, 95]:
#             lo_col = f'{model_class_name}-lo-{lvl}'
#             hi_col = f'{model_class_name}-hi-{lvl}'
#             if lo_col in y_pred_scaled_final.columns and hi_col in y_pred_scaled_final.columns:
#                 preds_df[f'ic_lo_{lvl}'] = y_pred_scaled_final[lo_col].values[:h]
#                 preds_df[f'ic_hi_{lvl}'] = y_pred_scaled_final[hi_col].values[:h]

#         colonnes = [
#             "unique_id", "ds", "Annee", "Semaine", "y", "y_pred",
#             "ic_lo_90", "ic_hi_90", "ic_lo_95", "ic_hi_95"
#         ]
#         colonnes = [c for c in colonnes if c in preds_df.columns]

#         rmse_final = mean_squared_error(y_true, y_pred_final, squared=False)
#         mae_final = mean_absolute_error(y_true, y_pred_final)
#         best_model_metrics = {'RMSE': rmse_final, 'MAE': mae_final}

#         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#         data_save_dir = os.path.join(project_root, "data", "models_data", f"models_{target_col_name}_{zone}")
#         os.makedirs(data_save_dir, exist_ok=True)

#         preds_df[colonnes].to_csv(os.path.join(data_save_dir, "predictions.csv"), index=False)
#         used_train.to_csv(os.path.join(data_save_dir, "train_data.csv"), index=False)
#         pd.DataFrame([best_model_metrics]).to_csv(os.path.join(data_save_dir, "best_metrics.csv"), index=False)

#         if verbose:
#             logging.info(f"Test et sauvegardes terminés pour {zone} - {target_col_name}")

#         best_params_path = os.path.join(save_dir, "best_params.json")
#         best_config = {}
#         if os.path.exists(best_params_path):
#             with open(best_params_path) as f:
#                 best_config = json.load(f)

#         return {
#             'zone': zone,
#             'best_params': best_config,
#             'best_metrics': best_model_metrics,
#             'predictions': preds_df,
#             'mlforecast': forecast,
#             'save_dir': save_dir
#         }
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         logging.error(f"Erreur dans test_model_zone_ML pour la zone {zone} : {repr(e)}")
#         return None



# def test_models_for_all_zones(
#     df_train, df_test, target_col_name , target_col='y',  h=12, 
#     base_path="./models", verbose=True
# ):
#     """
#     CHARGE ET TESTE CHAQUE MODÈLE MLFORECAST PAR ZONE (sauvegarde preds/metrics)
#     """
#     import logging
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#     zones_uniques = df_train['unique_id'].unique()
#     resultats_test = {}
#     for i, zone in enumerate(zones_uniques):
#         try:
#             df_train_zone = df_train[df_train['unique_id'] == zone].copy()
#             df_test_zone = df_test[df_test['unique_id'] == zone].copy()

#             result_test = test_model_zone_ML2(
#                 df_train_zone,
#                 df_test_zone,
#                 zone,
#                 target_col,
#                 target_col_name,
#                 h,
#                 base_path,
#                 verbose
#             )
#             resultats_test[zone] = result_test
#             if verbose:
#                 logging.info(f"Succès test {zone} | RMSE={result_test['best_metrics']['RMSE']:.3f} | MAE={result_test['best_metrics']['MAE']:.3f}")
#         except Exception as e:
#             if verbose:
#                 logging.error(f"ÉCHEC test {zone}: {e}")
#             continue
#     return resultats_test



# def predict_futur_for_all_zones(
#     df_train, df_test, target_col_name, base_path="./models", h=4, verbose=True, exog_cols=None
# ):
#     """
#     Prédit pour la période future pour chaque zone.
#     Utilise le modèle .pkl optimisé, prépare le train/test comme dans la phase test,
#     ajoute les intervalles de confiance, sauvegarde les prédictions futures et le train utilisé.
#     """


#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#     RANDOM_STATE = 42
#     np.random.seed(RANDOM_STATE)
#     random.seed(RANDOM_STATE)

#     zones_uniques = df_train['unique_id'].unique()
#     all_preds = {}

#     for zone in zones_uniques:
#         if verbose:
#             print(f"\n=== Zone {zone} ===")
            
#         save_dir = f"{base_path}/models_{target_col_name}_{zone}/"

#         # 1. Identifier le modèle gagnant via le .pkl
#         best_model_name = None
#         model_pkl_path = None
#         for model_name in ["xgb", "lgbm", "cat"]:
#             model_file = os.path.join(save_dir, f"{model_name}_mlforecast.pkl")
#             if os.path.exists(model_file):
#                 best_model_name = model_name
#                 model_pkl_path = model_file
#                 break
#         if best_model_name is None:
#             logging.error(f"Aucun modèle trouvé pour {zone} dans {save_dir}")
#             continue

#         # 2. Charger les meilleurs hyperparams
#         with open(os.path.join(save_dir, "best_params.json")) as f:
#             best_params = json.load(f)

#         # 3. Préparation des données (comme dans test)
#         df_train_zone = df_train[df_train['unique_id'] == zone].copy()
#         df_test_zone = df_test[df_test['unique_id'] == zone].copy()

#         # Exogènes
#         if exog_cols is None:
#             exog_cols_zone = [col for col in df_train_zone.columns if col not in ['unique_id', 'ds', 'y']]
#         else:
#             exog_cols_zone = exog_cols

#         df_train_proc, df_test_proc, final_exog_cols, ohe_dict, categorical_cols = preprocess_exog_and_target2(
#             df_train_zone, df_test_zone, exog_cols_zone
#         )

#         # 4. Construction et fit du modèle
#         MODELS = {
#             'lgbm': lambda **kwargs: LGBMRegressor(**kwargs),
#             'xgb': lambda **kwargs: XGBRegressor(**kwargs),
#             'cat': lambda **kwargs: CatBoostRegressor(**kwargs)
#         }
#         lags = list(range(1, 25))
#         lag_transforms = {lag: [RollingMean(window_size=lag), ExpandingMean(), RollingQuantile(window_size=lag, p=0.25), RollingQuantile(window_size=lag, p=0.75)] for lag in lags}
#         target_transforms = [
#             AutoDifferences(max_diffs=2),
#             LocalStandardScaler()
#         ]

#         if best_model_name == 'cat':
#             best_params['verbose'] = 0
#             used_train = df_train_zone.copy()
#             used_test = df_test_zone.copy()
#             used_exog_cols = exog_cols_zone
#             cat_features = categorical_cols
#             model = MODELS[best_model_name](**best_params, cat_features=cat_features)
#         elif best_model_name == 'lgbm':
#             best_params['verbosity'] = -1
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()
#             used_exog_cols = final_exog_cols
#             model = MODELS[best_model_name](**best_params)
#         elif best_model_name == 'xgb':
#             best_params['verbosity'] = 0
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()
#             used_exog_cols = final_exog_cols
#             model = MODELS[best_model_name](**best_params)
#         else:
#             used_train = df_train_proc.copy()
#             used_test = df_test_proc.copy()
#             used_exog_cols = final_exog_cols
#             model = MODELS[best_model_name](**best_params)

#         forecast = MLForecast(
#             models=[model],
#             freq='W-MON',
#             lags=lags,
#             lag_transforms=lag_transforms,
#             target_transforms=target_transforms,
#             date_features=['month', 'quarter']
#         )
#         forecast.fit(
#             used_train,
#             id_col='unique_id',
#             time_col='ds',
#             target_col='y',
#             static_features=[],
#             prediction_intervals=PredictionIntervals(n_windows=4, h=h)
#         )

#         # Prédiction pour la période future (h)
#         df_test_h = used_test.groupby('unique_id').head(h)
#         y_pred_scaled_final = forecast.predict(h=h, X_df=df_test_h, level=[90, 95])

#         # Sélection des colonnes de prédiction
#         pred_cols_final = [col for col in y_pred_scaled_final.columns if col not in ['unique_id', 'ds'] + [f'lo-{lvl}' for lvl in [90, 95]] + [f'hi-{lvl}' for lvl in [90, 95]]]
#         if not pred_cols_final:
#             logging.error(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled_final.columns}")
#             continue

#         y_pred_final = y_pred_scaled_final[pred_cols_final[0]].values[:h]

#         df_preds = df_test_h.reset_index(drop=True).copy()
#         df_preds['y_pred'] = y_pred_final
#         df_preds['type_prediction'] = 'futur'

#         # Ajoute les intervalles pour chaque niveau
#         for lvl in [90, 95]:
#             lo_col = f'{model.__class__.__name__}-lo-{lvl}'
#             hi_col = f'{model.__class__.__name__}-hi-{lvl}'
#             if lo_col in y_pred_scaled_final.columns and hi_col in y_pred_scaled_final.columns:
#                 df_preds[f'ic_lo_{lvl}'] = y_pred_scaled_final[lo_col].values[:h]
#                 df_preds[f'ic_hi_{lvl}'] = y_pred_scaled_final[hi_col].values[:h]

#         # Sauvegarde des prédictions futures et du train utilisé
#         colonnes = ["unique_id", "ds", "Annee", "Semaine", "y", "y_pred", "ic_lo_90", "ic_hi_90", "ic_lo_95", "ic_hi_95", "type_prediction"]
#         colonnes = [c for c in colonnes if c in df_preds.columns]

#         data_save_dir = os.path.join("data", "models_data", f"models_{target_col_name}_{zone}")
#         os.makedirs(data_save_dir, exist_ok=True)
#         futur_preds_path = os.path.join(data_save_dir, "futur_predictions.csv")
#         df_preds[colonnes].to_csv(futur_preds_path, index=False)
#         df_train_zone.reset_index(drop=True).to_csv(os.path.join(data_save_dir, "train_futur.csv"), index=False)
#         all_preds[zone] = df_preds

#         if verbose:
#             logging.info(f" Prédictions futures faites pour {zone} et sauvegardées dans {futur_preds_path}")

#     return all_preds


# def predict_futur_for_all_zones_ML(
#     df_train, df_test, target_col_name, base_path="./models", h=4, verbose=True, exog_cols=None,
#     ohe_dict=None, categorical_cols=None
# ):
#     """
#     Prédit pour la période future (h semaines) pour chaque zone en utilisant le modèle MLForecast
#     déjà optimisé et sauvegardé. Le modèle est chargé, mis à jour (update) avec le train,
#     puis la prédiction est faite sur le test (semaines futures).
#     Les prédictions et le train utilisé sont sauvegardés au même endroit que la fonction test.
#     """
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
#     RANDOM_STATE = 42
#     np.random.seed(RANDOM_STATE)
#     random.seed(RANDOM_STATE)

#     zones_uniques = df_train['unique_id'].unique()
#     all_preds = {}

#     for zone in zones_uniques:
#         if verbose:
#             print(f"\n=== Zone {zone} ===")
#         save_dir = f"{base_path}/models_{target_col_name}_{zone}/"
#         if not os.path.exists(save_dir):
#             logging.error(f"Dossier {save_dir} introuvable, lance d'abord optimize_model_zone pour la zone {zone}")
#             continue

#         try:
#             # 1. Charger le modèle MLForecast sauvegardé
#             forecast = MLForecast.load(save_dir)
#             # 2. Identifier le type de modèle (xgb/lgbm/cat)
#             first_model = list(forecast.models.values())[0]
#             model_name = first_model.__class__.__name__.lower()
#             if 'xgb' in model_name:
#                 main_model = 'xgb'
#             elif 'lgbm' in model_name:
#                 main_model = 'lgbm'
#             elif 'cat' in model_name:
#                 main_model = 'cat'
#             else:
#                 main_model = 'other'

#             # 3. Séparer train/test de la zone
#             df_train_zone = df_train[df_train['unique_id'] == zone].copy()
#             df_test_zone = df_test[df_test['unique_id'] == zone].copy()
#             # Exogènes
#             if exog_cols is None:
#                 exog_cols_zone = [col for col in df_train_zone.columns if col not in ['unique_id', 'ds', 'y']]
#             else:
#                 exog_cols_zone = exog_cols

#             # 4. Encodage des exogènes
#             if main_model in ['xgb', 'lgbm']:
#                 df_train_proc, df_test_proc, final_exog_cols, ohe_dict_out, categorical_cols_out = preprocess_exog_and_target2(
#                     df_train_zone, df_test_zone, exog_cols_zone
#                 )
#                 used_train = df_train_proc
#                 used_test = df_test_proc
#             elif main_model == 'cat':
#                 used_train = df_train_zone
#                 used_test = df_test_zone
#                 # Récupérer la liste des colonnes catégorielles si besoin
#                 if categorical_cols is None:
#                     _, _, _, _, categorical_cols = preprocess_exog_and_target2(df_train_zone, df_test_zone, exog_cols_zone)
#                 cat_features = categorical_cols
#             else:
#                 used_train = df_train_zone
#                 used_test = df_test_zone

#             # 5. Update du modèle uniquement avec le train (historique)
#             forecast.update(used_train)
#             forecast.save(save_dir)

#             # 6. Prédiction sur le test (futur)
#             y_pred_scaled_final = forecast.predict(h=h, X_df=used_test, level=[90, 95])

#             pred_cols_final = [
#                 col for col in y_pred_scaled_final.columns
#                 if col not in ['unique_id', 'ds'] +
#                 [f'lo-{lvl}' for lvl in [90, 95]] +
#                 [f'hi-{lvl}' for lvl in [90, 95]]
#             ]
#             if not pred_cols_final:
#                 logging.error(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled_final.columns} pour {zone}")
#                 continue

#             y_pred_final = y_pred_scaled_final[pred_cols_final[0]].values[:h]
#             #y_pred_final = np.clip(y_pred_final, 0, None)

#             df_preds = used_test.reset_index(drop=True).copy()
#             df_preds['y_pred'] = y_pred_final
#             df_preds['type_prediction'] = 'futur'

#             # Ajout des intervalles de confiance
#             model_class_name = first_model.__class__.__name__
#             for lvl in [90, 95]:
#                 lo_col = f'{model_class_name}-lo-{lvl}'
#                 hi_col = f'{model_class_name}-hi-{lvl}'
#                 if lo_col in y_pred_scaled_final.columns and hi_col in y_pred_scaled_final.columns:
#                     df_preds[f'ic_lo_{lvl}'] = y_pred_scaled_final[lo_col].values[:h]
#                     df_preds[f'ic_hi_{lvl}'] = y_pred_scaled_final[hi_col].values[:h]

#             colonnes = [
#                 "unique_id", "ds", "Annee", "Semaine", "y", "y_pred",
#                 "ic_lo_90", "ic_hi_90", "ic_lo_95", "ic_hi_95", "type_prediction"
#             ]
#             colonnes = [c for c in colonnes if c in df_preds.columns]

#             # 7. Sauvegarde dans le même dossier que la fonction test
#             project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#             data_save_dir = os.path.join(project_root, "data", "models_data", f"models_{target_col_name}_{zone}")
#             os.makedirs(data_save_dir, exist_ok=True)
#             preds_path = os.path.join(data_save_dir, "predictions_futur.csv")
#             df_preds[colonnes].to_csv(preds_path, index=False)
#             used_train.to_csv(os.path.join(data_save_dir, "train_futur.csv"), index=False)
#             all_preds[zone] = df_preds

#             if verbose:
#                 logging.info(f"Prédictions futures faites pour {zone} et sauvegardées dans {preds_path}")

#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             logging.error(f"Erreur dans predict_futur_for_all_zones_ML pour la zone {zone} : {repr(e)}")
#             continue

#     return all_preds



















# test pour afficher les predictions sur le train 

# def test_model_zone_train_cross(
#     df_train, df_test, zone, target_col, target_col_name, h, base_path, verbose,
#     exog_cols=None
# ):
#     """
#     Charge best_params.json, ré-entraine le modèle optimisé avec les bons hyperparams,
#     prédit sur le test, sauvegarde (en écrasant) le modèle .pkl, predictions.csv, train_data.csv, best_metrics.csv.
#     """
    

#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#     # Chemin du dossier zone
#     save_dir = f"{base_path}/models_{target_col_name}_{zone}/"
#     if not os.path.exists(save_dir):
#         raise ValueError(f"Dossier {save_dir} introuvable, lance d'abord optimize_model_zone")

#     # 1. Charger le nom du meilleur modèle (par le nom du fichier .pkl existant)
#     best_model_name = None
#     for model_name in ["xgb", "lgbm", "cat"]:
#         model_file = os.path.join(save_dir, f"{model_name}_mlforecast.pkl")
#         if os.path.exists(model_file):
#             best_model_name = model_name
#             model_pkl_path = model_file
#             break
#     if best_model_name is None:
#         raise FileNotFoundError(f"Aucun modèle trouvé pour {zone} dans {save_dir}")

#     # 2. Charger les meilleurs hyperparams
#     with open(os.path.join(save_dir, "best_params.json")) as f:
#         best_config = json.load(f)

#     # 3. Préparation des données
#     df_train = df_train[df_train['unique_id'] == zone].copy()
#     df_test = df_test[df_test['unique_id'] == zone].copy()

#     # 3.1 Préparation des exogènes
#     if exog_cols is None:
#         exog_cols = [col for col in df_train.columns if col not in ['unique_id', 'ds', 'y']]

#     df_train_proc, df_test_proc, final_exog_cols, ohe_dict, categorical_cols = preprocess_exog_and_target2(
#         df_train, df_test, exog_cols
#     )

#     # 4. Reconstruction et fit du modèle avec les bons hyperparams
#     # Définir ici tes objets MODELS, lags, lag_transforms, target_transforms comme dans optimize_model_zone
#     MODELS = {
#         'lgbm': lambda **kwargs: LGBMRegressor(**kwargs),
#         'xgb': lambda **kwargs: XGBRegressor(**kwargs),
#         'cat': lambda **kwargs: CatBoostRegressor(**kwargs)
#     }
#     lags = list(range(1, 25))
#     lag_transforms = {lag: [RollingMean(window_size=lag), ExpandingMean(), RollingQuantile(window_size=lag, p=0.25), RollingQuantile(window_size=lag, p=0.75)] for lag in lags}
#     target_transforms = [
#         AutoDifferences(max_diffs=2),
#         LocalStandardScaler()
#     ]

#     if best_model_name == 'cat':
#         best_config['verbose'] = 0
#         used_train = df_train.copy()
#         used_test = df_test.copy()
#         used_exog_cols = exog_cols
#         cat_features = categorical_cols
#         model = MODELS[best_model_name](**best_config, cat_features=cat_features)
#     elif best_model_name == 'lgbm':
#         best_config['verbosity'] = -1
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)
#     elif best_model_name == 'xgb':
#         best_config['verbosity'] = 0
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)
#     else:
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)

#     forecast = MLForecast(
#         models=[model],
#         freq='W-MON',
#         lags=lags,
#         lag_transforms=lag_transforms,
#         target_transforms=target_transforms,
#         date_features=['month', 'quarter']
#     )
#     #5.0 cross validation sur le test 
#     cv_results = forecast.cross_validation(
#         used_train,
#         h=4,             # horizon de prévision (comme le test)
#         n_windows=15,    # 15 fenêtres pour couvrir le train sans trop réduire la taille d'apprentissage
#         step_size=1,     # on avance de 4 points entre chaque fenêtre
#         refit=True,      # on refait le fit à chaque fenêtre (classique)
#         static_features=[]
#     )
#     print("Colonnes dans cv_results:", cv_results.columns)
#     pred_col_candidates = [c for c in cv_results.columns if c not in ['unique_id', 'ds', 'y', 'cutoff']]
#     if not pred_col_candidates:
#         raise ValueError(f"Aucune colonne de prédiction trouvée dans {cv_results.columns}")
#     pred_col = pred_col_candidates[0]
#     rmse_cv = mean_squared_error(cv_results['y'], cv_results[pred_col], squared=False)
#     print(f"RMSE moyen cross-validation (train): {rmse_cv:.4f}")

#     # RMSE par fenêtre de cross-validation (par cutoff)
#     cutoff_rmses = []
#     for cutoff, group in cv_results.groupby('cutoff'):
#         rmse_split = mean_squared_error(group['y'], group[pred_col], squared=False)
#         cutoff_rmses.append((cutoff, rmse_split))
#         print(f"Cutoff {cutoff}: RMSE = {rmse_split:.4f}")
#     # Optionnel : sauvegarder dans un CSV
#     cutoff_rmse_df = pd.DataFrame(cutoff_rmses, columns=['cutoff', 'rmse'])
#     cutoff_rmse_df.to_csv(os.path.join(save_dir, "cv_rmse_by_split.csv"), index=False)
#     #On sélectionne les colonnes intéressantes
#     tableau_cv = cv_results[['ds', 'cutoff', 'y', pred_col]].sort_values(['cutoff', 'ds'])

#     # Renomme pour clarté
#     tableau_cv = tableau_cv.rename(columns={pred_col: 'y_pred'})

#     # Sauvegarde en CSV 
#     tableau_cv.to_csv(os.path.join(save_dir, 'cv_predictions_by_window.csv'), index=False)
#     cv_results.to_csv(os.path.join(save_dir, "cv_results.csv"), index=False)

#     forecast.fit(
#         used_train,
#         id_col='unique_id',
#         time_col='ds',
#         target_col='y',
#         static_features=[], 
#         prediction_intervals= PredictionIntervals(n_windows=4, h=h)
#     )

    

#     # 5. Prédiction sur le test
#     y_pred_scaled_final = forecast.predict(h=h, X_df=used_test, level= [90, 95])
#     y_true = used_test['y'].values[:h]
#     pred_cols_final = [col for col in y_pred_scaled_final.columns if col not in ['unique_id', 'ds']+ [f'lo-{lvl}' for lvl in [90,95]] + [f'hi-{lvl}' for lvl in [90,95]]]
#     if not pred_cols_final:
#         raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled_final.columns}")

#     y_pred_final = y_pred_scaled_final[pred_cols_final[0]].values[:h]

#     # 6. Sauvegarde du modèle refitté (écrase l'ancien)
#     joblib.dump(forecast, model_pkl_path)
#     if verbose:
#         logging.info(f"Le modèle {best_model_name} (zone {zone}) a été ré-entraîné et sauvegardé dans {model_pkl_path} (ancien modèle écrasé)")

#     # 7. Sauvegardes des résultats
#     preds_df = df_test.reset_index(drop=True).copy()
#     preds_df['y_true'] = y_true
#     preds_df['y_pred'] = y_pred_final

#     # Ajoute les intervalles pour chaque niveau
#     for lvl in [90, 95]:
#         lo_col = f'{model.__class__.__name__}-lo-{lvl}'
#         hi_col = f'{model.__class__.__name__}-hi-{lvl}'
#         if lo_col in y_pred_scaled_final.columns and hi_col in y_pred_scaled_final.columns:
#             preds_df[f'ic_lo_{lvl}'] = y_pred_scaled_final[lo_col].values[:h]
#             preds_df[f'ic_hi_{lvl}'] = y_pred_scaled_final[hi_col].values[:h]


#     # Sélectionne uniquement les colonnes demandées (en gardant celles existantes)
#     colonnes = ["unique_id", "ds", "Annee", "Semaine","y", "y_pred", "ic_lo_90", "ic_hi_90", "ic_lo_95", "ic_hi_95"]
#     # Ne garde que celles qui existent déjà dans preds_df
#     colonnes = [c for c in colonnes if c in preds_df.columns]
#     preds_df[colonnes].to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
#     df_train.to_csv(os.path.join(save_dir, "train_data.csv"), index=False)

#     rmse_final = mean_squared_error(y_true, y_pred_final, squared=False)
#     mae_final = mean_absolute_error(y_true, y_pred_final)
#     best_model_metrics = {'RMSE': rmse_final, 'MAE': mae_final}
#     pd.DataFrame([best_model_metrics]).to_csv(os.path.join(save_dir, "best_metrics.csv"), index=False)

#     if verbose:
#         logging.info(f"Test et sauvegardes terminés pour {zone} - {target_col_name}")

#     return {
#         'zone': zone,
#         'best_model': best_model_name,
#         'best_params': best_config,
#         'best_metrics': best_model_metrics,
#         'predictions': preds_df,
#         'mlforecast': forecast,
#         'save_dir': save_dir
#     }



# def train_pred_zone_roll(
#     df_train, df_test, zone, target_col, target_col_name, h, base_path, verbose,
#     exog_cols=None
# ):
#     """
#     Charge best_params.json, ré-entraine le modèle optimisé avec les bons hyperparams,
#     prédit sur le test, sauvegarde (en écrasant) le modèle .pkl, predictions.csv, train_data.csv, best_metrics.csv.
#     Ajout: génère et sauvegarde les prédictions one-step-ahead (rolling) sur le train.
#     Ajout debug rolling.
#     """
#     import os
#     import json
#     import logging
#     import pandas as pd
#     import joblib
#     from sklearn.metrics import mean_squared_error, mean_absolute_error
#     from tqdm import tqdm

#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#     # Chemin du dossier zone
#     save_dir = f"{base_path}/models_{target_col_name}_{zone}/"
#     if not os.path.exists(save_dir):
#         raise ValueError(f"Dossier {save_dir} introuvable, lance d'abord optimize_model_zone")

#     # 1. Charger le nom du meilleur modèle (par le nom du fichier .pkl existant)
#     best_model_name = None
#     for model_name in ["xgb", "lgbm", "cat"]:
#         model_file = os.path.join(save_dir, f"{model_name}_mlforecast.pkl")
#         if os.path.exists(model_file):
#             best_model_name = model_name
#             model_pkl_path = model_file
#             break
#     if best_model_name is None:
#         raise FileNotFoundError(f"Aucun modèle trouvé pour {zone} dans {save_dir}")

#     # 2. Charger les meilleurs hyperparams
#     with open(os.path.join(save_dir, "best_params.json")) as f:
#         best_config = json.load(f)

#     # 3. Préparation des données
#     df_train = df_train[df_train['unique_id'] == zone].copy()
#     df_test = df_test[df_test['unique_id'] == zone].copy()

#     # Date à ajouter (lundi suivant la dernière semaine de 2024)
#     ds_to_add = pd.Timestamp("2024-12-30")

#     # Vérifie si cette date existe déjà pour la zone
#     if not (df_train["ds"] == ds_to_add).any():
#         # On cherche la ligne de la DERNIÈRE semaine de 2024 (2024-12-23)
#         prev_row = df_train[df_train["ds"] == pd.Timestamp("2024-12-23")]
#         if not prev_row.empty:
#             # On copie la ligne précédente
#             new_row = prev_row.copy()
#             new_row["ds"] = ds_to_add
#             # Année/semaine ISO pour la date ajoutée
#             iso = ds_to_add.isocalendar()
#             new_row["Annee"] = iso.year
#             new_row["Semaine"] = iso.week
#             # On reset l'index pour concat
#             new_row.index = [df_train.index.max() + 1]
#             # On insère cette ligne APRÈS la dernière de 2024-12-23, mais avant toute donnée 2025
#             df_train = pd.concat(
#                 [
#                     df_train[df_train["ds"] <= pd.Timestamp("2024-12-23")],
#                     new_row,
#                     df_train[df_train["ds"] > pd.Timestamp("2024-12-23")]
#                 ]
#             ).reset_index(drop=True)

#     # 3.1 Préparation des exogènes
#     if exog_cols is None:
#         exog_cols = [col for col in df_train.columns if col not in ['unique_id', 'ds', 'y']]

#     df_train_proc, df_test_proc, final_exog_cols, ohe_dict, categorical_cols = preprocess_exog_and_target2(
#         df_train, df_test, exog_cols
#     )
#     global_encoded_cols = [col for col in df_train_proc.columns if col not in ['unique_id', 'ds', 'y']]

#     # 4. Reconstruction et fit du modèle avec les bons hyperparams
#     MODELS = {
#         'lgbm': lambda **kwargs: LGBMRegressor(**kwargs),
#         'xgb': lambda **kwargs: XGBRegressor(**kwargs),
#         'cat': lambda **kwargs: CatBoostRegressor(**kwargs)
#     }
#     lags = list(range(1, 25))
#     lag_transforms = {lag: [RollingMean(window_size=lag), ExponentiallyWeightedMean(alpha=0.9), RollingQuantile(window_size=lag, p=0.25), RollingQuantile(window_size=lag, p=0.75)] for lag in lags}
#     target_transforms = [
#         AutoDifferences(max_diffs=2),
#         LocalStandardScaler()
#     ]

#     if best_model_name == 'cat':
#         best_config['verbose'] = 0
#         used_train = df_train.copy()
#         used_test = df_test.copy()
#         used_exog_cols = exog_cols
#         cat_features = categorical_cols
#         model = MODELS[best_model_name](**best_config, cat_features=cat_features)
#     elif best_model_name == 'lgbm':
#         best_config['verbosity'] = -1
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)
#     elif best_model_name == 'xgb':
#         best_config['verbosity'] = 0
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)
#     else:
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)

#     forecast = MLForecast(
#         models=[model],
#         freq='W-MON',
#         lags=lags,
#         lag_transforms=lag_transforms,
#         target_transforms=target_transforms,
#         date_features=['month', 'quarter']
#     )
#     # forecast.fit(
#     #     used_train,
#     #     id_col='unique_id',
#     #     time_col='ds',
#     #     target_col='y',
#     #     static_features=[], 
#     #     prediction_intervals= PredictionIntervals(n_windows=4, h=h)
#     # )

#     # =============================
#     # 5.0 Prédiction rolling sur le train (one-step-ahead) avec debug
#     # =============================
#     min_lag = max(lags)
#     y_true_rolling = []
#     y_pred_rolling = []
#     for i in tqdm(range(min_lag, len(df_train)), desc=f"Rolling one-step train prediction: {zone}"):
#         sub_train = df_train.iloc[:i].copy()
#         sub_test = df_train.iloc[[i]].copy()

#         # Debug: affiche la taille brute avant preprocess
#         #print(f"[DEBUG {zone} | {i}] sub_train size: {sub_train.shape}, sub_test size: {sub_test.shape}")

#         # Préprocessing identique à l'entraînement
#         if best_model_name == 'cat':
#             sub_train_proc = sub_train.copy()
#             sub_test_proc = sub_test.copy()
#             sub_exog_cols = exog_cols
#             sub_cat_features = categorical_cols
#             sub_model = MODELS[best_model_name](**best_config, cat_features=sub_cat_features)
#         else:
#             # On encode AVEC les OHE globaux déjà fités
#             sub_train_proc = sub_train.copy()
#             sub_test_proc = sub_test.copy()
#             for col in categorical_cols:
#                 ohe = ohe_dict[col]
#                 encoded_train = ohe.transform(sub_train_proc[[col]].astype(str))
#                 encoded_test = ohe.transform(sub_test_proc[[col]].astype(str))
#                 encoded_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
#                 encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_cols, index=sub_train_proc.index)
#                 encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_cols, index=sub_test_proc.index)
#                 sub_train_proc = pd.concat([sub_train_proc.drop(columns=[col]), encoded_train_df], axis=1)
#                 sub_test_proc = pd.concat([sub_test_proc.drop(columns=[col]), encoded_test_df], axis=1)

#             # Ajoute les colonnes OHE manquantes à 0, et réordonne comme à l'entraînement
#             for col in global_encoded_cols:
#                 if col not in sub_train_proc.columns:
#                     sub_train_proc[col] = 0
#                 if col not in sub_test_proc.columns:
#                     sub_test_proc[col] = 0
#             sub_train_proc = sub_train_proc[['unique_id', 'ds', 'y'] + global_encoded_cols]
#             sub_test_proc = sub_test_proc[['unique_id', 'ds', 'y'] + global_encoded_cols]

#             sub_exog_cols = final_exog_cols
#             sub_model = MODELS[best_model_name](**best_config)

#         # Debug: affiche la taille après preprocess et les colonnes
#         # print(f"[DEBUG {zone} | {i}] sub_train_proc size: {sub_train_proc.shape}, sub_test_proc size: {sub_test_proc.shape}")
#         # print(f"[DEBUG {zone} | {i}] sub_train_proc cols: {sub_train_proc.columns.tolist()}, sub_test_proc cols: {sub_test_proc.columns.tolist()}")

#         # Debug: vérifie DataFrames non vides et bonne forme
#         try:
#             assert not sub_train_proc.empty, f"Train proc vide à l'itération {i} (zone {zone})"
#             assert not sub_test_proc.empty, f"Test proc vide à l'itération {i} (zone {zone})"
#             assert sub_train_proc.shape[1] > 0, f"Train proc pas assez de colonnes à l'itération {i} (zone {zone})"
#             assert sub_test_proc.shape[1] > 0, f"Test proc pas assez de colonnes à l'itération {i} (zone {zone})"
#             if 'y' not in sub_train_proc.columns or sub_train_proc['y'].dropna().empty:
#                 raise AssertionError(f"Labels variable is empty à l'itération {i} (zone {zone})")
#         except AssertionError as e:
#             #print(f"[DEBUG {zone} | {i}] ERREUR DEBUG : {e}")
#             print(f"[DEBUG][{zone}][{best_model_name}] Rolling step {i} skipped: {e}")
#             print("sub_train_proc.columns:", list(sub_train_proc.columns))
#             print("sub_test_proc.columns:", list(sub_test_proc.columns))
#             print("sub_train_proc NaN:\n", sub_train_proc.isna().sum())
#             print("sub_test_proc NaN:\n", sub_test_proc.isna().sum())
#             y_pred_rolling.append(None)
#             y_true_rolling.append(df_train.iloc[i]['y'])
#             continue  

#         # Fit/forecast
#         try:
#             sub_forecast = MLForecast(
#                 models=[sub_model],
#                 freq='W-MON',
#                 lags=lags,
#                 lag_transforms=lag_transforms,
#                 target_transforms=target_transforms,
#                 date_features=['month', 'quarter']
#             )
#             print(f"[DEBUG][{zone}][{best_model_name}] shape sub_train_proc: {sub_train_proc.shape}, columns: {sub_train_proc.columns.tolist()}")
#             print(f"[DEBUG][{zone}][{best_model_name}] shape sub_test_proc: {sub_test_proc.shape}, columns: {sub_test_proc.columns.tolist()}")
#             print(f"[DEBUG][{zone}][{best_model_name}] sub_exog_cols: {sub_exog_cols}")
#             print(f"[DEBUG][{zone}][{best_model_name}] sub_train_proc[sub_exog_cols].shape: {sub_train_proc[sub_exog_cols].shape if sub_exog_cols else 'None'}")
#             print(f"[DEBUG][{zone}][{best_model_name}] sub_test_proc[sub_exog_cols].shape: {sub_test_proc[sub_exog_cols].shape if sub_exog_cols else 'None'}")
#             nb_valid = sub_train_proc[sub_exog_cols].dropna().shape[0]
#             print(f"[DEBUG][{zone}][{best_model_name}] Rolling step {i}: {nb_valid} lignes non-NaN pour les features lags.")

#             valid_train = sub_train_proc.dropna(subset=sub_exog_cols).copy()
#             if valid_train.shape[0] < 2:
#                 print(f"[DEBUG][{zone}][{best_model_name}] Rolling step {i}: pas assez de lignes non-NaN pour fit, on skip.")
#                 y_pred_rolling.append(None)
#                 y_true_rolling.append(df_train.iloc[i]['y'])
#                 continue
#             if valid_train['y'].nunique() < 2:
#                 print(f"[DEBUG][{zone}][{best_model_name}] Rolling step {i}: target constante ({valid_train['y'].iloc[0]}) ! Skip.")
#                 y_pred_rolling.append(None)
#                 y_true_rolling.append(df_train.iloc[i]['y'])
#                 continue
#             zero_var_cols = [col for col in sub_exog_cols if valid_train[col].nunique() < 2]
#             if len(zero_var_cols) == len(sub_exog_cols):
#                 print(f"[DEBUG][{zone}][{best_model_name}] Rolling step {i}: toutes les features lags sont constantes ! Skip.")
#                 y_pred_rolling.append(None)
#                 y_true_rolling.append(df_train.iloc[i]['y'])
#                 continue
                        
#             sub_forecast.fit(
#                 sub_train_proc if best_model_name != 'cat' else sub_train_proc,
#                 id_col='unique_id',
#                 time_col='ds',
#                 target_col='y',
#                 static_features=[]
#             )
#             pred = sub_forecast.predict(h=1, X_df=sub_test_proc if best_model_name != 'cat' else sub_test_proc)
#             pred_col = [col for col in pred.columns if col not in ['unique_id', 'ds']+ [f'lo-{lvl}' for lvl in [90,95]] + [f'hi-{lvl}' for lvl in [90,95]]]
#             if not pred_col:
#                 #print(f"[DEBUG {zone} | {i}] ERREUR DEBUG : Pas de colonne prédiction dans {pred.columns}")
#                 y_pred_rolling.append(None)
#             else:
#                 y_pred_rolling.append(pred[pred_col[0]].values[0])
#             y_true_rolling.append(df_train.iloc[i]['y'])
#         except Exception as e:
#             #print(f"[DEBUG {zone} | {i}] ERREUR DEBUG : {e}")
#             print(f"[DEBUG][{zone}][{best_model_name}] Rolling step {i} fit/predict ERROR: {e}")
#             print("sub_train_proc.columns:", list(sub_train_proc.columns))
#             print("sub_test_proc.columns:", list(sub_test_proc.columns))
#             print("sub_train_proc NaN:\n", sub_train_proc.isna().sum())
#             print("sub_test_proc NaN:\n", sub_test_proc.isna().sum())
#             y_pred_rolling.append(None)
#             y_true_rolling.append(df_train.iloc[i]['y'])
#             continue

#     df_train = df_train.reset_index(drop=True).copy()
#     df_train['y_pred_train_rolling'] = [None]*min_lag + y_pred_rolling

#     df_rolling_preds = df_train[['unique_id', 'ds', 'y', 'y_pred_train_rolling']].copy()
#     df_rolling_preds.to_csv(os.path.join(save_dir, "rolling_predictions.csv"), index=False)
#     valid_idx = [i for i, v in enumerate(y_pred_rolling) if v is not None]
#     y_true_valid = [y_true_rolling[i] for i in valid_idx]
#     y_pred_valid = [y_pred_rolling[i] for i in valid_idx]
#     if len(y_true_valid) > 1:
#         rmse_rolling = mean_squared_error(y_true_valid, y_pred_valid, squared=False)
#         mae_rolling = mean_absolute_error(y_true_valid, y_pred_valid)
#         print(f"[ROLLING SUMMARY] {zone} - RMSE: {rmse_rolling:.3f}, MAE: {mae_rolling:.3f} sur {len(y_true_valid)} steps")

#     # =============================

#     # 5. Prédiction sur le test
#     # y_pred_scaled_final = forecast.predict(h=h, X_df=used_test, level= [90, 95])
#     # y_true = used_test['y'].values[:h]
#     # pred_cols_final = [col for col in y_pred_scaled_final.columns if col not in ['unique_id', 'ds']+ [f'lo-{lvl}' for lvl in [90,95]] + [f'hi-{lvl}' for lvl in [90,95]]]
#     # if not pred_cols_final:
#     #     raise ValueError(f"Aucune colonne de prédiction trouvée dans {y_pred_scaled_final.columns}")

#     # y_pred_final = y_pred_scaled_final[pred_cols_final[0]].values[:h]

#     # # 6. Sauvegarde du modèle refitté (écrase l'ancien)
#     # joblib.dump(forecast, model_pkl_path)
#     # if verbose:
#     #     logging.info(f"Le modèle {best_model_name} (zone {zone}) a été ré-entraîné et sauvegardé dans {model_pkl_path} (ancien modèle écrasé)")

#     # # 7. Sauvegardes des résultats
#     # preds_df = df_test.reset_index(drop=True).copy()
#     # preds_df['y_true'] = y_true
#     # preds_df['y_pred'] = y_pred_final

#     # # Ajoute les intervalles pour chaque niveau
#     # for lvl in [90, 95]:
#     #     lo_col = f'{model.__class__.__name__}-lo-{lvl}'
#     #     hi_col = f'{model.__class__.__name__}-hi-{lvl}'
#     #     if lo_col in y_pred_scaled_final.columns and hi_col in y_pred_scaled_final.columns:
#     #         preds_df[f'ic_lo_{lvl}'] = y_pred_scaled_final[lo_col].values[:h]
#     #         preds_df[f'ic_hi_{lvl}'] = y_pred_scaled_final[hi_col].values[:h]

#     # # Sélectionne uniquement les colonnes demandées (en gardant celles existantes)
#     # colonnes = ["unique_id", "ds", "Annee", "Semaine","y", "y_pred", "ic_lo_90", "ic_hi_90", "ic_lo_95", "ic_hi_95"]
#     # colonnes = [c for c in colonnes if c in preds_df.columns]
#     # preds_df[colonnes].to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
#     # df_train.to_csv(os.path.join(save_dir, "train_prediction.csv"), index=False)

#     # rmse_final = mean_squared_error(y_true, y_pred_final, squared=False)
#     # mae_final = mean_absolute_error(y_true, y_pred_final)
#     # best_model_metrics = {'RMSE': rmse_final, 'MAE': mae_final}
#     # pd.DataFrame([best_model_metrics]).to_csv(os.path.join(save_dir, "best_metrics.csv"), index=False)

#     if verbose:
#         logging.info(f"Test et sauvegardes terminés pour {zone} - {target_col_name}")

#     return {
#         'zone': zone,
#         'best_model': best_model_name,
#         'best_params': best_config,
#         # 'best_metrics': best_model_metrics,
#         # 'predictions': preds_df,
#         'rmse_rolling': rmse_rolling if len(y_true_valid) > 1 else None,
#         'mae_rolling': mae_rolling if len(y_true_valid) > 1 else None,
#         'mlforecast': forecast,
#         'save_dir': save_dir
#     }

# def train_models_for_all_zones(
#     df_train, df_test, target_col_name , target_col='y',  h=12, 
#     base_path="./models", verbose=True
# ):
#     """
#     CHARGE ET TESTE CHAQUE MODÈLE MLFORECAST PAR ZONE (sauvegarde preds/metrics)
#     """
#     import logging
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#     zones_uniques = df_train['unique_id'].unique()
#     resultats_test = {}
#     for i, zone in enumerate(zones_uniques):
#         try:
#             df_train_zone = df_train[df_train['unique_id'] == zone].copy()
#             df_test_zone = df_test[df_test['unique_id'] == zone].copy()

#             result_test = train_pred_zone_roll(
#                 df_train_zone,
#                 df_test_zone,
#                 zone,
#                 target_col,
#                 target_col_name,
#                 h,
#                 base_path,
#                 verbose
#             )
#             resultats_test[zone] = result_test
#             if verbose:
#                 logging.info(f"Succès test {zone} | RMSE={result_test['rmse_rolling']:.3f} | MAE={result_test['mae_rolling']:.3f}")
#         except Exception as e:
#             if verbose:
#                 logging.error(f"ÉCHEC test {zone}: {e}")
#             continue
#     return resultats_test


### Test model zone avec la solution de split le train 

# def train_pred_zone(
#     df_train, df_test, zone, target_col, target_col_name, base_path, verbose,
#     exog_cols=None
# ):
#     """
#     Charge best_params.json, ré-entraine le modèle optimisé avec les bons hyperparams,
#     prédit sur le test, sauvegarde (en écrasant) le modèle .pkl, predictions.csv, train_data.csv, best_metrics.csv.
#     """
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#     # Chemin du dossier zone
#     save_dir = f"{base_path}/models_{target_col_name}_{zone}/"
#     if not os.path.exists(save_dir):
#         raise ValueError(f"Dossier {save_dir} introuvable, lance d'abord optimize_model_zone")

#     # 1. Charger le nom du meilleur modèle (par le nom du fichier .pkl existant)
#     best_model_name = None
#     for model_name in ["xgb", "lgbm", "cat"]:
#         model_file = os.path.join(save_dir, f"{model_name}_mlforecast.pkl")
#         if os.path.exists(model_file):
#             best_model_name = model_name
#             model_pkl_path = model_file
#             break
#     if best_model_name is None:
#         raise FileNotFoundError(f"Aucun modèle trouvé pour {zone} dans {save_dir}")

#     # 2. Charger les meilleurs hyperparams
#     with open(os.path.join(save_dir, "best_params.json")) as f:
#         best_config = json.load(f)

#     # 3. Préparation des données
#     df_train = df_train[df_train['unique_id'] == zone].copy()
#     df_test = df_test[df_test['unique_id'] == zone].copy()

#     # Date à ajouter (lundi suivant la dernière semaine de 2024)
#     ds_to_add = pd.Timestamp("2024-12-30")

#     # Vérifie si cette date existe déjà pour la zone
#     if not (df_train["ds"] == ds_to_add).any():
#         # On cherche la ligne de la DERNIÈRE semaine de 2024 (2024-12-23)
#         prev_row = df_train[df_train["ds"] == pd.Timestamp("2024-12-23")]
#         if not prev_row.empty:
#             # On copie la ligne précédente
#             new_row = prev_row.copy()
#             new_row["ds"] = ds_to_add
#             # Année/semaine ISO pour la date ajoutée
#             iso = ds_to_add.isocalendar()
#             new_row["Annee"] = iso.year
#             new_row["Semaine"] = iso.week
#             # On reset l'index pour concat
#             new_row.index = [df_train.index.max() + 1]
#             # On insère cette ligne APRÈS la dernière de 2024-12-23, mais avant toute donnée 2025
#             df_train = pd.concat(
#                 [
#                     df_train[df_train["ds"] <= pd.Timestamp("2024-12-23")],
#                     new_row,
#                     df_train[df_train["ds"] > pd.Timestamp("2024-12-23")]
#                 ]
#             ).reset_index(drop=True)

#     # 3.1 Préparation des exogènes
#     if exog_cols is None:
#         exog_cols = [col for col in df_train.columns if col not in ['unique_id', 'ds', 'y']]

#     df_train_proc, df_test_proc, final_exog_cols, ohe_dict, categorical_cols = preprocess_exog_and_target2(
#         df_train, df_test, exog_cols
#     )

#     # 4. Reconstruction et fit du modèle avec les bons hyperparams
#     # Définir ici tes objets MODELS, lags, lag_transforms, target_transforms comme dans optimize_model_zone
#     MODELS = {
#         'lgbm': lambda **kwargs: LGBMRegressor(**kwargs),
#         'xgb': lambda **kwargs: XGBRegressor(**kwargs),
#         'cat': lambda **kwargs: CatBoostRegressor(**kwargs)
#     }
#     lags = list(range(1, 25))
#     lag_transforms = {lag: [RollingMean(window_size=lag), ExponentiallyWeightedMean(alpha=0.9), RollingQuantile(window_size=lag, p=0.25), RollingQuantile(window_size=lag, p=0.75)] for lag in lags}
#     target_transforms = [
#         AutoDifferences(max_diffs=2),
#         LocalStandardScaler()
#     ]

#     if best_model_name == 'cat':
#         best_config['verbose'] = 0
#         used_train = df_train.copy()
#         used_test = df_test.copy()
#         used_exog_cols = exog_cols
#         cat_features = categorical_cols
#         model = MODELS[best_model_name](**best_config, cat_features=cat_features)
#     elif best_model_name == 'lgbm':
#         best_config['verbosity'] = -1
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)
#     elif best_model_name == 'xgb':
#         best_config['verbosity'] = 0
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)
#     else:
#         used_train = df_train_proc.copy()
#         used_test = df_test_proc.copy()
#         used_exog_cols = final_exog_cols
#         model = MODELS[best_model_name](**best_config)

#     forecast = MLForecast(
#         models=[model],
#         freq='W-MON',
#         lags=lags,
#         lag_transforms=lag_transforms,
#         target_transforms=target_transforms,
#         date_features=['month', 'quarter']
#     )

#     # 5.0 Prediction sur le train 
#     split_idx = 51

#     used_train1 = used_train.iloc[:split_idx].copy()
#     used_train2 = used_train.iloc[split_idx:].copy()  
    
#     # On refait un fit sur la partie utilisée pour le train
#     forecast.fit(
#         used_train1,
#         id_col='unique_id',
#         time_col='ds',
#         target_col='y',
#         static_features=[]
#     )
   
#     # On prédit sur la deuxième partie du train
#     X_pred = used_train2.copy()
#     y_pred_train2 = forecast.predict(h=len(used_train2), X_df= X_pred)
#     y_true_train2 = used_train2['y'].values[:len(used_train2)]
#     pred_cols_train2 = [col for col in y_pred_train2.columns if col not in ['unique_id', 'ds']]
#     y_pred_final_train2 = y_pred_train2[pred_cols_train2[0]].values[:len(used_train2)]

#     # Tu peux ensuite sauvegarder, afficher ou utiliser ces prédictions comme tu veux
#     # Par exemple pour mesurer l'overfitting sur cette partie :
#     rmse_train2 = mean_squared_error(y_true_train2, y_pred_final_train2, squared=False)
#     mae_train2 = mean_absolute_error(y_true_train2, y_pred_final_train2)

#     print(f"Overfitting check - RMSE: {rmse_train2:.3f} | MAE: {mae_train2:.3f}")
#     # Création du DataFrame de résultats pour sauvegarde
#     train_pred_df = used_train2.reset_index(drop=True).copy()  # On travaille sur la partie prédite seulement
#     train_pred_df['y_pred'] = y_pred_final_train2
#     cols_to_save = ["unique_id", "ds", "Annee", "Semaine", "y", "y_pred"]
#     cols_to_save = [col for col in cols_to_save if col in train_pred_df.columns]
#     train_pred_df[cols_to_save].to_csv(os.path.join(save_dir, "train_pred.csv"), index=False)

#     # 6. Sauvegarde du modèle refitté (écrase l'ancien)
#     joblib.dump(forecast, model_pkl_path)
#     if verbose:
#         logging.info(f"Le modèle {best_model_name} (zone {zone}) a été ré-entraîné et sauvegardé dans {model_pkl_path} (ancien modèle écrasé)")


#     if verbose:
#         logging.info(f"Train et sauvegardes terminés pour {zone} - {target_col_name}")

#     return {
#         'zone': zone,
#         'best_model': best_model_name,
#         'best_params': best_config,
#         'predictions': train_pred_df,
#         'mlforecast': forecast,
#         'save_dir': save_dir
#     }
