import sys
import os
# Ajoute le dossier parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.preparation_base import *
from src.plot_fonction import *
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
from mlforecast.lag_transforms import ExponentiallyWeightedMean, RollingMean, ExpandingMean, ExpandingMax,ExpandingMin,ExpandingStd,SeasonalRollingMean, RollingStd, RollingMax,RollingMin
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


import os
import contextlib
import streamlit as st
from tqdm.notebook import tqdm
from datetime import datetime, timedelta
import importlib
import pkg_resources

from src.model_fonction import *
import pickle 
from utilsforecast.losses import mape  
import warnings
warnings.filterwarnings('ignore')

from src.model_fonction import *
from src.preparation_base import definir_intensite
import argparse

# Gestion des arguments
parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str, default=None)
parser.add_argument("--horizon", type=int, default=4, help="nombre de semaines à prédire (test_size)")
args = parser.parse_args()

horizon = args.horizon

# Calcule le chemin absolu vers la racine du projet
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), ".."))  

# Chemin vers le dossier models à la racine
models_dir = os.path.join(project_root, "models")

################################### Importation des données et filtration des zones de traitement ###################################
path_data_brute, path_data_traite = get_data_paths()

df_merge = load_data(path_data_traite+ "/merge_final_mod_zone.parquet", database_name="merge1")
df_merge = df_merge.drop_duplicates(subset=["Zone_traitement", "Annee", "Semaine"], keep='first')

df_merge_sorted = df_merge.sort_values(['Zone_traitement', 'Annee', 'Semaine'], ascending=[True, True, True])

# 2. Pour supprimer les 4 dernières semaines de chaque zone :
df_merge_futur = (
    df_merge_sorted
    .groupby('Zone_traitement', group_keys=False)
    .apply(lambda x: x.iloc[:-24] if len(x) > 24 else x.iloc[0:0])
    .reset_index(drop=True)
)

print(df_merge[["Zone_traitement", "Station_meteo"]].nunique())

################################ Création de la base future #####################################


df_futur,derniere_annee, derniere_semaine = creer_semaines_futures_zones(
    df=df_merge_futur,
    colonnes_utiles=None,
    exog_list=None,  
    definir_intensite=definir_intensite
)



################################## Préparation et modélisation sur la base future #####################################
indicateurs = [
    'Dp_moy',
    'Etat_devolution_moy',
    'Pjft_moyen',
    'Pjfn_moyen',
    'Nff_moyen',
    'Nfr_moyen'
]

for indicateur in indicateurs:
    print(f"\n--- Traitement futur de l'indicateur : {indicateur} ---")
    
    # Étape 1 : Préparation des données pour la base future
    df_futur_m, target_col_name, available_exog  = prepare_data_zones(df_futur, target_col=indicateur, zone_col='Zone_traitement')
    df_futur_m = transform_exo(df_futur_m)
    # Étape 2 : Split spécifique à la base future
    df_trainf, df_testf = train_test_split_futur_zones(
    df_futur_m, split_annee=derniere_annee, split_semaine=derniere_semaine, test_size=horizon, min_total_size=15, min_train_size=3, verbose=True
)
    # Étape 3: Prédiction sur les données d'entrainement 

    results = train_all_zones_pred(df_train=df_trainf, base_path=models_dir, target_col_name=target_col_name,h=horizon, exog_cols=None, window_size=1, verbose=True,is_future= True)
    # Étape 4 : Optimisation et prédiction sur la base future
#     resultats_futur = predict_futur_for_all_zones_ML(
#     df_train=df_trainf,
#     df_test=df_testf,
#     target_col_name=target_col_name,
#     base_path=models_dir,
#     h=4,
#     verbose=True
# )

    resultats_futur = predict_futur_for_all_zones_ML2(
        df_train=df_trainf,
        df_test=df_testf,
        target_col_name=target_col_name,
        h=horizon,              
        base_path= models_dir,              # chemin où sont stockés les modèles par zone
        window_size=1,                     # rolling sur 1 semaine
        verbose=True                    
    )
