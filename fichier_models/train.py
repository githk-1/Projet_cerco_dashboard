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
from src.utils_models import *

import sys
import os
import argparse

# Gestion des arguments
parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str, default=None)
parser.add_argument("--horizon", type=int, default=6)
args = parser.parse_args()

horizon = args.horizon

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), ".."))  
models_dir = os.path.join(project_root, f"models") 

print(f"==> Lancement avec horizon = {horizon} semaines")
################################### Importation des données et filtration des zones de traitement ###################################
path_data_brute, path_data_traite = get_data_paths()

df_merge = load_data(path_data_traite+ "/merge_final_mod_zone.parquet", database_name="merge1")
df_merge = df_merge.drop_duplicates(subset=["Zone_traitement", "Annee", "Semaine"], keep='first')


################################## Préparation de la modèlisation #####################################
indicateurs = [
    'Dp_moy',
    'Etat_devolution_moy',
    'Pjft_moyen',
    'Pjfn_moyen',
    'Nff_moyen',
    'Nfr_moyen'
]

for indicateur in indicateurs:
    print(f"\n--- Traitement de l'indicateur : {indicateur} ---")
    
    # Étape 1 : Préparation des données
    df_merge_m, target_col_name, available_exog  = prepare_data_zones(df_merge, target_col=indicateur, zone_col='Zone_traitement')
    df_merge_m = transform_exo(df_merge_m)

    
    # Étape 2 : Split train/test
    df_train, df_test = train_test_split_cercos_zones(df=df_merge_m, test_size=horizon, min_total_size=15, min_train_size=3, verbose=True)

    # Étape 3 : Optimisation des modèles
    resultats = optimize_models_for_all_zones(df_train, df_test, target_col_name=target_col_name, target_col="y", h=horizon, base_path= models_dir, n_trials=35, verbose=True)

