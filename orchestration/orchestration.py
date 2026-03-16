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
import warnings
warnings.filterwarnings("ignore")
import missingno as msno 
import sidetable as stb
from itertools import product
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import logging

from src.recup_data import import_bases
from src.preparation_base import *
from src.plot_fonction import *
from src.analyse_fonction import *
from src.traitement_cercos import *
from src.traitement_cercos import load_all_data_cercos
from src.traitement_intrant import *
from src.traitement_meteo import *
from src.traitement_jointure import *
from src.model_fonction import *
from src.utils_models import *
from src.preparation_base import definir_intensite


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import missingno as msno 
import sidetable
import missingno as msno


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
import optuna 
import os
import contextlib
import streamlit as st
from tqdm.notebook import tqdm
from datetime import datetime, timedelta
import importlib
import pkg_resources
import pickle 
from utilsforecast.losses import mape  
import argparse



logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Gestion des arguments
parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str, default=None)
parser.add_argument("--horizon", type=int, default=6)
args = parser.parse_args()

horizon = args.horizon

logging.info("Début du processus d'importation des données...")
# Importation des bases de données depuis le fichier résaux.
import_bases()

logging.info("Début du traitement de la base cercos...")
path_data_brute, path_data_traite = get_data_paths()
df_cercos, df_liaison, df_coord = load_all_data_cercos(path_data_brute, path_data_traite)

df_cercos = traitement_cercos(df_cercos, df_coord)
df_cercos.to_parquet(path_data_traite + "/df_cercos_traite_mod_zone.parquet", index=False)


logging.info("Début du traitement de la base intrant...")

df_intrant = load_data(path= path_data_brute+"/Intrant.xlsx", database_name= "intrant")
df_cercos = load_data(path_data_traite+"/df_cercos_traite_mod_zone.parquet", database_name="cercos")
label_intrant= load_data(path = path_data_brute+"/intrant_label.xlsx", database_name= "intrant_label")
intrant_revu2 = load_data(path = path_data_traite+"/Intrant_revu2.xlsx", database_name= "intrant_rev")


df_intrant_complet = traitement_intrant(df_intrant, label_intrant, intrant_revu2, df_cercos, df_liaison)
df_intrant_complet.to_parquet(path_data_traite + "/df_intrant_complet_mod_zone.parquet", index=False)


logging.info("Début du traitement de la base météo...")

##### Importation des bases de données

df_cercos = load_data(path_data_traite+"/df_cercos_traite_mod_zone.parquet", database_name="cercos")
df_meteo = load_data(path_data_brute + "/df_meteoQlick_act.xlsx", database_name="25")

df_meteo = traitement_meteo(df_meteo, df_liaison, df_cercos)
df_meteo.to_parquet(path_data_traite + "/df_meteoQlick_traite_mod_zone.parquet", index=False)



logging.info("Début du traitement de la base de jointure...")
df_intrant_complet = load_data(path_data_traite+"/df_intrant_complet_mod_zone.parquet", database_name="intrant_comp")
df_meteo = load_data(path= path_data_traite+"/df_meteoQlick_traite_mod_zone.parquet", database_name= "meteotraite")


merge_final =  traitement_jointure(df_cercos, df_meteo, df_intrant_complet)
#df_merge = merge_final[~((merge_final["Annee"] == 2025) & (merge_final["Semaine"] == 34))]
merge_final.to_parquet(path= path_data_traite+"/merge_final_mod_zone.parquet", index=False)





logging.info("---- Début de la modélisation ----")

logging.info("Lancement de l'optimisation")
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), ".."))  
models_dir = os.path.join(project_root, "models") 
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


logging.info("Lancement des prévisions sur le test")
################################### Importation des données et filtration des zones de traitement ###################################


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

    # Étape 1 : Préparation des données
    df_merge_m, target_col_name, available_exog  = prepare_data_zones(df_merge, target_col=indicateur, zone_col='Zone_traitement')
    df_merge_m = transform_exo(df_merge_m)
    
    # Étape 2 : Split train/test
    df_train, df_test = train_test_split_cercos_zones(df=df_merge_m, test_size=horizon, min_total_size=15, min_train_size=3, verbose=True)

    # Étape 3 : Prédiction sur le train 
    results = train_all_zones_pred(df_train=df_train, base_path=models_dir, target_col_name=target_col_name,h=horizon, exog_cols=None, window_size=1, verbose=True)
    

    # Étape 4 : lancement des previsons sur le test
    #res = test_models_for_all_zones(df_train, df_test, target_col_name=target_col_name, target_col="y", h=4, base_path=models_dir, verbose=True)
    res = test_all_zones(df_train, df_test, base_path=models_dir, target_col="y", target_col_name=target_col_name, window_size=1, h=horizon, verbose=True)




logging.info("Lancement des prévisions pour les semaines futures")
################################### Importation des données et filtration des zones de traitement ###################################


df_merge = load_data(path_data_traite+ "/merge_final_mod_zone.parquet", database_name="merge1")
df_merge = df_merge.drop_duplicates(subset=["Zone_traitement", "Annee", "Semaine"], keep='first')

df_merge_sorted = df_merge.sort_values(['Zone_traitement', 'Annee', 'Semaine'], ascending=[True, True, True])

# 2. Pour supprimer les 4 dernières semaines de chaque zone :
df_merge_futur = (
    df_merge_sorted
    .groupby('Zone_traitement', group_keys=False)
    .apply(lambda x: x.iloc[:-horizon] if len(x) > horizon else x.iloc[0:0])
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
