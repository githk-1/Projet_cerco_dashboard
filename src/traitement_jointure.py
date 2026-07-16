import pandas as pd
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from .preparation_base import *
from .plot_fonction import *
from .analyse_fonction import *

import warnings
warnings.filterwarnings("ignore")
import missingno as msno 
import sidetable as stb
import missingno as msno
from itertools import product
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from itertools import product
import logging


logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)

path_data_brute, path_data_traite = get_data_paths()
df_cercos = load_data(path_data_traite+"/df_cercos_traite_mod_zone.parquet", database_name="cercos")
df_intrant_complet = load_data(path_data_traite+"/df_intrant_complet_mod_zone.parquet", database_name="intrant_comp")
df_meteo = load_data(path= path_data_traite+"/df_meteoQlick_traite_mod_zone.parquet", database_name= "meteotraite")

def jointure_cercos_meteo(df_cercos, df_meteo):
    """
    Effectue une jointure gauche entre df_cercos et df_meteo sur ['Annee', 'Semaine', 'Zone_traitement'],
    trie le résultat et log les informations de la base.
    """
    merge1 = pd.merge(
        df_cercos,
        df_meteo,
        on=['Annee', 'Semaine', 'Zone_traitement'],
        how='left',
        suffixes=("_cercos", "_meteo")
    )
    merge1 = merge1.sort_values(by=["Annee", "Semaine"], ascending=[True, True])
    logging.info("Info de la base merge1: %s", merge1.info())
    return merge1

def jointure_avec_intrant(merge1, df_intrant_complet):
    """
    Effectue la jointure gauche de merge1 avec df_intrant_complet sur ['Annee', 'Semaine', 'Zone_traitement'],
    supprime les doublons, trie, convertit les types et log les infos de la base finale.
    """
    merge_final = pd.merge(
        merge1,
        df_intrant_complet,
        on=["Annee", "Semaine", "Zone_traitement"],
        how="left",
        suffixes=("", "_intrant")
    )
    logging.info(f"Nombre de lignes dans merge1: {merge1.shape[0]}")
    logging.info(f"Nombre de lignes dans merge_final: {merge_final.shape[0]}")
    # Supprimer les lignes dupliquées
    merge_final = merge_final.drop_duplicates()
    # Vérifier qu'il n'y a plus de doublons
    n_doublons = merge_final.duplicated().sum()
    logging.info(f"Nombre de doublons restants : {n_doublons}")
    merge_final = merge_final.sort_values(by=["Annee", "Semaine"], ascending=[True, True])
    logging.info("Shape de la base finale: %s", merge_final.shape)
    # Conversion des types de colonnes
    merge_final['Annee'] = merge_final['Annee'].astype('int32')
    merge_final['Semaine'] = merge_final['Semaine'].astype('int32')
    logging.info(merge_final.info())
    return merge_final

def traitement_jointure(df_cercos, df_meteo, df_intrant_complet):
    """
    Pipeline pour faire les deux jointures successives et nettoyage final.
    """
    try:
        merge1 = jointure_cercos_meteo(df_cercos, df_meteo)
    except Exception as e:
        logging.error(f"Erreur lors de la jointure entre df_cercos et df_meteo : {e}")

    try:
        merge_final = jointure_avec_intrant(merge1, df_intrant_complet)
    except Exception as e:
        logging.error(f"Erreur lors de la jointure entre merge1 et df_intrant_complet : {e}")
    return merge_final