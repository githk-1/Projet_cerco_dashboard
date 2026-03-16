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
from src.analyse_fonction import *
from src.traitement_jointure import *
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

#Importation des données
path_data_brute, path_data_traite = get_data_paths()
df_cercos = load_data(path_data_traite+"/df_cercos_traite_mod_zone.parquet", database_name="cercos")
df_intrant_complet = load_data(path_data_traite+"/df_intrant_complet_mod_zone.parquet", database_name="intrant_comp")
df_meteo = load_data(path= path_data_traite+"/df_meteoQlick_traite_mod_zone.parquet", database_name= "meteotraite")

df_cercos = df_cercos[~((df_cercos['Annee'] == 2025) & (df_cercos['Semaine'] == 24))].copy()
merge_final =  traitement_jointure(df_cercos, df_meteo, df_intrant_complet)

merge_final.to_parquet(path= path_data_traite+"/merge_final_mod_zone.parquet", index=False)
# Créer df_cercos_t en excluant la semaine 24 de 2025
#df_cercos = df_cercos[~((df_cercos['Annee'] == 2025) & (df_cercos['Semaine'] == 24))].copy()


# # Effectuer la jointure gauche on garde toute les lignes de cerco même si il n'ya pas de correspondance dans la base meteo

# merge1 = pd.merge(df_cercos, df_meteo, 
#                   on =['Annee', 'Semaine', 'Zone_traitement'], 
#                   how='left', 
#                   suffixes= ("_cercos", "_meteo"))

# merge1 = merge1.sort_values(by=["Annee", "Semaine"], ascending= [True,True])
# print("Info de la base merge1", merge1.info())



# # Fusion de merge1 avec df_intrant (jointure à gauche)

# merge_final = pd.merge(
#     merge1,                   
#     df_intrant_complet,                     
#     on=["Annee", "Semaine", "Zone_traitement"],  
#     how="left",                     
#     suffixes=("", "_intrant")       
# )

# # Vérification du résultat
# print(f"Nombre de lignes dans merge1: {merge1.shape[0]}")
# print(f"Nombre de lignes dans merge_final: {merge_final.shape[0]}")

# # Supprimer les lignes dupliquées
# merge_final = merge_final.drop_duplicates()

# # Vérifier qu'il n'y a plus de doublons
# print(f"Nombre de doublons restants : {merge_final.duplicated().sum()}")

# merge_final = merge_final.sort_values(by=["Annee", "Semaine"], ascending= [True, True])

# print("Shape de la base finale", merge_final.shape)

# # Conversion des types de colonnes
# merge_final['Annee'] = merge_final['Annee'].astype('int32')
# merge_final['Semaine'] = merge_final['Semaine'].astype('int32')

# print(merge_final.info())


