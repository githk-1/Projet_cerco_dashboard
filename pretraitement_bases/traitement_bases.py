
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
from src.traitement_cercos import *
from src.traitement_intrant import *
from src.traitement_meteo import *
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


############################## Base cercos #############################

path_data_brute, path_data_traite = get_data_paths()
df_cercos, df_liaison, df_coord = load_all_data_cercos(path_data_brute, path_data_traite)

#traitement de la base cercos
df_cercos = traitement_cercos(df_cercos, df_coord)
df_cercos.to_parquet(path_data_traite + "/df_cercos_traite_mod_zone.parquet", index=False)

######################################### Base Intrant #########################################
df_intrant = load_data(path= path_data_brute+"/Intrant.xlsx", database_name= "intrant")
df_liaison = load_data(path_data_traite + "/df_liaison.xlsx", database_name="liason")
df_cercos = load_data(path_data_traite+"/df_cercos_traite_mod_zone.parquet", database_name="cercos")
label_intrant= load_data(path = path_data_brute+"/intrant_label.xlsx", database_name= "intrant_label")
intrant_revu2 = load_data(path = path_data_traite+"/Intrant_revu2.xlsx", database_name= "intrant_rev")


df_intrant_complet = traitement_intrant(df_intrant, label_intrant, intrant_revu2, df_cercos, df_liaison)

df_intrant_complet.to_parquet(path_data_traite + "/df_intrant_complet_mod_zone.parquet", index=False)


################################################# base météo #################################################

##### Importation des bases de données
path_data_brute, path_data_traite = get_data_paths()
df_cercos = load_data(path_data_traite+"/df_cercos_traite_mod_zone.parquet", database_name="cercos")
df_liaison= load_data(path_data_traite +"/df_liaison.xlsx", database_name= "liason")
df_meteo = load_data(path_data_brute + "/df_meteoQlick_act.xlsx", database_name="25")

df_meteo = traitement_meteo(df_meteo, df_liaison, df_cercos)

df_meteo.to_parquet(path_data_traite + "/df_meteoQlick_traite_mod_zone.parquet", index=False)


################################################# base jointure #################################################
df_intrant_complet = load_data(path_data_traite+"/df_intrant_complet_mod_zone.parquet", database_name="intrant_comp")
df_meteo = load_data(path= path_data_traite+"/df_meteoQlick_traite_mod_zone.parquet", database_name= "meteotraite")


merge_final =  traitement_jointure(df_cercos, df_meteo, df_intrant_complet)
merge_final.to_parquet(path= path_data_traite+"/merge_final_mod_zone.parquet", index=False)