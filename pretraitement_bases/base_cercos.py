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


########################## Importation des bases de données ##########################

path_data_brute, path_data_traite = get_data_paths()
df_cercos, df_liaison, df_coord = load_all_data_cercos(path_data_brute, path_data_traite)

df_cercos = traitement_cercos(df_cercos, df_coord)
df_cercos.to_parquet(path_data_traite + "/df_cercos_traite_mod.parquet", index=False)