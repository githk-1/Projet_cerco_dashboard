import sys
import os
# Ajoute le dossier parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.preparation_base import *
from src.plot_fonction import *
import warnings
warnings.filterwarnings("ignore")
import os
from src.model_fonction import *
import warnings
warnings.filterwarnings('ignore')

from src.model_fonction import *
from src.utils_models import *

import argparse

# Gestion des arguments
parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str, default=None)
parser.add_argument("--horizon", type=int, default=6)
args = parser.parse_args()

horizon = args.horizon

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), ".."))
models_dir = os.path.join(project_root, "models")
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
