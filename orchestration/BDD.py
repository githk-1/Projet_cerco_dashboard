import sys
import os
# Ajoute le dossier parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings("ignore")
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

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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