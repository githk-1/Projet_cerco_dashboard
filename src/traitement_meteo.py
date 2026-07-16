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
import pandas as pd
import logging

##### Importation des bases de données
path_data_brute, path_data_traite = get_data_paths()
df_cercos = load_data(path_data_traite+"/df_cercos_traite_mod_zone.parquet", database_name="cercos")
df_liaison= load_data(path_data_traite +"/df_liaison.xlsx", database_name= "liason")

df_meteo = load_data(path_data_brute + "/df_meteoQlick_act.xlsx", database_name="lick")
df_intrant_complet = load_data(path = path_data_traite+"/df_intrant_complet_mod_zone.parquet", database_name= "Intnt")

import logging
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from itertools import product

import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def enlever_stations_meteo(df_meteo):
    """
    Enlève les stations météo indésirables de la base météo
    (liste définie en dur dans la fonction) et log le nombre de stations restantes.

    Parameters
    ----------
    df_meteo : pd.DataFrame
        DataFrame météo d'origine

    Returns
    -------
    pd.DataFrame
        DataFrame filtré sans les stations à enlever
    """
    stations_a_enlever = [
        'Badema', 'Nieky Centre_manuelle','Nieky Nord_manuelle', 'Nieky Centre',
        'Nieky Nord', 'Nieky Sud' ,'Nieky Sud_manuelle', 'Vantage Confluence'
    ]
    df_meteo= df_meteo[~df_meteo['Station_meteo'].isin(stations_a_enlever)]
    logging.info(f"Stations météo restantes : {df_meteo['Station_meteo'].nunique()}")
    return df_meteo

def complete_weather_data(data):
    """
    Complète la base de données météo en ajoutant les jours manquants pour chaque station.
    Ajoute une colonne '_merge' pour identifier l'origine des données :
    - 'both' : date présente dans les données originales
    - 'left_only' : date ajoutée (manquante dans les données originales)   
     
     Parametres:
    -----------
    data : pd.DataFrame
        DataFrame contenant au minimum les colonnes 'Date' et 'Station_meteo'
    
    Returns :
    --------
    pd.DataFrame
        DataFrame complet avec tous les jours pour chaque station météo
        et une colonne '_merge' indiquant l'origine des données
    """
    #  Vérification des colonnes requises
    required_cols = ['Date', 'Station_meteo']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"La base de données doit contenir les colonnes : {missing_cols}")

    #  Conversion de la colonne Date si nécessaire
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])

    #  Obtenir la période globale
    date_min_global = data['Date'].min()
    date_max_global = data['Date'].max()


    #  Créer la grille complète de dates et stations
    stations = data['Station_meteo'].unique()
    all_dates = pd.date_range(start=date_min_global, end=date_max_global, freq='D', inclusive='both')

    logging.info(f" Jours à créer : {len(all_dates)} par station")

    #  Créer toutes les combinaisons (station x date)
    combinations = list(product(stations, all_dates))
    complete_grid = pd.DataFrame(combinations, columns=['Station_meteo', 'Date'])

    logging.info(f" Grille complète créée : {len(complete_grid):,} lignes")

    #  Fusion avec indicateur de source
    complete_data = pd.merge(
        complete_grid, 
        data, 
        on=['Date', 'Station_meteo'], 
        how='left',
        indicator=True
    )

    #  Tri pour une meilleure organisation
    complete_data = complete_data.sort_values(['Station_meteo', 'Date']).reset_index(drop=True)

    #  Statistiques finales
    original_rows = len(data)
    complete_rows = len(complete_data)
    added_rows = (complete_data['_merge'] == 'left_only').sum()

    logging.info(f" RÉSULTATS :")
    logging.info(f"   • Lignes originales : {original_rows:,}")
    logging.info(f"   • Lignes complètes : {complete_rows:,}")
    logging.info(f"   • Lignes ajoutées : {added_rows:,}")
    logging.info(f"   • Pourcentage complété : {(added_rows/complete_rows*100):.1f}%")

    return complete_data

def log_jours_ajoutes(df_meteo_complet):
    """
    Log le nombre de jours ajoutés, le détail par station et le nombre de lignes originales.
    """
    lignes_ajoutees = df_meteo_complet[df_meteo_complet['_merge'] == 'left_only']
    logging.info(f"Nombre de jours ajoutés : {len(lignes_ajoutees)}")
    logging.info("Détail par station :\n%s", lignes_ajoutees.groupby('Station_meteo').size())
    lignes_originales = df_meteo_complet[df_meteo_complet['_merge'] == 'both']
    logging.info(f"Nombre de lignes originales : {len(lignes_originales)}")

def imputation_jours_manquants_meteo(df_meteo_complet):
    """ 
    Imputation des jours manquants dans les données météo en utilisant KNNImputer. L'imputation est effectuée en deux étapes :
    1. Imputation de la température moyenne (Tmoy) en utilisant uniquement les variables temporelles. Car la Tmoy est une des 
    variables qui possède le moins de valeurs manquantes et qui est fortement corrélé avec les autres indicateurs météo.
    2. Imputation des autres indicateurs météo (Pluie, Humidité, Tmax, Tmin) en utilisant Tmoy et les variables temporelles.
    Ce qui nous permet de pouvoir imputer les jours manquants avec une information météo qui est corrélée avec les autres indicateurs
    qui nous donne donc un résultat plus robuste que le faire directement en une seule étape.                              
    """
    colonnes_meteo = ['Pluie','Humidite_max', 'Humidite_min', 'Tmax', 'Tmin', 'Tmoy']

    # Corriger le masque pour identifier les lignes ajoutées
    mask_lignes_ajoutees = df_meteo_complet['_merge'] == 'left_only'
    logging.info(f"Nombre de lignes ajoutées à imputer : {mask_lignes_ajoutees.sum()}")

    # Vérifier que ces lignes ont bien toutes les valeurs météo manquantes
    lignes_ajoutees_check = df_meteo_complet[mask_lignes_ajoutees]
    logging.info(f"Vérification - lignes avec toutes valeurs météo NaN : {lignes_ajoutees_check[colonnes_meteo].isna().all(axis=1).sum()}")

    # 2. Préparer les variables temporelles basées sur la Date
    # Convertir la date en timestamp numérique pour capturer la progression temporelle
    df_meteo_complet['Date_numeric'] = pd.to_datetime(df_meteo_complet['Date']).astype('int64') / 10**9  # Timestamp en secondes

    # Extraire le jour de l'année pour capturer les patterns saisonniers
    df_meteo_complet['Jour_annee'] = pd.to_datetime(df_meteo_complet['Date']).dt.dayofyear

    # Extraire le mois pour capturer les variations mensuelles
    df_meteo_complet['Mois'] = pd.to_datetime(df_meteo_complet['Date']).dt.month


    # ÉTAPE 1 : Imputation de 'Tmoy' uniquement (SANS Station_meteo)
    logging.info("=== ÉTAPE 1 : Imputation de Tmoy SANS Station_meteo ===")

    # Variables pour l'imputation de 'Tmoy' (SANS Station_meteo)
    num_cols_etape1 = ['Date_numeric', 'Jour_annee', 'Mois']  # Seulement temporel

    # Préparer les données pour l'imputation
    df_for_imputation_etape1 = df_meteo_complet[num_cols_etape1 + ['Tmoy']].copy()

    # Créer le preprocesseur (seulement standardisation)
    preprocessor_etape1 = StandardScaler()

    # Appliquer les transformations sur les features temporelles
    features_scaled = preprocessor_etape1.fit_transform(df_for_imputation_etape1[num_cols_etape1])

    # Recombiner avec la colonne cible
    data_for_knn = np.column_stack([features_scaled, df_for_imputation_etape1['Tmoy'].values])

    # Appliquer KNN pour l'imputation avec plus de voisins
    imputer_etape1 = KNNImputer(n_neighbors=5, weights='distance')
    imputed_data_etape1 = imputer_etape1.fit_transform(data_for_knn)

    # Récupérer les valeurs imputées pour 'Tmoy' (dernière colonne)
    tmoy_col_idx = -1

    # Remplacer Tmoy uniquement pour les lignes ajoutées
    mask_tmoy_lignes_ajoutees = mask_lignes_ajoutees & df_meteo_complet['Tmoy'].isna()
    df_meteo_complet.loc[mask_tmoy_lignes_ajoutees, 'Tmoy'] = imputed_data_etape1[mask_tmoy_lignes_ajoutees, tmoy_col_idx]

    logging.info(f"Tmoy imputé pour {mask_tmoy_lignes_ajoutees.sum()} lignes")

    
    # ÉTAPE 2 : Imputation des autres indicateurs météo (SANS Station_meteo)
    logging.info("=== ÉTAPE 2 : Imputation des autres indicateurs météo SANS Station_meteo ===")

    # Variables pour l'imputation des autres indicateurs (inclut maintenant Tmoy)
    num_cols_etape2 = ['Date_numeric', 'Jour_annee', 'Mois', 'Tmoy']  # Temporel + Tmoy
    autres_indicateurs = ['Pluie','Humidite_max', 'Humidite_min', 'Tmax', 'Tmin']

    # Préparer les données pour l'imputation des autres indicateurs
    df_for_imputation_etape2 = df_meteo_complet[num_cols_etape2 + autres_indicateurs].copy()

    # Créer le preprocesseur pour l'étape 2 (seulement standardisation)
    preprocessor_etape2 = StandardScaler()

    # Appliquer les transformations sur les features
    features_scaled_etape2 = preprocessor_etape2.fit_transform(df_for_imputation_etape2[num_cols_etape2])

    # Recombiner avec les colonnes cibles
    targets_data = df_for_imputation_etape2[autres_indicateurs].values
    data_for_knn_etape2 = np.column_stack([features_scaled_etape2, targets_data])

    # Appliquer KNN pour l'imputation
    imputer_etape2 = KNNImputer(n_neighbors=5, weights='distance')
    imputed_data_etape2 = imputer_etape2.fit_transform(data_for_knn_etape2)

    # Récupérer et remplacer les valeurs pour chaque indicateur
    n_features = len(num_cols_etape2)

    for i, indicateur in enumerate(autres_indicateurs):
        col_idx = n_features + i
        
        # Masque pour cet indicateur spécifiquement pour les lignes ajoutées
        mask_indicateur_lignes_ajoutees = mask_lignes_ajoutees & df_meteo_complet[indicateur].isna()
        
        # Remplacer les valeurs uniquement pour les lignes ajoutées
        df_meteo_complet.loc[mask_indicateur_lignes_ajoutees, indicateur] = imputed_data_etape2[mask_indicateur_lignes_ajoutees, col_idx]
        
        logging.info(f"{indicateur} imputé pour {mask_indicateur_lignes_ajoutees.sum()} lignes")

    
    # VÉRIFICATION DES RÉSULTATS
   

    logging.info("=== VÉRIFICATION DES RÉSULTATS ===")

    # Vérifier qu'il ne reste plus de valeurs manquantes dans les lignes ajoutées
    lignes_ajoutees_apres = df_meteo_complet[mask_lignes_ajoutees]
    valeurs_manquantes_apres = lignes_ajoutees_apres[colonnes_meteo].isna().sum()

    for col in colonnes_meteo:
        logging.info(f"{col}: {valeurs_manquantes_apres[col]}")

    if valeurs_manquantes_apres.sum() == 0:
        logging.info("Toutes les valeurs manquantes ont été imputées avec succès !")
    else:
        logging.warning("Il reste encore des valeurs manquantes")

    # Afficher quelques statistiques des lignes imputées
    logging.info("Statistiques des lignes imputées :\n%s", lignes_ajoutees_apres[colonnes_meteo].describe().round(2))

  
    # DIAGNOSTIC DE L'IMPUTATION
   

    logging.info("=== DIAGNOSTIC DE L'IMPUTATION ===")

    # Analyser les résultats de l'imputation par station
    for station in lignes_ajoutees_apres['Station_meteo'].unique():
        station_imputee = lignes_ajoutees_apres[lignes_ajoutees_apres['Station_meteo'] == station]
        station_originale = df_meteo_complet[(df_meteo_complet['Station_meteo'] == station) & (~mask_lignes_ajoutees)]
        
        logging.info(f"--- {station} ---")
        logging.info(f"Lignes imputées: {len(station_imputee)}")
        
        # Analyser les températures
        tmoy_stats = station_imputee['Tmoy'].describe()
        logging.info(f"Tmoy imputé - Min: {tmoy_stats['min']:.1f}, Max: {tmoy_stats['max']:.1f}, Moyenne: {tmoy_stats['mean']:.1f}")
        
        tmax_stats = station_imputee['Tmax'].describe()
        logging.info(f"Tmax imputé - Min: {tmax_stats['min']:.1f}, Max: {tmax_stats['max']:.1f}, Moyenne: {tmax_stats['mean']:.1f}")
        
        tmin_stats = station_imputee['Tmin'].describe()
        logging.info(f"Tmin imputé - Min: {tmin_stats['min']:.1f}, Max: {tmin_stats['max']:.1f}, Moyenne: {tmin_stats['mean']:.1f}")
        
        if len(station_originale) > 0:
            logging.info(f"Tmoy original - Min: {station_originale['Tmoy'].min():.1f}, Max: {station_originale['Tmoy'].max():.1f}, Moyenne: {station_originale['Tmoy'].mean():.1f}")
        
        # Détecter les valeurs suspectes pour les températures
        valeurs_suspectes_tmoy = station_imputee[(station_imputee['Tmoy'] < 15) | (station_imputee['Tmoy'] > 45)]
        valeurs_suspectes_tmax = station_imputee[(station_imputee['Tmax'] < 15) | (station_imputee['Tmax'] > 50)]
        valeurs_suspectes_tmin = station_imputee[(station_imputee['Tmin'] < 10) | (station_imputee['Tmin'] > 40)]
        
        if len(valeurs_suspectes_tmoy) > 0:
            logging.warning(f"   {len(valeurs_suspectes_tmoy)} valeurs Tmoy suspectes détectées!")
        if len(valeurs_suspectes_tmax) > 0:
            logging.warning(f"   {len(valeurs_suspectes_tmax)} valeurs Tmax suspectes détectées!")
        if len(valeurs_suspectes_tmin) > 0:
            logging.warning(f"   {len(valeurs_suspectes_tmin)} valeurs Tmin suspectes détectées!")

    # Nettoyer les colonnes temporaires
    df_meteo_complet.drop(columns=['Date_numeric', 'Jour_annee', 'Mois'], inplace=True)

    logging.info(f"Taille finale après imputation : {df_meteo_complet.shape}")

    # Retourner le dataframe après imputation
    return df_meteo_complet

def modif_meteo(df_meteo):
    """
    Exécute tout le pipeline de nettoyage météo :
    - enlève les stations à ignorer
    - complète les jours manquants
    - log les jours ajoutés
    - impute les valeurs manquantes sur les jours ajoutés

    Retourne la base météo prête à l'emploi.
    """
    try:
        df_meteo = enlever_stations_meteo(df_meteo)
    except Exception as e:
        logging.error(f"Erreur lors de la sélection des stations météo : {e}")
    
    try:
        df_meteo = df_meteo.sort_values(['Date', 'Station_meteo'])
        df_meteo_complet = complete_weather_data(df_meteo)
        log_jours_ajoutes(df_meteo_complet)
    except Exception as e:
        logging.error(f"Erreur lors de la complétion des données météo : {e}")
    try:
        df_meteo_complet = df_meteo_complet.sort_values(['Date', 'Station_meteo'])
        df_meteo_final = imputation_jours_manquants_meteo(df_meteo_complet)
    except Exception as e:
        logging.error(f"Erreur lors de l'imputation des jours manquants : {e}")
    return df_meteo_final


import logging
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def traiter_valeurs_aberrantes_et_features(df_meteo):
    """
    Traite les valeurs aberrantes et prépare les colonnes temporelles sur la base météo :
    - Met à NaN les Tmax > 42.5
    - Met à NaN les Tmin/Tmax <= 0
    - Met à NaN l'humidité max aberrante (48 pour 'spadi'), ou <= 0
    - Ajoute les colonnes Année et Semaine
    """
    # Remplacement des valeurs supérieures à 42.5 dans la colonne Tmax
    df_meteo.loc[df_meteo['Tmax'] > 42.5, 'Tmax'] = np.nan

    # Remplacer les valeurs inférieures ou égales à 0 par np.nan pour Tmin et Tmax
    df_meteo['Tmin'] = df_meteo['Tmin'].apply(lambda x: np.nan if x <= 0 else x)
    df_meteo['Tmax'] = df_meteo['Tmax'].apply(lambda x: np.nan if x <= 0 else x)

    # Remplacer la valeur d'humidité max aberrante par NaN
    df_meteo['Humidite_max'] = df_meteo['Humidite_max'].apply(lambda x: np.nan if x < 50 else x)

    # Extraire l'année et la semaine
    df_meteo["Annee"] = pd.to_datetime(df_meteo["Date"]).dt.isocalendar()["year"]
    df_meteo["Semaine"] = pd.to_datetime(df_meteo["Date"]).dt.isocalendar()["week"]

    logging.info("Valeurs aberrantes corrigées et colonnes Année/Semaine ajoutées")

    return df_meteo

def imputation_valeurs_manquantes_meteo(df_meteo):
    """
    Impute les valeurs manquantes de la base météo en deux passes KNN :
    1. Tmoy et Pluie avec Tmax, Tmin, Annee, Semaine, Station_meteo
    2. Humidite_max et Humidite_min avec les variables précédentes et les valeurs imputées
    Log les étapes et les résultats.
    """
    df_meteo_copy = df_meteo.copy()

    logging.info("=== IMPUTATION EN DEUX PASSES ===")
    logging.info(f"Valeurs manquantes avant imputation:")
    logging.info(f"Tmoy: {df_meteo_copy['Tmoy'].isna().sum()}")
    logging.info(f"Pluie: {df_meteo_copy['Pluie'].isna().sum()}")
    logging.info(f"Humidite_max: {df_meteo_copy['Humidite_max'].isna().sum()}")
    logging.info(f"Humidite_min: {df_meteo_copy['Humidite_min'].isna().sum()}")

    # PREMIÈRE PASSE : Imputation de Tmoy et Pluie
    logging.info("=== PREMIÈRE PASSE : Imputation de Tmoy et Pluie ===")
    cat_cols_pass1 = ['Station_meteo']
    num_cols_pass1 = ['Tmax', 'Tmin', 'Annee', 'Semaine']
    target_cols_pass1 = ['Tmoy', 'Pluie']

    df_for_imputation_pass1 = df_meteo_copy[cat_cols_pass1 + num_cols_pass1 + target_cols_pass1].copy()
    preprocessor_pass1 = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols_pass1),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols_pass1)
        ],
        remainder='passthrough'
    )
    transformed_data_pass1 = preprocessor_pass1.fit_transform(df_for_imputation_pass1)
    imputer_pass1 = KNNImputer(n_neighbors=5, weights='distance')
    imputed_data_pass1 = imputer_pass1.fit_transform(transformed_data_pass1)
    tmoy_index = transformed_data_pass1.shape[1] - 2
    pluie_index = transformed_data_pass1.shape[1] - 1
    tmoy_imputed = imputed_data_pass1[:, tmoy_index]
    pluie_imputed = imputed_data_pass1[:, pluie_index]
    mask_tmoy = df_meteo_copy['Tmoy'].isna()
    mask_pluie = df_meteo_copy['Pluie'].isna()
    df_meteo_copy.loc[mask_tmoy, 'Tmoy'] = tmoy_imputed[mask_tmoy]
    df_meteo_copy.loc[mask_pluie, 'Pluie'] = pluie_imputed[mask_pluie]
    logging.info(f"Tmoy imputé : {mask_tmoy.sum()} valeurs")
    logging.info(f"Pluie imputé : {mask_pluie.sum()} valeurs")

    # DEUXIÈME PASSE : Imputation de Humidite_max et Humidite_min
    logging.info("=== DEUXIÈME PASSE : Imputation de Humidite_max et Humidite_min ===")
    cat_cols_pass2 = ['Station_meteo']
    num_cols_pass2 = ['Tmax', 'Tmin', 'Tmoy', 'Pluie', 'Annee', 'Semaine']
    target_cols_pass2 = ['Humidite_max', 'Humidite_min']

    df_for_imputation_pass2 = df_meteo_copy[cat_cols_pass2 + num_cols_pass2 + target_cols_pass2].copy()
    preprocessor_pass2 = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols_pass2),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols_pass2)
        ],
        remainder='passthrough'
    )
    transformed_data_pass2 = preprocessor_pass2.fit_transform(df_for_imputation_pass2)
    imputer_pass2 = KNNImputer(n_neighbors=5, weights='distance')
    imputed_data_pass2 = imputer_pass2.fit_transform(transformed_data_pass2)
    humidity_max_index = transformed_data_pass2.shape[1] - 2
    humidity_min_index = transformed_data_pass2.shape[1] - 1
    humidity_max_imputed = imputed_data_pass2[:, humidity_max_index]
    humidity_min_imputed = imputed_data_pass2[:, humidity_min_index]
    mask_humidity_max = df_meteo_copy['Humidite_max'].isna()
    mask_humidity_min = df_meteo_copy['Humidite_min'].isna()
    df_meteo_copy.loc[mask_humidity_max, 'Humidite_max'] = humidity_max_imputed[mask_humidity_max]
    df_meteo_copy.loc[mask_humidity_min, 'Humidite_min'] = humidity_min_imputed[mask_humidity_min]
    logging.info(f"Humidite_max imputé : {mask_humidity_max.sum()} valeurs")
    logging.info(f"Humidite_min imputé : {mask_humidity_min.sum()} valeurs")

    # VÉRIFICATION DES RÉSULTATS
    logging.info("=== VÉRIFICATION DES RÉSULTATS ===")
    logging.info(f"Valeurs manquantes après imputation:")
    logging.info(f"Tmoy: {df_meteo_copy['Tmoy'].isna().sum()}")
    logging.info(f"Pluie: {df_meteo_copy['Pluie'].isna().sum()}")
    logging.info(f"Humidite_max: {df_meteo_copy['Humidite_max'].isna().sum()}")
    logging.info(f"Humidite_min: {df_meteo_copy['Humidite_min'].isna().sum()}")

    return df_meteo_copy

def clean_meteo(df_meteo):
    """
    Pipeline de nettoyage météorologique :
    - traite les valeurs aberrantes et ajoute année/semaine
    - impute les valeurs manquantes restantes
    """
    try:
        df_meteo = traiter_valeurs_aberrantes_et_features(df_meteo)
        df_meteo = imputation_valeurs_manquantes_meteo(df_meteo)
    except Exception as e:
        logging.error(f"Erreur lors du nettoyage des données météo : {e}")
    return df_meteo


import logging
import pandas as pd

# Fonctions déciles
def D1(x): return x.quantile(0.1)
def D2(x): return x.quantile(0.2)
def D3(x): return x.quantile(0.3)
def D4(x): return x.quantile(0.4)
def D5(x): return x.quantile(0.5)
def D6(x): return x.quantile(0.6)
def D7(x): return x.quantile(0.7)
def D8(x): return x.quantile(0.8)
def D9(x): return x.quantile(0.9)

def agregation_meteo_deciles(df_meteo):
    """
    Agrège les indicateurs météo au format hebdomadaire avec tous les déciles, 
    puis aplatit les colonnes multi-niveaux.
    """
    logging.info("="*25 + " Agrégation " + "="*25)
    try:
        df_agg = df_meteo.groupby(["Station_meteo","Annee", "Semaine"]).agg({
            "Tmoy": ["mean", D1, D2, D3, D4, D5, D6, D7, D8, D9],  
            "Tmax": ["mean", D1, D2, D3, D4, D5, D6, D7, D8, D9],  
            "Tmin": ["mean", D1, D2, D3, D4, D5, D6, D7, D8, D9],
            "Pluie": ["sum", D1, D2, D3, D4, D5, D6, D7, D8, D9],   
            "Humidite_max": ["mean", D1, D2, D3, D4, D5, D6, D7, D8, D9],  
            "Humidite_min": ["mean", D1, D2, D3, D4, D5, D6, D7, D8, D9]   
        }).reset_index()
    except Exception as e:
        logging.error(f"Erreur lors de l'agrégation des données météo : {e}")
        
    try:
        # Aplatir les colonnes multi-niveaux
        df_agg.columns = [
            '_'.join(col).strip() if col[1] else col[0] 
            for col in df_agg.columns.values
        ]
        logging.info("Nombre de valeurs manquantes" + "\n%s", df_agg.isna().sum())
    except Exception as e:
        logging.error(f"Erreur lors de l'aplatissement des colonnes : {e}")
    return df_agg

def mise_au_niveau_zones_meteo(df_meteo, df_liaison, df_cercos):
    """
    Ajoute la colonne Zone_traitement via jointure avec la table de liaison,
    supprime les doublons, et filtre pour ne garder que les zones présentes dans df_cercos.
    """
    logging.info("="*25 + " Mise au niveau les zones de traitement " + "="*25)
    try:
        df = df_meteo.merge(
            df_liaison[["Station_meteo", "Zone_traitement"]],
            on="Station_meteo",
            how="left"
        )   
    except Exception as e:
        logging.error(f"Erreur lors de la mise à niveau zone_traitement : {e}")
    # On supprime les lignes qui ont été dupliquées par l'ajout des zones de traitement
    before = df.shape[0]
    df = df.drop_duplicates(subset=['Annee', 'Semaine', 'Zone_traitement'])
    after = df.shape[0]
    logging.info("Nombre de ligne dupliquée après l'ajout des zones : %d", before - after)
    # On garde seulement les zones de traitement présentes dans la base cercos
    df = df[df['Zone_traitement'].isin(df_cercos['Zone_traitement'])]
    return df

def definir_intensite(pluviometrie):
    """
    Fonction pour catégoriser l'intensité de la pluie selon la somme glissante sur 4 semaines.
    """
    if pluviometrie >= 200:
        return 'Alerte'
    elif pluviometrie > 150:
        return 'Très_forte'
    elif pluviometrie > 100:
        return 'Forte'
    elif pluviometrie > 50:
        return 'Faible'
    else:
        return 'Calme'

def ajout_intensite_pluie(df_meteo):
    """
    Crée la variable d’intensité de pluie sur la base météo agrégée hebdomadaire.
    """
    try:
        df_meteo = df_meteo.sort_values(by=["Zone_traitement", "Annee", "Semaine"])
        # Calculer la somme glissante sur 4 semaines pour chaque zone de traitement
        df_meteo['Pluviometrie_4semaines'] = (
            df_meteo.groupby('Zone_traitement')['Pluie_sum']
            .rolling(window=4, min_periods=1).sum()
            .reset_index(0, drop=True)
        )
        # Créer la variable d'intensité de pluie
        df_meteo['Intensite_pluie'] = df_meteo['Pluviometrie_4semaines'].apply(definir_intensite)
    except Exception as e:
        logging.error(f"Erreur lors de l'ajout de l'intensité de pluie : {e}")
    return df_meteo

def traitement_meteo(df_meteo, df_liaison, df_cercos):
    """
    Pipeline complet pour le traitement de la base météo :
    - Nettoyage des stations et complétion des jours manquants
    - Nettoyage des valeurs aberrantes et imputation
    - Agrégation hebdomadaire avec déciles
    - Mise au niveau des zones
    - Calcul de l'intensité de pluie
    Log la shape et l'info finale de la base.
    """
    df_meteo = modif_meteo(df_meteo)
    df_meteo = clean_meteo(df_meteo)
    df_meteo = agregation_meteo_deciles(df_meteo)
    df_meteo = mise_au_niveau_zones_meteo(df_meteo, df_liaison, df_cercos)
    df_meteo = ajout_intensite_pluie(df_meteo)
    logging.info("shape de df_meteo: %s", df_meteo.shape)
    logging.info("info de df_meteo: %s", df_meteo.info())
    return df_meteo