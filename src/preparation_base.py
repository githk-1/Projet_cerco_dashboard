import pandas as pd
import numpy as np
from .analyse_fonction import*
from .plot_fonction import *
import os 
# Définition des chemins de données
def get_data_paths():
    # Remonte à la racine du projet à partir de src
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path_data_brute = os.path.join(project_root, "data", "data_brute")
    path_data_traite = os.path.join(project_root, "data", "data_traite")
    return path_data_brute, path_data_traite


def detect_file_extension(path):
    return path.split('.')[-1].lower()

def read_file(path, file_extension):
    if file_extension in ['xlsx', 'xls']:
        return pd.read_excel(path)
    elif file_extension == 'csv':
        return pd.read_csv(path)
    elif file_extension == 'parquet':
        return pd.read_parquet(path)
    elif file_extension == 'pickle':
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Format de fichier non supporté: {file_extension}")

def replace_dash_with_nan(data, path):
    nb_avant = data.isna().sum().sum()
    data.replace("-", np.nan, inplace=True)
    nb_apres = data.isna().sum().sum()
    nb_remplace = nb_apres - nb_avant
    if nb_remplace > 0:
        print(f"Valeurs manquantes après remplacement des '-' dans {path.split('/')[-1]}:")
        print(data.isna().sum())
    return data

# def pivot_cntrlpro(data):
#     required_cols = ['id_control', 'jour', 'secteur', 'parcelle', 'contre_control', 'defi_code', 'valeur']
#     if all(col in data.columns for col in required_cols):
#         return pd.pivot_table(
#             data,
#             index=['id_control', 'jour', 'secteur', 'parcelle', 'contre_control'],
#             columns=['defi_code'],
#             values="valeur"
#         ).reset_index()
#     else:
#         print(f"Avertissement: Pivot non effectué pour 'cntrlpro' car colonnes manquantes")
#         return data

def rename_special_columns(data, database_name):
    if database_name.lower() in ["cerco", "cercos"]:
        data.rename(columns={"Poste d'observation": "Post_observation"}, inplace=True)
        data.rename(columns={"Zone de traitement": "Zone_traitement"}, inplace=True)
    if database_name.lower() in ["intrant", "intran"]:
        data.rename(columns={"Zone de traitement": "Zone_traitement"}, inplace=True)
    return data

def clean_column_names(data):
    data.columns = data.columns.str.replace(' ', '_') \
                              .str.replace("é", "e") \
                              .str.replace("°", "") \
                              .str.replace("(", "") \
                              .str.replace(")", "") \
                              .str.replace(".", "") \
                              .str.replace("'", "") \
                              .str.replace("+", "p") \
                              .str.lower() \
                              .str.capitalize()
    return data

def rename_jour_to_date(data, database_name):
    if database_name in ["25"] and 'Jour' in data.columns:
        data = data.rename(columns={'Jour': 'Date'})
    return data

def load_data(path, database_name):
    """
    Fonction de chargement de données qui lit un fichier (Excel, CSV ou Parquet) et effectue des opérations de nettoyage et de transformation.                                                                              

    Fonctionnalités :
    - Détection automatique du format de fichier et chargement approprié.
    - Remplacement des valeurs '-' par NaN pour faciliter le nettoyage.
    - Pivot automatique pour certaines bases (ex : 'meteo', 'cntrlpro') si les colonnes nécessaires sont présentes.
    - Renommage automatique de colonnes spécifiques pour la base 'cercos' (ex : 'Poste d'observation' → 'Post_observation').
    - Nettoyage des noms de colonnes : suppression des espaces, accents, caractères spéciaux, mise en minuscule puis capitalisation.
    - Renommage de la colonne 'Jour' en 'Date' pour les bases météo annuelles ("23", "24", "25").
    - Retourne un DataFrame Pandas prêt à l'emploi, ou un DataFrame vide en cas d'erreur.

    Args:
        path (str): Chemin vers le fichier (Excel, CSV ou Parquet)
        database_name (str): Nom de la base de données ('temperature', 'production', etc.)
    Returns:
        pd.DataFrame: DataFrame avec colonnes nettoyées
    """
    try:
        file_extension = detect_file_extension(path)
        data = read_file(path, file_extension)
        data = replace_dash_with_nan(data, path)
        # if database_name == "cntrlpro":
        #     data = pivot_cntrlpro(data)
        data = rename_special_columns(data, database_name)
        data = clean_column_names(data)
        data = rename_jour_to_date(data, database_name)
        if data is None:
            print(f"Erreur: Le DataFrame est None après traitement pour {database_name}")
            return pd.DataFrame()
        return data
    except Exception as e:
        print(f"Erreur lors du chargement de {path}: {str(e)}")
        return pd.DataFrame()
    
# Créer la variable d'intensité de pluie
def definir_intensite(pluviometrie):
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

################## Création des périodes futures ###############################

def creer_semaines_futures(
    df: pd.DataFrame,
    colonnes_utiles: list,
    exog_list: list,
    definir_intensite,  
    poste_col: str = 'Post_observation',
    annee_col: str = 'Annee',
    semaine_col: str = 'Semaine',
    intensite_cat: str = 'Intensite_pluie'
):
    """
    Sur une base 'df' complète, génère toutes les semaines futures jusqu'à la semaine 52 de l'année max
    pour chaque poste. Remplit les exogènes par la moyenne historique (poste, semaine),
    et calcule 'Intensite_pluie' pour les lignes futures selon la somme glissante de Pluie_sum.
    Seules les colonnes de 'colonnes_utiles' sont gardées.
    (Sans colonne 'ds')
    """

    if colonnes_utiles is None:
        colonnes_utiles = ['Annee', 'Semaine', 'Post_observation', 'Nff_moyen',
        'Nfr_moyen', 'Pjfn_moyen', 'Pjft_moyen', 'Etat_devolution_moy',
        'Dp_moy', 'Tmoy_mean', 'Tmoy_d1',
        'Tmoy_d2', 'Tmoy_d3', 'Tmoy_d4', 'Tmoy_d5', 'Tmoy_d6', 'Tmoy_d7',
        'Tmoy_d8', 'Tmoy_d9', 'Tmax_mean', 'Tmax_d1', 'Tmax_d2', 'Tmax_d3',
        'Tmax_d4', 'Tmax_d5', 'Tmax_d6', 'Tmax_d7', 'Tmax_d8', 'Tmax_d9',
        'Tmin_mean', 'Tmin_d1', 'Tmin_d2', 'Tmin_d3', 'Tmin_d4', 'Tmin_d5',
        'Tmin_d6', 'Tmin_d7', 'Tmin_d8', 'Tmin_d9', 'Pluie_sum', 'Pluie_d1',
        'Pluie_d2', 'Pluie_d3', 'Pluie_d4', 'Pluie_d5', 'Pluie_d6', 'Pluie_d7',
        'Pluie_d8', 'Pluie_d9', 'Humidite_max_mean', 'Humidite_max_d1',
        'Humidite_max_d2', 'Humidite_max_d3', 'Humidite_max_d4',
        'Humidite_max_d5', 'Humidite_max_d6', 'Humidite_max_d7',
        'Humidite_max_d8', 'Humidite_max_d9', 'Humidite_min_mean',
        'Humidite_min_d1', 'Humidite_min_d2', 'Humidite_min_d3',
        'Humidite_min_d4', 'Humidite_min_d5', 'Humidite_min_d6',
        'Humidite_min_d7', 'Humidite_min_d8', 'Humidite_min_d9', 'Intensite_pluie']

    if exog_list is None:
        exog_list = ['Tmoy_mean', 'Tmoy_d1',
        'Tmoy_d2', 'Tmoy_d3', 'Tmoy_d4', 'Tmoy_d5', 'Tmoy_d6', 'Tmoy_d7',
        'Tmoy_d8', 'Tmoy_d9', 'Tmax_mean', 'Tmax_d1', 'Tmax_d2', 'Tmax_d3',
        'Tmax_d4', 'Tmax_d5', 'Tmax_d6', 'Tmax_d7', 'Tmax_d8', 'Tmax_d9',
        'Tmin_mean', 'Tmin_d1', 'Tmin_d2', 'Tmin_d3', 'Tmin_d4', 'Tmin_d5',
        'Tmin_d6', 'Tmin_d7', 'Tmin_d8', 'Tmin_d9', 'Pluie_sum', 'Pluie_d1',
        'Pluie_d2', 'Pluie_d3', 'Pluie_d4', 'Pluie_d5', 'Pluie_d6', 'Pluie_d7',
        'Pluie_d8', 'Pluie_d9', 'Humidite_max_mean', 'Humidite_max_d1',
        'Humidite_max_d2', 'Humidite_max_d3', 'Humidite_max_d4',
        'Humidite_max_d5', 'Humidite_max_d6', 'Humidite_max_d7',
        'Humidite_max_d8', 'Humidite_max_d9', 'Humidite_min_mean',
        'Humidite_min_d1', 'Humidite_min_d2', 'Humidite_min_d3',
        'Humidite_min_d4', 'Humidite_min_d5', 'Humidite_min_d6',
        'Humidite_min_d7', 'Humidite_min_d8', 'Humidite_min_d9', 'Intensite_pluie']

    # 1. Filtrer la base sur les colonnes utiles
    df_all = df[colonnes_utiles].copy()
    
    # 2. Colonnes exogènes disponibles
    available_exog = [col for col in exog_list if col in df_all.columns]
    
    # 2. bis on prend un object qui est l'annee et semaine max disponible dans la base à data 

    derniere_annee = df_all[annee_col].max()
    derniere_semaine = df_all[df_all[annee_col] == derniere_annee][semaine_col].max()

    # 3. Liste des postes
    postes = df_all[poste_col].unique()
    futures_data = []

    # 4. Pour chaque poste, détecter (année, semaine) max
    for poste in postes:
        df_poste = df_all[df_all[poste_col] == poste].copy()
        if df_poste.empty:
            continue
        derniere_annee = df_poste[annee_col].max()
        derniere_semaine = df_poste[df_poste[annee_col] == derniere_annee][semaine_col].max()
        historique_poste = df_poste.copy()
        for future_semaine in range(derniere_semaine+1, 53):  # 53 exclu
            future_annee = derniere_annee
            row = {
                poste_col: poste,
                annee_col: future_annee,
                semaine_col: future_semaine,
            }
            # Exogènes classiques
            for exog in available_exog:
                if exog == intensite_cat:
                    continue  # remplir plus bas
                else:
                    historique = historique_poste[
                        (historique_poste[semaine_col] == future_semaine) & 
                        (historique_poste[annee_col] < future_annee)
                    ][exog]
                    if not historique.empty:
                        row[exog] = historique.mean()
                    else:
                        row[exog] = historique_poste[exog].mean()
            # Calcul Pluviometrie_4semaines
            temp = pd.concat([historique_poste, pd.DataFrame([row])], ignore_index=True)
            temp = temp.sort_values(by=[annee_col, semaine_col])
            temp['Pluviometrie_4semaines'] = temp['Pluie_sum'].rolling(window=4, min_periods=1).sum()
            pluvio_4sem = temp.iloc[-1]['Pluviometrie_4semaines']
            row[intensite_cat] = definir_intensite(pluvio_4sem)
            historique_poste = temp.drop(columns=['Pluviometrie_4semaines'])
            futures_data.append(row)

    df_futures = pd.DataFrame(futures_data)
    
    # 5. Assembler historique + futures
    df_final = pd.concat([df_all, df_futures], ignore_index=True, sort=False)
    df_final = df_final.sort_values([poste_col, annee_col, semaine_col]).reset_index(drop=True)
    if 'Pluviometrie_4semaines' in df_final.columns:
        df_final = df_final.drop(columns=['Pluviometrie_4semaines'])
    # 6. Ne garder que les colonnes utiles
    cols_finales = [col for col in colonnes_utiles if col in df_final.columns]
    for col in [poste_col, annee_col, semaine_col, intensite_cat]:
        if col not in cols_finales:
            cols_finales.append(col)
    df_final = df_final[cols_finales]
    return df_final, derniere_annee, derniere_semaine








def creer_semaines_futures_zones(
    df: pd.DataFrame,
    colonnes_utiles: list,
    exog_list: list,
    definir_intensite,  
    zone_col: str = 'Zone_traitement',
    annee_col: str = 'Annee',
    semaine_col: str = 'Semaine',
    intensite_cat: str = 'Intensite_pluie'
):
    """
    Sur une base 'df' complète, génère toutes les semaines futures jusqu'à la semaine 52 de l'année max
    pour chaque zone de traitement. Remplit les exogènes par la moyenne historique (zone, semaine),
    et calcule 'Intensite_pluie' pour les lignes futures selon la somme glissante de Pluie_sum.
    Seules les colonnes de 'colonnes_utiles' sont gardées.
    (Sans colonne 'ds')
    """

    if colonnes_utiles is None:
        colonnes_utiles = ['Annee', 'Semaine', 'Zone_traitement', 'Nff_moyen',
        'Nfr_moyen', 'Pjfn_moyen', 'Pjft_moyen', 'Etat_devolution_moy',
        'Dp_moy', 'Tmoy_mean', 'Tmoy_d1',
        'Tmoy_d2', 'Tmoy_d3', 'Tmoy_d4', 'Tmoy_d5', 'Tmoy_d6', 'Tmoy_d7',
        'Tmoy_d8', 'Tmoy_d9', 'Tmax_mean', 'Tmax_d1', 'Tmax_d2', 'Tmax_d3',
        'Tmax_d4', 'Tmax_d5', 'Tmax_d6', 'Tmax_d7', 'Tmax_d8', 'Tmax_d9',
        'Tmin_mean', 'Tmin_d1', 'Tmin_d2', 'Tmin_d3', 'Tmin_d4', 'Tmin_d5',
        'Tmin_d6', 'Tmin_d7', 'Tmin_d8', 'Tmin_d9', 'Pluie_sum', 'Pluie_d1',
        'Pluie_d2', 'Pluie_d3', 'Pluie_d4', 'Pluie_d5', 'Pluie_d6', 'Pluie_d7',
        'Pluie_d8', 'Pluie_d9', 'Humidite_max_mean', 'Humidite_max_d1',
        'Humidite_max_d2', 'Humidite_max_d3', 'Humidite_max_d4',
        'Humidite_max_d5', 'Humidite_max_d6', 'Humidite_max_d7',
        'Humidite_max_d8', 'Humidite_max_d9', 'Humidite_min_mean',
        'Humidite_min_d1', 'Humidite_min_d2', 'Humidite_min_d3',
        'Humidite_min_d4', 'Humidite_min_d5', 'Humidite_min_d6',
        'Humidite_min_d7', 'Humidite_min_d8', 'Humidite_min_d9', 'Intensite_pluie']

    if exog_list is None:
        exog_list = ['Tmoy_mean', 'Tmoy_d1',
        'Tmoy_d2', 'Tmoy_d3', 'Tmoy_d4', 'Tmoy_d5', 'Tmoy_d6', 'Tmoy_d7',
        'Tmoy_d8', 'Tmoy_d9', 'Tmax_mean', 'Tmax_d1', 'Tmax_d2', 'Tmax_d3',
        'Tmax_d4', 'Tmax_d5', 'Tmax_d6', 'Tmax_d7', 'Tmax_d8', 'Tmax_d9',
        'Tmin_mean', 'Tmin_d1', 'Tmin_d2', 'Tmin_d3', 'Tmin_d4', 'Tmin_d5',
        'Tmin_d6', 'Tmin_d7', 'Tmin_d8', 'Tmin_d9', 'Pluie_sum', 'Pluie_d1',
        'Pluie_d2', 'Pluie_d3', 'Pluie_d4', 'Pluie_d5', 'Pluie_d6', 'Pluie_d7',
        'Pluie_d8', 'Pluie_d9', 'Humidite_max_mean', 'Humidite_max_d1',
        'Humidite_max_d2', 'Humidite_max_d3', 'Humidite_max_d4',
        'Humidite_max_d5', 'Humidite_max_d6', 'Humidite_max_d7',
        'Humidite_max_d8', 'Humidite_max_d9', 'Humidite_min_mean',
        'Humidite_min_d1', 'Humidite_min_d2', 'Humidite_min_d3',
        'Humidite_min_d4', 'Humidite_min_d5', 'Humidite_min_d6',
        'Humidite_min_d7', 'Humidite_min_d8', 'Humidite_min_d9', 'Intensite_pluie']

    # 1. Filtrer la base sur les colonnes utiles
    df_all = df[colonnes_utiles].copy()
    
    # 2. Colonnes exogènes disponibles
    available_exog = [col for col in exog_list if col in df_all.columns]
    
    # 2. bis on prend un object qui est l'annee et semaine max disponible dans la base à data 

    derniere_annee = df_all[annee_col].max()
    derniere_semaine = df_all[df_all[annee_col] == derniere_annee][semaine_col].max()

    # 3. Liste des zones
    zones = df_all[zone_col].unique()
    futures_data = []

    # 4. Pour chaque zone, détecter (année, semaine) max
    for zone in zones:
        df_zone = df_all[df_all[zone_col] == zone].copy()
        if df_zone.empty:
            continue
        derniere_annee = df_zone[annee_col].max()
        derniere_semaine = df_zone[df_zone[annee_col] == derniere_annee][semaine_col].max()
        historique_zone = df_zone.copy()
        for future_semaine in range(derniere_semaine+1, 53):  # 53 exclu
            future_annee = derniere_annee
            row = {
                zone_col: zone,
                annee_col: future_annee,
                semaine_col: future_semaine,
            }
            # Exogènes classiques
            for exog in available_exog:
                if exog == intensite_cat:
                    continue  # remplir plus bas
                else:
                    historique = historique_zone[
                        (historique_zone[semaine_col] == future_semaine) & 
                        (historique_zone[annee_col] < future_annee)
                    ][exog]
                    if not historique.empty:
                        row[exog] = historique.mean()
                    else:
                        row[exog] = historique_zone[exog].mean()
            # Calcul Pluviometrie_4semaines
            temp = pd.concat([historique_zone, pd.DataFrame([row])], ignore_index=True)
            temp = temp.sort_values(by=[annee_col, semaine_col])
            temp['Pluviometrie_4semaines'] = temp['Pluie_sum'].rolling(window=4, min_periods=1).sum()
            pluvio_4sem = temp.iloc[-1]['Pluviometrie_4semaines']
            row[intensite_cat] = definir_intensite(pluvio_4sem)
            historique_zone = temp.drop(columns=['Pluviometrie_4semaines'])
            futures_data.append(row)

    df_futures = pd.DataFrame(futures_data)
    
    # 5. Assembler historique + futures
    df_final = pd.concat([df_all, df_futures], ignore_index=True, sort=False)
    df_final = df_final.sort_values([zone_col, annee_col, semaine_col]).reset_index(drop=True)
    if 'Pluviometrie_4semaines' in df_final.columns:
        df_final = df_final.drop(columns=['Pluviometrie_4semaines'])
    # 6. Ne garder que les colonnes utiles
    cols_finales = [col for col in colonnes_utiles if col in df_final.columns]
    for col in [zone_col, annee_col, semaine_col, intensite_cat]:
        if col not in cols_finales:
            cols_finales.append(col)
    df_final = df_final[cols_finales]
    return df_final, derniere_annee, derniere_semaine























import pandas as pd

import pandas as pd

def prep_base_futurt(
    df: pd.DataFrame,
    definir_intensite,
    h: int = 4,
    colonnes_utiles: list = None,
    exog_list: list = None,
    poste_col: str = 'Post_observation',
    annee_col: str = 'Annee',
    semaine_col: str = 'Semaine',
    intensite_cat: str = 'Intensite_pluie'
):
    """
    Pour chaque poste de la base, sélectionne les h dernières semaines (en se basant sur année/semaine max du poste).
    Pour ces lignes, remplace les valeurs des exogènes par leur moyenne historique (même semaine, années précédentes, même poste).
    Recalcule l'indicateur intensité pluie selon la somme glissante de Pluie_sum sur 4 semaines.
    Les colonnes utiles et exogènes sont identiques à la fonction d'origine.
    Retourne la base modifiée.
    Affiche pour chaque poste les semaines modifiées et si tous les exogènes ont bien été modifiés.
    """

    if colonnes_utiles is None:
        colonnes_utiles = ['Annee', 'Semaine', 'Post_observation', 'Nff_moyen',
        'Nfr_moyen', 'Pjfn_moyen', 'Pjft_moyen', 'Etat_devolution_moy',
        'Dp_moy', 'Tmoy_mean', 'Tmoy_d1',
        'Tmoy_d2', 'Tmoy_d3', 'Tmoy_d4', 'Tmoy_d5', 'Tmoy_d6', 'Tmoy_d7',
        'Tmoy_d8', 'Tmoy_d9', 'Tmax_mean', 'Tmax_d1', 'Tmax_d2', 'Tmax_d3',
        'Tmax_d4', 'Tmax_d5', 'Tmax_d6', 'Tmax_d7', 'Tmax_d8', 'Tmax_d9',
        'Tmin_mean', 'Tmin_d1', 'Tmin_d2', 'Tmin_d3', 'Tmin_d4', 'Tmin_d5',
        'Tmin_d6', 'Tmin_d7', 'Tmin_d8', 'Tmin_d9', 'Pluie_sum', 'Pluie_d1',
        'Pluie_d2', 'Pluie_d3', 'Pluie_d4', 'Pluie_d5', 'Pluie_d6', 'Pluie_d7',
        'Pluie_d8', 'Pluie_d9', 'Humidite_max_mean', 'Humidite_max_d1',
        'Humidite_max_d2', 'Humidite_max_d3', 'Humidite_max_d4',
        'Humidite_max_d5', 'Humidite_max_d6', 'Humidite_max_d7',
        'Humidite_max_d8', 'Humidite_max_d9', 'Humidite_min_mean',
        'Humidite_min_d1', 'Humidite_min_d2', 'Humidite_min_d3',
        'Humidite_min_d4', 'Humidite_min_d5', 'Humidite_min_d6',
        'Humidite_min_d7', 'Humidite_min_d8', 'Humidite_min_d9', 'Intensite_pluie']

    if exog_list is None:
        exog_list = ['Tmoy_mean', 'Tmoy_d1',
        'Tmoy_d2', 'Tmoy_d3', 'Tmoy_d4', 'Tmoy_d5', 'Tmoy_d6', 'Tmoy_d7',
        'Tmoy_d8', 'Tmoy_d9', 'Tmax_mean', 'Tmax_d1', 'Tmax_d2', 'Tmax_d3',
        'Tmax_d4', 'Tmax_d5', 'Tmax_d6', 'Tmax_d7', 'Tmax_d8', 'Tmax_d9',
        'Tmin_mean', 'Tmin_d1', 'Tmin_d2', 'Tmin_d3', 'Tmin_d4', 'Tmin_d5',
        'Tmin_d6', 'Tmin_d7', 'Tmin_d8', 'Tmin_d9', 'Pluie_sum', 'Pluie_d1',
        'Pluie_d2', 'Pluie_d3', 'Pluie_d4', 'Pluie_d5', 'Pluie_d6', 'Pluie_d7',
        'Pluie_d8', 'Pluie_d9', 'Humidite_max_mean', 'Humidite_max_d1',
        'Humidite_max_d2', 'Humidite_max_d3', 'Humidite_max_d4',
        'Humidite_max_d5', 'Humidite_max_d6', 'Humidite_max_d7',
        'Humidite_max_d8', 'Humidite_max_d9', 'Humidite_min_mean',
        'Humidite_min_d1', 'Humidite_min_d2', 'Humidite_min_d3',
        'Humidite_min_d4', 'Humidite_min_d5', 'Humidite_min_d6',
        'Humidite_min_d7', 'Humidite_min_d8', 'Humidite_min_d9', 'Intensite_pluie']

    df_all = df[colonnes_utiles].copy()
    available_exog = [col for col in exog_list if col in df_all.columns]

    postes = df_all[poste_col].unique()
    df_modif = df_all.copy()

    for poste in postes:
        df_poste = df_modif[df_modif[poste_col] == poste].copy()
        if df_poste.empty:
            continue
        df_poste = df_poste.sort_values([annee_col, semaine_col])
        annee_max = df_poste[annee_col].max()
        semaine_max = df_poste[df_poste[annee_col] == annee_max][semaine_col].max()
        df_poste_h = df_poste[
            ((df_poste[annee_col] == annee_max) & (df_poste[semaine_col] >= semaine_max - h + 1))
        ].copy()

        semaines_changees = []
        exog_changement = {}

        for idx, row in df_poste_h.iterrows():
            future_semaine = row[semaine_col]
            future_annee = row[annee_col]
            semaines_changees.append((future_annee, future_semaine))
            exog_changement[(future_annee, future_semaine)] = []

            for exog in available_exog:
                if exog == intensite_cat:
                    continue
                historique = df_poste[
                    (df_poste[semaine_col] == future_semaine) &
                    (df_poste[annee_col] < future_annee)
                ][exog]
                if not historique.empty:
                    df_modif.at[idx, exog] = historique.mean()
                else:
                    df_modif.at[idx, exog] = df_poste[exog].mean()
                exog_changement[(future_annee, future_semaine)].append(exog)

            temp = df_poste.copy()
            temp.loc[idx, available_exog] = df_modif.loc[idx, available_exog]
            temp = temp.sort_values([annee_col, semaine_col])
            temp['Pluviometrie_4semaines'] = temp['Pluie_sum'].rolling(window=4, min_periods=1).sum()
            pluvio_4sem = temp.loc[idx, 'Pluviometrie_4semaines']
            df_modif.at[idx, intensite_cat] = definir_intensite(pluvio_4sem)

        # PRINTS pour debug
        print(f"\nPoste : {poste}")
        print(f"Semaines modifiées (Année, Semaine) : {semaines_changees}")
        for s in semaines_changees:
            exogs = exog_changement[s]
            if set(exogs) == set([e for e in available_exog if e != intensite_cat]):
                print(f"- Semaine {s[1]} Année {s[0]} : Tous les exogènes ont été changés.")
            else:
                print(f"- Semaine {s[1]} Année {s[0]} : Changement partiel des exogènes : {exogs}")

    if 'Pluviometrie_4semaines' in df_modif.columns:
        df_modif = df_modif.drop(columns=['Pluviometrie_4semaines'])
    df_modif = df_modif.sort_values([poste_col, annee_col, semaine_col]).reset_index(drop=True)
    cols_finales = [col for col in colonnes_utiles if col in df_modif.columns]
    for col in [poste_col, annee_col, semaine_col, intensite_cat]:
        if col not in cols_finales:
            cols_finales.append(col)
    df_modif = df_modif[cols_finales]
    return df_modif