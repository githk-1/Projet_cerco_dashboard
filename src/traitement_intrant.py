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


import pandas as pd
import logging


logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Importation des données
path_data_brute, path_data_traite = get_data_paths()
df_intrant = load_data(path= path_data_brute+"/Intrant_v4.xlsx", database_name= "intrant")
df_liaison = load_data(path_data_traite + "/df_liaison.xlsx", database_name="liason")
df_cercos = load_data(path_data_traite+"/df_cercos_traite_mod_zone.parquet", database_name="cercos")
label_intrant= load_data(path = path_data_brute+"/intrant_label.xlsx", database_name= "intrant_label")
intrant_revu2 = load_data(path = path_data_traite+"/Intrant_revu2.xlsx", database_name= "intrant_rev")


def extraire_annee_semaine_intrant(df_intrant):
    """
    Ajoute deux colonnes à df_intrant :
    - 'Annee' : extraite de la date d’utilisation (année ISO)
    - 'Semaine' : extraite de la date d’utilisation (numéro de semaine ISO)
    Trie ensuite le DataFrame par Année, Semaine et Zone_traitement.
    """
    df_intrant["Annee"] = pd.to_datetime(df_intrant["Date_dutilisation"]).dt.isocalendar()["year"]
    df_intrant["Semaine"] = pd.to_datetime(df_intrant["Date_dutilisation"]).dt.isocalendar()["week"]
    df_intrant = df_intrant.sort_values(by=['Annee', 'Semaine', 'Zone_traitement'])
    return df_intrant

def traitement_colonnes_intrant(df_intrant):
    """
    Traite les colonnes ayant potentiellement des valeurs manquantes :
    - Remplace les NaN de 'Nb_jours_entre_2_traitements' et 'Quantite_huile' par 0.
    Fait ensuite un contrôle sur les valeurs manquantes et les doublons avec des logs.
    """
    df_intrant['Nb_jours_entre_2_traitements'].fillna(0, inplace=True)
    df_intrant["Quantite_huile"].fillna(0, inplace=True)
    logging.info("Nombre de valeurs manquantes après le traitement:\n%s", df_intrant.isna().sum())
    logging.info("Nombre de doublons dans la base:\n%s", df_intrant.duplicated().sum())
    return df_intrant

def creation_categorie_intrant(df_intrant, label_intrant):
    """
    Ajoute une colonne 'Intrant_revu' à df_intrant après mapping avec un dictionnaire de labels,
    à partir du DataFrame label_intrant. Logge les intrants non mappés pour contrôle.
    """    
    label = dict(zip(label_intrant['Intrant'], label_intrant['Intrant_revu']))
    df_intrant["Intrant_revu"] = df_intrant["Intrant"].map(label)

    intrants_non_mappees = df_intrant[~df_intrant["Intrant"].isin(label.keys())]

    if not intrants_non_mappees.empty:
        logging.warning("Les intrants suivants ne sont pas dans le dictionnaire de mapping :")
        logging.warning("%s", intrants_non_mappees["Intrant"].unique())
    else:
        logging.info("Tous les intrants sont présents dans le dictionnaire de mapping.")
    return df_intrant

def second_mapping_intrant(df_intrant, intrant_revu2):
    """
    Effectue un second mapping de la catégorie d’intrant :
    - Ajoute une colonne 'Intrant_revu2' obtenue à partir d’un mapping sur la colonne 'Intrant_revu'.
    Logge les dictionnaires utilisés, le nombre de valeurs (non) mappées et la répartition finale.
    """
    mapping_intrant_revu2 = dict(zip(intrant_revu2['Intrant_revu'], intrant_revu2['Intrant_revu2']))

    logging.info("Dictionnaire de mapping Intrant_revu -> Intrant_revu2 : %s", mapping_intrant_revu2)

    df_intrant['Intrant_revu2'] = df_intrant['Intrant_revu'].map(mapping_intrant_revu2)
    logging.info("Nombre de valeurs mappées : %s", df_intrant['Intrant_revu2'].notna().sum())
    logging.info("Nombre de valeurs non mappées : %s", df_intrant['Intrant_revu2'].isna().sum())

    valeurs_non_mappees = df_intrant[df_intrant['Intrant_revu2'].isna()]['Intrant_revu'].unique()
    if len(valeurs_non_mappees) > 0:
        logging.warning("Valeurs Intrant_revu non mappées : %s", valeurs_non_mappees)
    logging.info("Répartition des catégories Intrant_revu2 :\n%s", df_intrant['Intrant_revu2'].value_counts())
    return df_intrant




def suppression_duplication_intrant(df_intrant):
    """
    Supprime une duplication spécifique dans df_intrant (zone Sindressou à une date donnée).
    Logge le nombre de lignes après suppression.
    """
    df_intrant = df_intrant[~((df_intrant['Zone_traitement'] == 'Sindressou') & (df_intrant['Date_dutilisation'] == '2024-12-07 00:00:00'))]
    logging.info("Nombre de lignes après la suppression : %s", df_intrant.shape)
    return df_intrant

def creation_jours_entre_2mm_traitements_intrant(df_intrant):
    """
    Ajoute une colonne 'Nb_jours_2mm_tr' qui indique le nombre de jours entre deux traitements successifs
    du même produit, pour chaque zone de traitement. Remplace les valeurs manquantes par zéro.
    """
    df_intrant = df_intrant.sort_values(['Zone_traitement', 'Intrant_revu', 'Date_dutilisation'])
    df_intrant['Nb_jours_2mm_tr'] = df_intrant.groupby(['Zone_traitement', 'Intrant_revu'])['Date_dutilisation'].diff().dt.days
    df_intrant['Nb_jours_2mm_tr'] = df_intrant['Nb_jours_2mm_tr'].fillna(0)
    return df_intrant

import logging

def renommer_colonnes_intrant_securise(df):
    """
    Renomme les colonnes d'intrant de manière sécurisée et idempotente.
    
    Objectif :
    - 'Intrant_revu' (catégories détaillées) -> 'Intrant_revu14'
    - 'Intrant_revu2' (catégories groupées) -> 'Intrant_revu'
    
    Paramètres :
    -----------
    df : pandas.DataFrame
        DataFrame contenant les colonnes à renommer
        
    Retourne :
    ----------
    pandas.DataFrame
        DataFrame avec les colonnes renommées
    """
    # Faire une copie pour éviter de modifier l'original
    df_copy = df.copy()
    colonnes_actuelles = df_copy.columns.tolist()

    logging.info("=== RENOMMAGE SÉCURISÉ DES COLONNES INTRANT ===")
    logging.info(f"Colonnes actuelles : {colonnes_actuelles}")

    # État cible : avoir 'Intrant_revu14' (détaillé) et 'Intrant_revu' (groupé)

    # Cas 1: Les colonnes sont déjà dans l'état souhaité
    if 'Intrant_revu14' in colonnes_actuelles and 'Intrant_revu' in colonnes_actuelles:
        if 'Intrant_revu2' not in colonnes_actuelles:
            logging.info("✓ Les colonnes sont déjà correctement nommées")
            logging.info("  - 'Intrant_revu14' : catégories détaillées")
            logging.info("  - 'Intrant_revu' : catégories groupées")
            return df_copy

    # Cas 2: Configuration initiale (avant premier renommage)
    if 'Intrant_revu' in colonnes_actuelles and 'Intrant_revu2' in colonnes_actuelles and 'Intrant_revu14' not in colonnes_actuelles:
        logging.info("→ Configuration initiale détectée")
        logging.info("  Renommage : 'Intrant_revu' -> 'Intrant_revu14'")
        logging.info("             'Intrant_revu2' -> 'Intrant_revu'")
        
        df_copy = df_copy.rename(columns={
            'Intrant_revu': 'Intrant_revu14',
            'Intrant_revu2': 'Intrant_revu'
        })
        
        logging.info("Renommage effectué avec succès")
    
    # Cas 3: Situation inattendue
    else:
        logging.warning("  Configuration inattendue des colonnes")
        logging.warning(f"   Colonnes trouvées : {colonnes_actuelles}")
        logging.warning("   Aucun renommage effectué par sécurité")
        
        # Afficher l'aide pour déboguer
        logging.warning("\n Configurations attendues :")
        logging.warning("   • État initial : ['Intrant_revu', 'Intrant_revu2']")
        logging.warning("   • État final : ['Intrant_revu14', 'Intrant_revu']")

    # Vérification finale et affichage du résultat
    colonnes_finales = df_copy.columns.tolist()
    logging.info(f"\n Colonnes finales : {colonnes_finales}")

    # Vérifier les valeurs uniques dans les colonnes finales
    if 'Intrant_revu14' in df_copy.columns:
        n_unique = df_copy['Intrant_revu14'].nunique()
        logging.info(f"   • 'Intrant_revu14' : {n_unique} catégories uniques")
        
    if 'Intrant_revu' in df_copy.columns:
        n_unique = df_copy['Intrant_revu'].nunique()
        categories = df_copy['Intrant_revu'].unique()[:5]  # Afficher les 5 premières
        logging.info(f"   • 'Intrant_revu' : {n_unique} catégories uniques")
        logging.info(f"     Exemples : {list(categories)}")
    
    return df_copy

def modif_intrant(df_intrant, label_intrant, intrant_revu2):
    """
    Pipeline principal de nettoyage et traitement de la base df_intrant :
    - Ajoute l’année et la semaine,
    - Traite les valeurs manquantes et doublons,
    - Crée les catégories d’intrant par mapping successifs,
    - Renomme les colonnes,
    - Supprime certaines duplications,
    - Calcule le nombre de jours entre traitements pour un même produit.

    Retourne : df_intrant finalisé.
    """
    try:
        df_intrant = extraire_annee_semaine_intrant(df_intrant)
        df_intrant = traitement_colonnes_intrant(df_intrant)
        df_intrant = creation_categorie_intrant(df_intrant, label_intrant)
        df_intrant = second_mapping_intrant(df_intrant, intrant_revu2)
        df_intrant = renommer_colonnes_intrant_securise(df_intrant)
        df_intrant = suppression_duplication_intrant(df_intrant)
        df_intrant = creation_jours_entre_2mm_traitements_intrant(df_intrant)
    except Exception as e:
        logging.error(f"Erreur lors de modif_intrant: {e}")
    return df_intrant


## Partie aggregation et nettoyage de la base agreger

def aggregation_intrant(df_intrant):
    """
    Agrège la base intrant au niveau hebdomadaire.
    Trie puis agrège par Zone_traitement, Annee, Semaine, Intrant_revu, Unite.
    """
    logging.info("="*25 + " Agrégation " + "="*25)
    df_intrant = df_intrant.sort_values(by=['Zone_traitement','Date_dutilisation', 'Intrant_revu'])
    df_intrant = df_intrant.groupby(
        ["Zone_traitement", "Annee", "Semaine", "Intrant_revu", "Unite"]
    ).agg({
        "Quantite": "sum",
        "Quantite_huile": "sum",
        "Nb_jours_entre_2_traitements": "first",
        "Nb_jours_2mm_tr": "first"
    }).reset_index()
    return df_intrant

def assign_category(quantite):
    """
    Attribue une catégorie en fonction de la quantité d'huile.
    """
    if 0 <= quantite <= 2:
        return "categorie1"
    elif 2 < quantite <= 6:
        return "categorie2"
    else:  # quantite > 6
        return "categorie3"

def create_categorie_column(df_intrant):
    """
    Crée la colonne catégorie en concaténant Intrant_revu et la catégorie d'huile.
    Supprime la colonne temporaire.
    """
    df_intrant['temp_categorie'] = df_intrant['Quantite_huile'].apply(assign_category)
    df_intrant['Categorie'] = df_intrant['Intrant_revu'] + "_" + df_intrant['temp_categorie']
    df_intrant = df_intrant.drop('temp_categorie', axis=1)
    return df_intrant

def filter_zones_intrant(df_intrant, df_cercos):
    """
    Filtre les zones de traitement non désirées et
    conserve uniquement celles présentes dans la base cercos.
    """
    df_intrant = df_intrant[~df_intrant['Zone_traitement'].isin([
        'BADEMA PLANTAIN', 'Beoumi', 'BADEMA', 'NIEKY_SUD', 'NIEKY_CENTRE', 'NIEKY_AVERTISSEMENT', 'NIEKY_NORD'
    ])]
    df_intrant = df_intrant[df_intrant['Zone_traitement'].isin(df_cercos['Zone_traitement'])]
    return df_intrant

def traitement_doublons_quantite_huile(df_intrant):
    """
    Remplace la quantité par 0 pour les lignes où Intrant_revu est 'Huile' et Quantite est 12.00.
    Log le nombre de lignes modifiées.
    """
    logging.info("="*25 + " Traitement des doublons de la quantité d'huile qui se retrouve dans la colonne quantité d'intrant " + "="*25)
    maskh = (df_intrant['Intrant_revu'] == 'Huile') & (df_intrant['Quantite'] == 12.00)
    lignes_modifiees = maskh.sum()
    df_intrant.loc[maskh, 'Quantite'] = 0
    logging.info(f"Nombre de lignes modifiées : {lignes_modifiees}")
    return df_intrant

def analyse_doublons_valeurs_manquantes(df_intrant):
    """
    Analyse et log le nombre de doublons et de valeurs manquantes.
    Supprime les doublons.
    """
    logging.info("="*25 + " Analyse des doublons et des  valeurs manquantes " + "="*25)
    nb_doublons_avant = df_intrant.duplicated().sum()
    df_intrant.drop_duplicates(inplace=True)
    nb_doublons_apres = df_intrant.duplicated().sum()
    logging.info(f"Nombre de doublons après suppression : {nb_doublons_apres}")
    logging.info(f"Nombre de valeurs manquantes :\n{df_intrant.isna().sum()}")
    return df_intrant

def aggregation_and_clean_intrant(df_intrant, df_cercos):
    """
    Regroupe toutes les étapes d'agrégation et de nettoyage de la base intrant.
    """
    try:
        df_intrant = aggregation_intrant(df_intrant)
        df_intrant = create_categorie_column(df_intrant)
        df_intrant = filter_zones_intrant(df_intrant, df_cercos)
        df_intrant = traitement_doublons_quantite_huile(df_intrant)
        df_intrant = analyse_doublons_valeurs_manquantes(df_intrant)
    except Exception as e:
        logging.error(f"Erreur lors de l'agrégation et du nettoyage de la base intrant : {e}")
    return df_intrant


## travail de la base intrant pour la jointure 

def mise_a_niveau_de_la_base_intrant(df_cercos, df_intrant, df_liaison):
    """
    Complète le dataframe intrant pour qu'il contienne toutes les combinaisons 
    (Année, Semaine, Zone_traitement) présentes dans df_cercos.
    
    Parameters:
    -----------
    df_cercos : DataFrame
        Dataframe de référence contenant toutes les combinaisons nécessaires
    df_intrant : DataFrame
        Dataframe à compléter
    df_liaison : DataFrame
        Dataframe contenant les informations de liaison entre zones et stations
        
    Returns:
    --------
    DataFrame
        Dataframe intrant complété
    """
    logging.info("="*150)
    logging.info('Vérification du traitement de la base intrant %s', df_intrant.shape)
    logging.info('Vérification du traitement de la base intrant %s', df_intrant.info())

    # On identifie les combinaisons existantes dans df_cercos
    combinaisons_cercos = df_cercos[['Annee', 'Semaine', 'Zone_traitement']].drop_duplicates()

    # On Identifie les combinaisons existantes dans df_intrant
    combinaisons_intrant = df_intrant[['Annee', 'Semaine', 'Zone_traitement']].drop_duplicates()

    # Étape 3 : Trouver les combinaisons manquantes
    combinaisons_manquantes = combinaisons_cercos.merge(
        combinaisons_intrant, 
        on=['Annee', 'Semaine', 'Zone_traitement'], 
        how='left', 
        indicator=True
    ).query('_merge == "left_only"').drop(columns=['_merge']) 
    # on garde que les lignes qui sont présente dans le df_cercos mais pas dans intrant

    # On ajoute les lignes manquantes à df_intrant
    # Remplir les colonnes avec des valeurs par défaut
    lignes_manquantes = combinaisons_manquantes.copy()
    lignes_manquantes['Quantite'] = 0  
    lignes_manquantes['Quantite_huile'] = 0  
    lignes_manquantes['Intrant_revu'] = 'Aucun traitement'  
    lignes_manquantes['Unite'] = 'Aucun traitement'  
    lignes_manquantes['Nb_jours_entre_2_traitements'] = 0 
    lignes_manquantes['Nb_jours_2mm_tr'] = 0 
    lignes_manquantes['Categorie'] = 'Aucun traitement'

    # On ajoute les lignes manquantes à df_intrant
    df_intrant_complet = pd.concat([df_intrant, lignes_manquantes], ignore_index=True)

    # Vérification
    logging.info(f"Lignes ajoutées : {len(lignes_manquantes)}")
    logging.info(f"Taille finale de df_intrant_complet : {df_intrant_complet.shape}")

    #Créer un dictionnaire de correspondance entre Post_observation et Zone_traitement
    #liaison_dict = dict(zip(df_cercos['Post_observation'], df_cercos['Zone_traitement']))

    # Ajouter la Zone_traitement manquante pour les Post_observation
    #df_intrant_complet['Zone_traitement'] = df_intrant_complet['Post_observation'].map(liaison_dict)

    # Remplir les valeurs manquantes en fonction de la Zone_traitement
    # Créer un dictionnaire pour chaque colonne à remplir
    #station_meteo_dict = dict(zip(df_liaison['Zone_traitement'], df_liaison['Station_meteo']))

    # Remplir la colonne Station_meteo dans df_intrant_complet
    #df_intrant_complet['Station_meteo'] = df_intrant_complet['Zone_traitement'].map(station_meteo_dict)

    # Vérifier les valeurs manquantes restantes
    logging.info(df_intrant_complet.isna().sum())
    
    # Initialiser une variable pour suivre si des périodes manquantes existent
    manquantes_trouvees = False

    # Vérifier les périodes manquantes par zone de traitement
    for zone in df_cercos['Zone_traitement'].unique():
        cercos_periodes = df_cercos[df_cercos['Zone_traitement'] == zone][['Annee', 'Semaine']].drop_duplicates()
        intrant_periodes = df_intrant_complet[df_intrant_complet['Zone_traitement'] == zone][['Annee', 'Semaine']].drop_duplicates()
        
        manquantes = cercos_periodes.merge(
            intrant_periodes, 
            on=['Annee', 'Semaine'], 
            how='left', 
            indicator=True
        ).query('_merge == "left_only"')
        
        if not manquantes.empty:
            manquantes_trouvees = True
            logging.warning(f"Périodes manquantes pour la zone {zone} :")
            logging.warning(manquantes)

    # Afficher le message si aucune période manquante n'a été trouvée
    if not manquantes_trouvees:
        logging.info("Toutes les périodes sont présentes pour toutes les zones.")

    # Vérifier les années et semaines globales
    annees_cercos = df_cercos['Annee'].unique()
    annees_intrant = df_intrant_complet['Annee'].unique()

    semaines_cercos = df_cercos['Semaine'].unique()
    semaines_intrant = df_intrant_complet['Semaine'].unique()

    # Vérification des années
    annees_manquantes = set(annees_cercos) - set(annees_intrant)
    if not annees_manquantes:
        logging.info("Toutes les années de df_cercos sont présentes dans df_intrant_complet.")
    else:
        logging.warning(f"Années manquantes : {annees_manquantes}")

    # Vérification des semaines
    semaines_manquantes = set(semaines_cercos) - set(semaines_intrant)
    if not semaines_manquantes:
        logging.info("Toutes les semaines de df_cercos sont présentes dans df_intrant_complet.")
    else:
        logging.warning(f"Semaines manquantes : {semaines_manquantes}")
    
    return df_intrant_complet

def finalise_intrant_complet(df_intrant, df_cercos, df_liaison):
    """
    Applique la mise à niveau, le tri et le reset d'index sur la base intrant.
    Loggue les informations importantes pour contrôle.
    """
    logging.info("="*150)
    logging.info("="*25 + " Re travail de la base intrant pour la jointure " + "="*25)
    try:
        df_intrant_complet = mise_a_niveau_de_la_base_intrant(df_cercos, df_intrant, df_liaison)
        df_intrant_complet = df_intrant_complet.reset_index(drop=True)
        df_intrant_complet = df_intrant_complet.sort_values(by=['Annee', 'Semaine', 'Zone_traitement'])
    except Exception as e:
        logging.error(f"Erreur lors de la finalisation de la base intrant : {e}")

    logging.info("Info de la base intrant\n%s", df_intrant_complet.info())
    return df_intrant_complet

def traitement_intrant(df_intrant, label_intrant, intrant_revu2, df_cercos, df_liaison):
    """
    Pipeline global pour le traitement complet de la base intrant :
    - Modifie et nettoie la base intrant,
    - Procède à l'agrégation et au nettoyage avancé,
    - Applique la finalisation (mise à niveau, tri, reset index),
    - Log l'info finale de la base.

    Retourne :
    ----------
    DataFrame
        DataFrame intrant complètement traité et finalisé.
    """
    df_intrant = modif_intrant(df_intrant, label_intrant, intrant_revu2)
    df_intrant = aggregation_and_clean_intrant(df_intrant, df_cercos)
    df_intrant_complet = finalise_intrant_complet(df_intrant, df_cercos, df_liaison)
    logging.info("Info de la base intrant\n%s", df_intrant_complet.info())
    return df_intrant_complet