import pandas as pd
import numpy as np
from .analyse_fonction import  *
from .plot_fonction import *
from .preparation_base import *

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_all_data_cercos(path_data_brute, path_data_traite):
    """
    Charge les principales bases de données nécessaires à l'analyse cercos.
    Retourne : df_cercos, df_liaison, df_coord
    """
    df_cercos = load_data(path=path_data_brute + "/base_cerco.xlsx", database_name="cercos")
    df_liaison = load_data(path_data_traite + "/df_liaison.xlsx", database_name="liason")
    df_coord = load_data(path_data_brute + "/base_coord_post.pickle", database_name="coordonnees")
    return df_cercos, df_liaison, df_coord

def drop_missing_post_and_convert(df):
    """
    Cette fonction à pour but de supprimer les lignes où Post_observation est manquant lors du chargement de la base brute.
    Elle permet également de convertir les colonnes Annee et Semaine en types Int64.
    """
    df = df.dropna(subset=['Post_observation']).copy()
    logging.info(f"Nouvelle forme du DataFrame : {df.shape}")
    # Convertir les colonnes Annee et Semaine en types Int64
    try:
        df['Annee'] = df['Annee'].astype('Int64')
        df['Semaine'] = df['Semaine'].astype('Int64')
    except Exception as e:
        logging.error(f"Erreur de conversion des colonnes Annee/Semaine : {e}")
    return df

def rename_e3_broukro(df):
    """
    Renome le poste d'observation E3 en E3_broukro pour la zone Broukro, car deux postes de deux zones différentes portent le même nom.
    """

    mask_e3_broukro = (df['Post_observation'] == 'E3') & (df['Zone_traitement'] == 'Broukro')
    df.loc[mask_e3_broukro, 'Post_observation'] = 'E3_broukro'
    return df

def remove_bad_zones(df):
    """
    Supprime les zones de traitement qui ne sont pas pertinentes pour l'analyse. Nous supprimons notamment les zones relatives au plantin
     avec les zones qui ont un problème de remonté de données.
     """
    bad_zones = [
        'BADEMA PLANTAIN', 'Beoumi', 'BADEMA', 'NIEKY_SUD', 
        'NIEKY_CENTRE', 'NIEKY_AVERTISSEMENT', 'NIEKY_NORD'
    ]
    df = df[~df['Zone_traitement'].isin(bad_zones)]
    return df

def apply_special_value_correction(df):
    """
    Applique des corrections de valeurs spéciales pour certaines observations.
    """
    df.loc[
        (df['Annee'] == 2023) & 
        (df['Semaine'] == 13) & 
        (df['Zone_traitement'] == 'Broukro') & 
        (df['Post_observation'] == 'E7') & 
        (df['Pjfn_moyen'] == 22.6), 
        'Pjfn_moyen'
    ] = 16
    return df

def clean_indicator_ranges(df, indicators):
    """
    Nettoie les indicateurs en remplaçant les valeurs hors plage qui sont donc considéré comme étant abérrante par des NaN.
    """
    for col, (min_val, max_val) in indicators.items():
        try:
            if col in df.columns:
                df.loc[(df[col] < min_val) | (df[col] > max_val), col] = np.nan
        except Exception as e:
            logging.error(f"Erreur lors du nettoyage de la colonne {col} dans la fonction clean_indicator_ranges: {e}")
    return df



def clean_cercos_data1(df_cercos, df_coord):
    """
    Nettoie et prépare la base cercos. Cette fonction utilise plusieurs fonctions pour effectuer les étapes de nettoyage des zones de traitement avec des problèmes de remonté de donnée, 
    des valeurs aberrantes, de renommage des colonnes. Nous donnons en input notre base brute afin quelle nettoyée. 
    """
    try:
        df_cercos = drop_missing_post_and_convert(df_cercos)
        df_cercos = rename_e3_broukro(df_cercos)
        df_cercos = remove_bad_zones(df_cercos)
        df_cercos = apply_special_value_correction(df_cercos)
        # Nettoyage généralisé des indicateurs
        indicators_ranges = {
            'Nff_moyen': (6, 16),
            'Nfr_moyen': (0, 13),
            'Etat_devolution_moy': (0, 4000),
            'Pjfn_moyen': (6, 17),
            'Pjft_moyen': (2, 11),
            'Dp_moy': (0, 20)
        }
        df_cercos = clean_indicator_ranges(df_cercos, indicators_ranges)
    except Exception as e:
        logging.error(f"Erreur dans clean_cercos_data1 : {e}")
    return df_cercos

#### Fonction pour detecter et ajouter les semaines manquantes 


def create_theoretical_weeks(annees, semaines_par_annee):
    """
    L'objectif de cette fonction est la création d'un DataFrame contenant toutes les semaines théoriques pour chaque année afin de détecter par la suite
    les occurences manquantes au niveau de tout les postes de la base de données.
    """
    try: 
        toutes_semaines = []
        for annee in annees:
            for semaine in range(1, semaines_par_annee[annee] + 1):
                toutes_semaines.append((annee, semaine))
        df_theorique = pd.DataFrame(toutes_semaines, columns=['Annee', 'Semaine'])
    
    except Exception as e:
        logging.error(f"Erreur dans create_theoretical_weeks : {e}")
    return df_theorique

def find_missing_weeks(df_cercos, df_theorique):
    """
    Cette fonction nous permet de joindre notre base de données avec la base de données que nous avons crée avec l'ensemble des occurences présentes pour l'ensemble des postes.
    Une fois la jointure faite, nous pouvons détecter les semaines manquantes pour chaque poste d'observation en vue de les ajouter par la suite.
    """
    try: 
        postes_zones = df_cercos[['Post_observation', 'Zone_traitement']].drop_duplicates()
    except Exception as e:
        logging.error(f"Colonnes manquantes dans df_cercos: {e}")
    
    resultats_semaines_manquantes = []
    try:
        for _, row in postes_zones.iterrows():
            poste = row['Post_observation']
            zone = row['Zone_traitement']
            semaines_presentes = df_cercos[df_cercos['Post_observation'] == poste][['Annee', 'Semaine']].drop_duplicates()
            semaines_manquantes = df_theorique.merge(
                semaines_presentes, 
                on=['Annee', 'Semaine'], 
                how='left', 
                indicator=True
            ).query('_merge == "left_only"')[['Annee', 'Semaine']]
            for _, semaine_row in semaines_manquantes.iterrows():
                resultats_semaines_manquantes.append({
                    'Post_observation': poste,
                    'Zone_traitement': zone,
                    'Annee': semaine_row['Annee'],
                    'Semaine': semaine_row['Semaine'],
                    'Year_Week': f"{semaine_row['Annee']}_S{semaine_row['Semaine']:02d}"
                })
    except Exception as e:
        logging.error(f"Erreur lors de la détection des semaines manquantes: {e}")

    df_semaines_manquantes = pd.DataFrame(resultats_semaines_manquantes)
    df_semaines_manquantes = df_semaines_manquantes.sort_values(['Post_observation', 'Annee', 'Semaine'])
    
    if df_semaines_manquantes.empty:
        logging.info("Aucune semaine manquante détectée.")
    return df_semaines_manquantes


def analyser_semaines_manquantes_par_poste1(df_cercos, afficher_details=False, afficher_resume=False):
    """
    Analyse détaillée des semaines manquantes par poste d'observation. Cette fonction nous permet d'avoir l'information sur les semaines manquantes pour chaque 
    poste dans un tableau résumé.Grâce à cette fonction, on va savoir exactement quelle semaine ajouter sur quelle poste, elle nous permet également de voir 
    si il y'a un problème de remonté trop important sur certains postes.

    Args:
        df_cercos: DataFrame contenant les données cercos
        afficher_details: bool, afficher le tableau détaillé des semaines manquantes
        afficher_resume: bool, afficher le résumé par poste d'observation
    
    Returns:
        tuple: (df_semaines_manquantes, resume_par_poste)
    """
    logging.info("=== TABLEAU DÉTAILLÉ DES SEMAINES MANQUANTES PAR POSTE ===")
    
    annees = sorted(df_cercos['Annee'].unique())
    annee_max = max(annees)

    try:
        semaines_max_annee_max = df_cercos[df_cercos['Annee'] == annee_max]['Semaine'].max() if annee_max in df_cercos['Annee'].values else 0
        nb_semaines_theorique = 0
        semaines_par_annee = {}
        for annee in annees:
            if annee == annee_max:
                semaines_par_annee[annee] = semaines_max_annee_max
            else:
                semaines_par_annee[annee] = 52
            nb_semaines_theorique += semaines_par_annee[annee]
    except Exception as e:
        logging.error(f"Erreur lors du calcul du nombre de semaines théoriques: {e}")

    logging.info(f"Nombre théorique de semaines: {nb_semaines_theorique}")

    for annee in annees:
        detail = " (détectées automatiquement)" if annee == annee_max else ""
        logging.info(f"  - {annee}: {semaines_par_annee[annee]} semaines{detail}")

    try: 
        df_theorique = create_theoretical_weeks(annees, semaines_par_annee)
        df_semaines_manquantes = find_missing_weeks(df_cercos, df_theorique)
    except Exception as e:
        logging.error(f"Erreur lors du calcul du nombre de semaines manquantes: {e}")

    logging.info(f"Nombre total de semaines manquantes: {len(df_semaines_manquantes)}")
    logging.info(f"Nombre de postes concernés: {df_semaines_manquantes['Post_observation'].nunique()}")

    if afficher_details and not df_semaines_manquantes.empty:
        logging.info("Tableau des semaines manquantes par poste d'observation:")
        logging.info("\n" + df_semaines_manquantes.to_string(index=False))

    if not df_semaines_manquantes.empty:
        resume_par_poste = df_semaines_manquantes.groupby(['Post_observation', 'Zone_traitement']).size().reset_index(name='Nb_semaines_manquantes')
        resume_par_poste['Pourcentage_completude'] = ((nb_semaines_theorique - resume_par_poste['Nb_semaines_manquantes']) / nb_semaines_theorique * 100).round(2)
        resume_par_poste = resume_par_poste.sort_values('Nb_semaines_manquantes', ascending=False)
        if afficher_resume:
            logging.info("=== RÉSUMÉ PAR POSTE D'OBSERVATION ===")
            logging.info("\n" + resume_par_poste.to_string(index=False))
    else:
        resume_par_poste = pd.DataFrame()
    
    return df_semaines_manquantes, resume_par_poste

### Ajouter les lignes manquantes 

def build_infos_postes(df_cercos):
    """
    Construit un dictionnaire contenant les informations sur chaque poste d'observation. L'objectif est de récupéré les informations sur postes pour remplir par la suite
    les informations sur les lignes que nous avons ajoutées.
    """
    infos_postes = {}
    for poste in df_cercos['Post_observation'].unique():
        poste_data = df_cercos[df_cercos['Post_observation'] == poste].iloc[0]
        infos_postes[poste] = {
            'Zone_traitement': poste_data['Zone_traitement']
        }
    return infos_postes

def create_nouvelles_lignes(df_semaines_manquantes, infos_postes, indicateurs_cercos):
    """
    Crée un DataFrame avec les nouvelles lignes à ajouter pour les semaines manquantes. Ces lignes sont remplies avec les informations générales que nous connaissons sur les postes
    (Annee, semaine, à quelle zone il appartient).
    """
    nouvelles_lignes = []
    for _, row in df_semaines_manquantes.iterrows():
        poste = row['Post_observation']
        if poste in infos_postes:
            infos_poste = infos_postes[poste]
            nouvelle_ligne = {
                'Annee-semaine': f"{row['Annee']}-{row['Semaine']:02d}",
                'Annee': row['Annee'],
                'Semaine': row['Semaine'],
                'Zone_traitement': infos_poste['Zone_traitement'],
                'Post_observation': poste,
            }
            for indicateur in indicateurs_cercos:
                nouvelle_ligne[indicateur] = np.nan
            nouvelles_lignes.append(nouvelle_ligne)
        else:
            logging.warning(f"Poste {poste} non trouvé dans df_cercos")
    return pd.DataFrame(nouvelles_lignes)




def ajouter_semaines_manquantes1(df_cercos, df_semaines_manquantes, indicateurs_cercos=None):
    """
    Ajoute automatiquement les semaines manquantes au DataFrame original. On ajoute les lignes en ajoutant les informations déjà connues 
    sur ces nouvelles lignes comme la temporalité ou la zone qui correspond au poste. Pour les indicateurs cercos, nous laissons des NaN.
    """
    if indicateurs_cercos is None:
        indicateurs_cercos = ['Nff_moyen', 'Nfr_moyen', 'Pjfn_moyen', 'Pjft_moyen', 'Etat_devolution_moy', 'Dp_moy']
    
    nb_lignes_avant = len(df_cercos)
    logging.info("=== AJOUT AUTOMATIQUE DES SEMAINES MANQUANTES À DF_CERCOS ===")
    logging.info(f"Nombre de lignes dans df_cercos avant ajout : {nb_lignes_avant}")
    
    try:
        infos_postes = build_infos_postes(df_cercos)
    except Exception as e:
        logging.error(f"Erreur lors de build_infos_postes: {e}")
    
    try:
        df_nouvelles_lignes = create_nouvelles_lignes(df_semaines_manquantes, infos_postes, indicateurs_cercos)
    except Exception as e:
        logging.error(f"Erreur lors de create_nouvelles_lignes: {e}")
    try:
        df_cercos_complet = pd.concat([df_cercos, df_nouvelles_lignes], ignore_index=True)
        df_cercos_complet = df_cercos_complet.sort_values(['Post_observation', 'Annee', 'Semaine'])
    except Exception as e:
        logging.error(f"Erreur lors de la concaténation/tri: {e}")
    
    nb_lignes_apres = len(df_cercos_complet)
    nb_lignes_ajoutees = nb_lignes_apres - nb_lignes_avant
    stats = {
        'lignes_avant': nb_lignes_avant,
        'lignes_apres': nb_lignes_apres,
        'lignes_ajoutees': nb_lignes_ajoutees,
        'semaines_manquantes_attendues': len(df_semaines_manquantes)
    }
    
    logging.info(f"Nombre de lignes ajoutées : {nb_lignes_ajoutees}")
    logging.info(f"Nombre de lignes dans df_cercos après ajout : {nb_lignes_apres}")
   
    # if not df_nouvelles_lignes.empty:
    #     logging.info("Exemples de lignes ajoutées (avec NaN pour les indicateurs cercos) :")
    #     lignes_exemples = df_nouvelles_lignes[['Post_observation', 'Zone_traitement', 'Annee', 'Semaine'] + indicateurs_cercos[:2]].head()
    #     logging.info("\n" + lignes_exemples.to_string(index=False))
    
    return df_cercos_complet, df_nouvelles_lignes, stats

# imputer les semaine manquantes que nous avons ajoutées.
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def create_combinaisons_manquantes(df_semaines_manquantes, df_cercos):
    """
    Crée un DataFrame avec les combinaisons manquantes de semaines, zones de traitement et postes d'observation.
    """
    combinaisons_manquantes = df_semaines_manquantes[['Annee', 'Semaine', 'Post_observation']].copy()
    post_to_zone = dict(zip(df_cercos['Post_observation'], df_cercos['Zone_traitement']))
    combinaisons_manquantes['Zone_traitement'] = combinaisons_manquantes['Post_observation'].map(post_to_zone)
    combinaisons_manquantes['id_unique'] = (
        combinaisons_manquantes['Annee'].astype(str) + '_' +
        combinaisons_manquantes['Semaine'].astype(str) + '_' +
        combinaisons_manquantes['Zone_traitement'] + '_' +
        combinaisons_manquantes['Post_observation']
    )
    return combinaisons_manquantes

def add_id_unique(df_cercos):
    """
    Ajoute une colonne 'id_unique' au DataFrame df_cercos pour identifier de manière unique chaque ligne."""
    df_cercos['id_unique'] = (
        df_cercos['Annee'].astype(str) + '_' +
        df_cercos['Semaine'].astype(str) + '_' +
        df_cercos['Zone_traitement'] + '_' +
        df_cercos['Post_observation']
    )
    return df_cercos

def encode_labels(df_cercos):
    """
    Encode les variables catégorielles dans le DataFrame df_cercos afin de préparer l'imputation des variables avec notre algorithme de similitude.
    """
    le_zone = LabelEncoder()
    le_poste = LabelEncoder()
    df_cercos['Zone_traitement_encoded'] = le_zone.fit_transform(df_cercos['Zone_traitement'])
    df_cercos['Post_observation_encoded'] = le_poste.fit_transform(df_cercos['Post_observation'])
    return df_cercos

def imputer_pjft_moyen_ajoutés(df_cercos, mask_lignes_ajoutees, n_neighbors):
    """
    Impute la colonne 'Pjft_moyen' pour les lignes ajoutées en utilisant KNNImputer. Tout nos indicateurs cercos sont corrélées entre eux,
    Cette fonction permet d'imputer dans un premier temps un premier indicateur en se basant sur des variables temporelles (Annee, Semaine)
    et géographiques (Zone_traitement, Post_observation).
    """
    logging.info("=== ÉTAPE 1 : Imputation de Pjft_moyen ===")
    num_cols_etape1 = ['Annee', 'Semaine', 'Zone_traitement_encoded', 'Post_observation_encoded']
    df_for_imputation_etape1 = df_cercos[num_cols_etape1 + ['Pjft_moyen']].copy()
    
    try:
        preprocessor_etape1 = StandardScaler()
        features_scaled = preprocessor_etape1.fit_transform(df_for_imputation_etape1[num_cols_etape1])
    except Exception as e:
        logging.error(f"Erreur lors du scaling des features : {e}")

    try:
        data_for_knn = np.column_stack([features_scaled, df_for_imputation_etape1['Pjft_moyen'].values])
    except Exception as e:
        logging.error(f"Erreur lors de la préparation des données pour KNN : {e}")

    try:
        imputer_etape1 = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        imputed_data_etape1 = imputer_etape1.fit_transform(data_for_knn)
    except Exception as e:
        logging.error(f"Erreur lors de l'imputation KNN étape 1: {e}")

    try:
        df_cercos.loc[mask_lignes_ajoutees, 'Pjft_moyen'] = imputed_data_etape1[mask_lignes_ajoutees, -1]
        logging.info(f"Pjft_moyen imputé pour {mask_lignes_ajoutees.sum()} lignes")
    except Exception as e:
        logging.error(f"Erreur lors de l'affectation des valeurs imputées : {e}")
    return df_cercos

def imputer_autres_indicateurs_ajoutés(df_cercos, mask_lignes_ajoutees, indicateurs_cercos, n_neighbors):
    """
    Impute les autres indicateurs cercos disponibles en utilisant toujours KNNImputer. Cette fois nous prenons les mêmes variables que pour l'étape 1,
    mais nous ajoutons en plus la colonne Pjft_moyen qui nous donne en plus une information cercos.
    """
    logging.info("=== ÉTAPE 2 : Imputation des autres indicateurs cercos ===")
    num_cols_etape2 = ['Annee', 'Semaine', 'Zone_traitement_encoded', 'Post_observation_encoded', 'Pjft_moyen']
    autres_indicateurs = [ind for ind in indicateurs_cercos if ind != 'Pjft_moyen']
    df_for_imputation_etape2 = df_cercos[num_cols_etape2 + autres_indicateurs].copy()
    try:
        preprocessor_etape2 = StandardScaler()
        features_scaled_etape2 = preprocessor_etape2.fit_transform(df_for_imputation_etape2[num_cols_etape2])
    except Exception as e:  
        logging.error(f"Erreur lors du scaling des features : {e}")
    try:   
        targets_data = df_for_imputation_etape2[autres_indicateurs].values
        data_for_knn_etape2 = np.column_stack([features_scaled_etape2, targets_data])
    except Exception as e:
        logging.error(f"Erreur lors de la préparation des données pour KNN étape 2 : {e}")
    try:
        imputer_etape2 = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        imputed_data_etape2 = imputer_etape2.fit_transform(data_for_knn_etape2)
    except Exception as e:
        logging.error(f"Erreur lors de l'imputation KNN étape 2: {e}")

    n_features = len(num_cols_etape2)
    for i, indicateur in enumerate(autres_indicateurs):
        col_idx = n_features + i
        df_cercos.loc[mask_lignes_ajoutees, indicateur] = imputed_data_etape2[mask_lignes_ajoutees, col_idx]
        logging.info(f"{indicateur} imputé pour {mask_lignes_ajoutees.sum()} lignes")
    return df_cercos

def log_verification(df_cercos, mask_lignes_ajoutees, indicateurs_cercos):
    """
    Vérifie les résultats de l'imputation en affichant l'évolution des valeurs manquantes, ainsi que des contrôles de cohérence de cette imputation."""
    logging.info("=== VÉRIFICATION DES RÉSULTATS ===")
    lignes_ajoutees_apres = df_cercos[mask_lignes_ajoutees]
    valeurs_manquantes_apres = lignes_ajoutees_apres[indicateurs_cercos].isna().sum()
    logging.info("Valeurs manquantes après imputation dans les lignes ajoutées :")
    for col in indicateurs_cercos:
        logging.info(f"{col}: {valeurs_manquantes_apres[col]}")
    if valeurs_manquantes_apres.sum() == 0:
        logging.info("Toutes les valeurs manquantes ont été imputées avec succès !")
    else:
        logging.warning("Il reste encore des valeurs manquantes")
    logging.info(f"Taille finale après imputation : {df_cercos.shape}")
    logging.info("=== EXEMPLES DE LIGNES IMPUTÉES ===")
    exemples = lignes_ajoutees_apres[['Post_observation', 'Zone_traitement', 'Annee', 'Semaine'] + indicateurs_cercos].head(10)
    logging.info("\n" + exemples.to_string(index=False))
    logging.info("=== STATISTIQUES DES LIGNES IMPUTÉES ===")
    logging.info(f"Postes d'observation concernés : {lignes_ajoutees_apres['Post_observation'].nunique()}")
    logging.info("Répartition par poste :")
    repartition_postes = lignes_ajoutees_apres['Post_observation'].value_counts().head(10)
    for poste, count in repartition_postes.items():
        logging.info(f"  {poste}: {count} lignes")

def imputer_lignes_ajoutees1(df_cercos, df_semaines_manquantes, indicateurs_cercos=None, n_neighbors=5):
    """
    Impute les indicateurs cercos uniquement pour les lignes correspondant aux semaines manquantes ajoutées.
    """
    logging.info("=== IMPUTATION DES LIGNES AJOUTÉES POUR LES INDICATEURS CERCOS ===")

    if indicateurs_cercos is None:
        indicateurs_cercos = ['Nff_moyen', 'Nfr_moyen', 'Pjfn_moyen', 'Pjft_moyen', 'Etat_devolution_moy', 'Dp_moy']

    combinaisons_manquantes = create_combinaisons_manquantes(df_semaines_manquantes, df_cercos)
    df_cercos = add_id_unique(df_cercos)
    mask_lignes_ajoutees = df_cercos['id_unique'].isin(combinaisons_manquantes['id_unique'])

    logging.info(f"Nombre de lignes à imputer : {mask_lignes_ajoutees.sum()}")
    logging.info(f"Nombre de semaines manquantes identifiées : {len(df_semaines_manquantes)}")
    if mask_lignes_ajoutees.sum() != len(df_semaines_manquantes):
        logging.warning(f"Attention: Différence entre le masque ({mask_lignes_ajoutees.sum()}) et df_semaines_manquantes ({len(df_semaines_manquantes)})")
    else:
        logging.info("Le masque correspond parfaitement aux semaines manquantes")

    df_cercos = encode_labels(df_cercos)
    df_cercos = imputer_pjft_moyen_ajoutés(df_cercos, mask_lignes_ajoutees, n_neighbors)
    df_cercos = imputer_autres_indicateurs_ajoutés(df_cercos, mask_lignes_ajoutees, indicateurs_cercos, n_neighbors)
    log_verification(df_cercos, mask_lignes_ajoutees, indicateurs_cercos)
    df_cercos.drop(columns=['Zone_traitement_encoded', 'Post_observation_encoded', 'id_unique'], inplace=True)

    return df_cercos

from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

## imputer valeurs manquantes dans le dataset de base 
def imputer_pjft_moyen(df_cercos, num_cols, cat_cols):
    """
    Impute la colonne 'Pjft_moyen' en utilisant KNNImputer. Tout nos indicateurs cercos sont corrélées entre eux,
    Cette fonction permet d'imputer dans un premier temps un premier indicateur en se basant sur des variables temporelles (Annee, Semaine)
    et géographiques (Zone_traitement, Post_observation).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ],
        remainder='passthrough'
    )
    features_df = df_cercos[num_cols + cat_cols + ['Pjft_moyen']].copy()
    transformed_data = preprocessor.fit_transform(features_df)
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    imputed_data = imputer.fit_transform(transformed_data)
    pjft_idx = transformed_data.shape[1] - 1
    mask_missing_pj = df_cercos['Pjft_moyen'].isna()
    df_cercos.loc[mask_missing_pj, 'Pjft_moyen'] = imputed_data[mask_missing_pj, pjft_idx]
    logging.info(f"Nombre de valeurs imputées pour le PJFT : {mask_missing_pj.sum()}")
    return df_cercos

def imputer_autres_indicateurs(df_cercos, num_cols, cat_cols, indicator_cols):
    """
    Impute les autres indicateurs cercos disponibles en utilisant toujours KNNImputer. Cette fois nous prenons les mêmes variables que pour l'étape 1,
    mais nous ajoutons en plus la colonne Pjft_moyen qui nous donne en plus une information cercos.
    """
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    for indicator in indicator_cols:
        num_cols_step2 = num_cols + ['Pjft_moyen']
        imputation_df = df_cercos[num_cols_step2 + cat_cols + [indicator]].copy()
        preprocessor_step2 = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols_step2),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
            ],
            remainder='passthrough'
        )
        transformed_data = preprocessor_step2.fit_transform(imputation_df)
        imputed_data = imputer.fit_transform(transformed_data)
        indicator_idx = transformed_data.shape[1] - 1
        mask_missing_indicator = df_cercos[indicator].isna()
        df_cercos.loc[mask_missing_indicator, indicator] = imputed_data[mask_missing_indicator, indicator_idx]
        logging.info(f"Nombre de valeurs imputées pour {indicator}: {mask_missing_indicator.sum()}")
    return df_cercos


def imputer_na_cercos1(df_cercos):
    """
    Impute les valeurs manquantes dans le dataframe cercos en utilisant KNN en deux étapes :
    1. Imputation de PJFT_moyen
    2. Imputation des autres indicateurs
    """
    df_cercos = df_cercos.copy()
    num_cols = ['Annee', 'Semaine']
    cat_cols = ['Post_observation', 'Zone_traitement']
    indicator_cols = ['Pjfn_moyen', 'Etat_devolution_moy', 'Nff_moyen', 'Nfr_moyen', 'Dp_moy']

    try:
        df_cercos = imputer_pjft_moyen(df_cercos, num_cols, cat_cols)
        df_cercos = imputer_autres_indicateurs(df_cercos, num_cols, cat_cols, indicator_cols)

    except Exception as e:
        logging.error(f"Erreur lors de l'imputation des valeurs manquantes originales : {e}")

    missing_after = df_cercos[['Pjft_moyen'] + indicator_cols].isna().sum()
    logging.info("Valeurs manquantes restantes après imputation:")
    logging.info(f"\n{missing_after}")


    return df_cercos


def process_and_impute_cercos1(df_cercos):
    """
    Trie, analyse les semaines manquantes, ajoute les lignes manquantes, impute les valeurs manquantes,
    effectue la visualisation et affiche des infos sur la base finale.
    """

    # Analyse des lignes manquantes pour chaque poste d'observation (semaines manquantes)
    try:
        df_semaines_manquantes, resume_par_poste = analyser_semaines_manquantes_par_poste1(df_cercos)
    
    except Exception as e:
        logging.error(f"Erreur lors de l'analyse des semaines manquantes : {e}")

    # Ajout des Semaines manquantes pour chaque poste d'observation
    try:
        df_cercos_complet, df_nouvelles_lignes, stats = ajouter_semaines_manquantes1(df_cercos, df_semaines_manquantes)
        df_cercos = df_cercos_complet
    except Exception as e:                          
        logging.error(f"Erreur lors de l'ajout des semaines manquantes : {e}")

    # Imputation des valeurs manquantes seulement pour les lignes que nous avons ajoutées
    try:
        df_cercos = imputer_lignes_ajoutees1(df_cercos, df_semaines_manquantes, n_neighbors=5)
    except Exception as e:
        logging.error(f"Erreur lors de l'imputation des lignes ajoutées : {e}")

    # Imputation des valeurs manquantes pour tous les indicateurs cercos
    df_cercos = df_cercos.sort_values(by=['Annee', 'Semaine', 'Post_observation'])
    try:
        df_cercos = imputer_na_cercos1(df_cercos)
    except Exception as e:
        logging.error(f"Erreur lors de l'imputation des valeurs manquantes : {e}")

    if 'Annee-semaine' in df_cercos.columns:
        df_cercos = df_cercos.drop(columns=['Annee-semaine'])

    return df_cercos


def filter_outliers_boxplot(group, ind):
    """
    Cette fonction filtre les valeurs aberrantes dans chaque groupe (qui représentent les zones de traitement) en utilisant la méthode du boxplot.
    """
    q1 = group[ind].quantile(0.25)
    q3 = group[ind].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = group[(group[ind] >= lower) & (group[ind] <= upper) | (group[ind].isna())]
    logging.debug(f"Filtering outliers for {ind}: Q1={q1}, Q3={q3}, IQR={iqr}, lower={lower}, upper={upper}, kept={len(filtered)}, orig={len(group)}")
    return filtered

def clean_and_aggregate_cercos1(df_cercos, indicateurs=None):
    """
    Nettoie les valeurs aberrantes par la méthode du boxplot pour chaque zone/semaine/année/indicateur,
    puis agrège les indicateurs par la moyenne au niveau de la zone.
    """
    key_cols = ["Annee", "Semaine", "Zone_traitement"]
    poste_col = "Post_observation"
    
    if indicateurs is None:
        indicateurs = [
            col for col in df_cercos.select_dtypes("number").columns
            if col not in key_cols + [poste_col]
        ]
        logging.info(f"Indicateurs détectés automatiquement : {indicateurs}")
        
    df = df_cercos.copy()
    
    for ind in indicateurs:
        logging.info(f"Début du filtrage des valeurs aberrantes pour l'indicateur : {ind}")
        try:
            df = (
                df.groupby(key_cols, group_keys=False)
                .apply(lambda group: filter_outliers_boxplot(group, ind))
            )
            logging.info(f"Fin du filtrage pour {ind}, nombre de lignes : {len(df)}")
        except Exception as e:
            logging.error(f"Erreur lors du filtrage des valeurs aberrantes pour {ind} : {e}")

    df_clean = (
        df.groupby(key_cols, as_index=False)[indicateurs]
        .mean()
    )
    logging.info(f"Dataframe agrégé, forme finale : {df_clean.shape}")
    
    return df_clean



def traitement_cercos(df_cercos, df_coord):
    """
    Pipeline complet pour le traitement et nettoyage de la base de données cercos :
    - Nettoie la base cercos,
    - Pré-traite avec ajout des semaines manquantes et imputation des NA,
    - Effectue le nettoyage final et l’agrégation,
    - Log l’info finale de la base.

    Retourne :
    ----------
    DataFrame
        DataFrame cercos complètement traité et finalisé.
    """
    df_cercos = clean_cercos_data1(df_cercos, df_coord)
    df_cercos = process_and_impute_cercos1(df_cercos)
    df_cercos = clean_and_aggregate_cercos1(df_cercos)
    logging.info("Info de la base cercos\n%s", df_cercos.info())
    return df_cercos