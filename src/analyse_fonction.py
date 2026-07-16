import pandas as pd
from itertools import product
import pandas as pd
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import seaborn as sns
from src.preparation_base import *
from src.plot_fonction import *
import warnings
warnings.filterwarnings("ignore")
import missingno as msno 
import sidetable as stb
import missingno as msno
from scipy.stats import shapiro

from plotly.subplots import make_subplots
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

###################################### Partie Météo ######################################                      

def doublons_cle(df, colonnes_cle, afficher_details=True, trier=True, afficher_tableau=True):
    """
    Identifie et analyse les doublons dans un DataFrame selon des colonnes spécifiées.
    
    Parameters:
    -----------
    df : DataFrame
        Le DataFrame à analyser
    colonnes_cle : list
        Liste des colonnes qui définissent l'unicité (clés de dédoublonnage)
    afficher_details : bool, default=True
        Si True, affiche les statistiques détaillées
    trier : bool, default=True
        Si True, trie le résultat par les colonnes clés
    afficher_tableau : bool, default=True
        Si True, affiche le tableau des doublons trouvés
        
    Returns:
    --------
    DataFrame
        DataFrame contenant uniquement les lignes dupliquées
        
    Example:
    --------
    >>> # Chercher les doublons par station et date
    >>> doublons = doublons_cle(df_meteo, ['Station_meteo', 'Date'])
    
    >>> # Chercher les doublons par année, semaine et poste
    >>> doublons = doublons_cle(df_meteo, ['Annee', 'Semaine', 'Post_observation'])
    
    >>> # Sans affichage des détails
    >>> doublons = doublons_cle(df_meteo, ['Station_meteo'], afficher_details=False)
    
    >>> # Juste le nombre, sans le tableau
    >>> doublons = doublons_cle(df_meteo, ['Station_meteo'], afficher_tableau=False)
    """
    
    # Identifier les doublons selon les colonnes spécifiées
    doublons_df = df[df.duplicated(subset=colonnes_cle, keep=False)]
    
    if afficher_details:
        # Créer le nom des clés pour l'affichage
        nom_cles = " + ".join(colonnes_cle)
        
        print("=" * 60)
        print(f" ANALYSE DES DOUBLONS SUR : {nom_cles}")
        print("=" * 60)
        print(f" Nombre total de lignes dans la base : {len(df)}")
        print(f" Nombre de lignes dupliquées trouvées : {len(doublons_df)}")
        
        if len(doublons_df) == 0:
            print(" Aucun doublon trouvé !")
        
        print("=" * 60)
    
    # Trier le résultat si demandé
    if trier and len(doublons_df) > 0:
        doublons_df = doublons_df.sort_values(by=colonnes_cle)
    
    # Afficher le tableau si demandé
    if afficher_tableau and len(doublons_df) > 0:
        print(f"\n TABLEAU DES DOUBLONS :")
        print(doublons_df)
    
    return doublons_df

    
def concatener_meteo_annees(meteo_2025, meteo_2024, meteo_2023):
    """
    Concatène les données météo de trois années dans l'ordre chronologique inverse.
    
    Parameters:
    -----------
    meteo_2023 : DataFrame
        Données météo de l'année 2023
    meteo_2024 : DataFrame  
        Données météo de l'année 2024
    meteo_2025 : DataFrame
        Données météo de l'année 2025
        
    Returns:
    --------
    DataFrame
        DataFrame concaténé avec 2025 en premier, puis 2024, puis 2023
        
    Example:
    --------
    >>> meteo_complet = concatener_meteo_annees(meteo23, meteo24, meteo25)
    >>> print(f"Taille finale : {meteo_complet.shape}")
    """
    
    # Concaténer dans l'ordre : 2025, 2024, 2023 avec reset de l'index
    df_meteo_concat = pd.concat([meteo_2025, meteo_2024, meteo_2023], 
                                ignore_index=True)
    
    # Afficher des informations sur la concaténation
    print("=== DÉTAIL DE LA CONCATÉNATION ===")
    print(f" Données 2025 : {meteo_2025.shape[0]} lignes")
    print(f" Données 2024 : {meteo_2024.shape[0]} lignes") 
    print(f" Données 2023 : {meteo_2023.shape[0]} lignes")
    print(f" Total final : {df_meteo_concat.shape[0]} lignes")
    print(f" Nombre de colonnes : {df_meteo_concat.shape[1]}")
    
    return df_meteo_concat

def station_complete(df, stations):
    """
    Compare les stations météorologiques en termes de valeurs manquantes, 
    couverture temporelle et distributions des indicateurs.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données météorologiques
    stations : list
        Liste des stations à comparer
    
    Returns:
    --------
    dict : Dictionnaire contenant 3 analyses :
        - 'valeurs_manquantes' : Proportion de valeurs manquantes par station et colonne
        - 'couverture_temporelle' : Période couverte et nombre d'observations par station
        - 'distributions' : Statistiques descriptives des indicateurs météo par station
    """
    
    # Filtrer le dataframe pour ne garder que les stations d'intérêt
    df_filtered = df[df['Station_meteo'].isin(stations)].copy()
    
    if df_filtered.empty:
        print("Aucune station trouvée dans les données")
        return {}
    
    # =============================================================================
    # 1. ANALYSE DES VALEURS MANQUANTES
    # =============================================================================
    
    # Colonnes météorologiques à analyser
    colonnes_meteo = ['Tmax', 'Tmin', 'Tmoy', 'Pluie', 'Humidite_min', 'Humidite_max']
    
    # Calculer les proportions de valeurs manquantes par station
    resultats_missing = []
    
    for station in stations:
        station_data = df_filtered[df_filtered['Station_meteo'] == station]
        
        if not station_data.empty:
            # Calculer les statistiques de valeurs manquantes
            missing_stats = station_data[colonnes_meteo].isnull().mean() * 100
            
            for colonne in colonnes_meteo:
                resultats_missing.append({
                    'Station_meteo': station,
                    'Colonne': colonne,
                    'Proportion_manquantes_pct': round(missing_stats[colonne], 2),
                    'Nb_valeurs_manquantes': station_data[colonne].isnull().sum(),
                    'Total_observations': len(station_data)
                })
    
    df_valeurs_manquantes = pd.DataFrame(resultats_missing)
    
    # Pivot pour avoir les stations en colonnes
    if not df_valeurs_manquantes.empty:
        pivot_missing = df_valeurs_manquantes.pivot(
            index='Colonne', 
            columns='Station_meteo', 
            values='Proportion_manquantes_pct'
        ).fillna(0)
    else:
        pivot_missing = pd.DataFrame()
    
    # =============================================================================
    # 2. ANALYSE DE LA COUVERTURE TEMPORELLE
    # =============================================================================
    
    resultats_temporels = []
    
    for station in stations:
        station_data = df_filtered[df_filtered['Station_meteo'] == station]
        
        if not station_data.empty:
            # Convertir Date en datetime si ce n'est pas déjà fait
            station_data['Date'] = pd.to_datetime(station_data['Date'])
            
            date_min = station_data['Date'].min()
            date_max = station_data['Date'].max()
            nb_jours_periode = (date_max - date_min).days + 1
            nb_observations = len(station_data)
            
            # Calculer le taux de couverture
            taux_couverture = (nb_observations / nb_jours_periode) * 100 if nb_jours_periode > 0 else 0
            
            resultats_temporels.append({
                'Station_meteo': station,
                'Date_debut': date_min.strftime('%Y-%m-%d'),
                'Date_fin': date_max.strftime('%Y-%m-%d'),
                'Periode_jours': nb_jours_periode,
                'Nb_observations': nb_observations,
                'Taux_couverture_pct': round(taux_couverture, 2),
                'Jours_manques': nb_jours_periode - nb_observations
            })
        else:
            resultats_temporels.append({
                'Station_meteo': station,
                'Date_debut': 'Aucune donnée',
                'Date_fin': 'Aucune donnée',
                'Periode_jours': 0,
                'Nb_observations': 0,
                'Taux_couverture_pct': 0,
                'Jours_manques': 0
            })
    
    df_couverture_temporelle = pd.DataFrame(resultats_temporels)
    
    # =============================================================================
    # 3. ANALYSE DES DISTRIBUTIONS
    # =============================================================================
    
    resultats_distributions = []
    
    for station in stations:
        station_data = df_filtered[df_filtered['Station_meteo'] == station]
        
        if not station_data.empty:
            # Calculer les statistiques descriptives pour chaque colonne météo
            for colonne in colonnes_meteo:
                if not station_data[colonne].dropna().empty:
                    stats = station_data[colonne].describe()
                    
                    resultats_distributions.append({
                        'Station_meteo': station,
                        'Indicateur': colonne,
                        'Count': int(stats['count']),
                        'Mean': round(stats['mean'], 2),
                        'Std': round(stats['std'], 2),
                        'Min': round(stats['min'], 2),
                        'Q25': round(stats['25%'], 2),
                        'Median': round(stats['50%'], 2),
                        'Q75': round(stats['75%'], 2),
                        'Max': round(stats['max'], 2)
                    })
                else:
                    # Si toutes les valeurs sont manquantes
                    resultats_distributions.append({
                        'Station_meteo': station,
                        'Indicateur': colonne,
                        'Count': 0,
                        'Mean': np.nan,
                        'Std': np.nan,
                        'Min': np.nan,
                        'Q25': np.nan,
                        'Median': np.nan,
                        'Q75': np.nan,
                        'Max': np.nan
                    })
    
    df_distributions = pd.DataFrame(resultats_distributions)
    
    # =============================================================================
    # CRÉATION DU DICTIONNAIRE DE RÉSULTATS
    # =============================================================================
    
    resultats = {
        'valeurs_manquantes': {
            'tableau_detaille': df_valeurs_manquantes,
            'tableau_pivot': pivot_missing,
            'resume': f"Analyse de {len(stations)} stations sur {len(colonnes_meteo)} indicateurs"
        },
        'couverture_temporelle': {
            'tableau': df_couverture_temporelle,
            'resume': f"Période d'analyse : {df_filtered['Date'].min()} à {df_filtered['Date'].max()}"
        },
        'distributions': {
            'tableau': df_distributions,
            'resume': f"Statistiques descriptives pour {len(colonnes_meteo)} indicateurs"
        }
    }
    
    # =============================================================================
    # AFFICHAGE DES RÉSULTATS
    # =============================================================================
    
    print("="*80)
    print("ANALYSE COMPARATIVE DES STATIONS MÉTÉOROLOGIQUES")
    print("="*80)
    
    print(f"\n1. VALEURS MANQUANTES (en %)")
    print("-"*50)
    if not pivot_missing.empty:
        print(pivot_missing.to_string())
    else:
        print("Aucune donnée à afficher")
    
    print(f"\n2. COUVERTURE TEMPORELLE")
    print("-"*50)
    print(df_couverture_temporelle.to_string(index=False))
    
    print(f"\n3. APERÇU DES DISTRIBUTIONS (moyennes)")
    print("-"*50)
    if not df_distributions.empty:
        # Créer un tableau pivot des moyennes pour un aperçu rapide
        pivot_moyennes = df_distributions.pivot(
            index='Indicateur', 
            columns='Station_meteo', 
            values='Mean'
        )
        print(pivot_moyennes.to_string())
    else:
        print("Aucune donnée à afficher")
    
    return resultats



def analyser_completude_weather_data(data):
    """
    Analyse la complétude de la base de données météo en comparant les occurrences 
    théoriques vs réelles pour chaque station, basé sur les dates min/max globales.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame contenant au minimum les colonnes 'Date' et 'Station_meteo'
    
    Returns:
    --------
    dict
        Dictionnaire contenant :
        - 'analyse_globale' : DataFrame avec l'analyse par station
        - 'dates_manquantes' : DataFrame détaillé des dates manquantes
        - 'statistiques' : Dict avec les statistiques globales
        - 'periode_analyse' : Tuple (date_min, date_max)
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
    
    #  Obtenir la période globale (min/max de toutes les stations)
    date_min_global = data['Date'].min()
    date_max_global = data['Date'].max()
    
    print(f" PÉRIODE D'ANALYSE : {date_min_global.date()} → {date_max_global.date()}")
    
    #  Calculer le nombre théorique de jours
    all_dates = pd.date_range(start=date_min_global, end=date_max_global, freq='D', inclusive='both')
    nb_jours_theorique = len(all_dates)
    
    print(f" Nombre de jours théoriques par station : {nb_jours_theorique}")
    
    #  Obtenir toutes les stations
    stations = data['Station_meteo'].unique()
    nb_stations = len(stations)
    
    print(f" Nombre de stations météo : {nb_stations}")
    print(f" Nombre total d'occurrences théoriques : {nb_jours_theorique * nb_stations:,}")
    
    # =============================================================================
    # ANALYSE PAR STATION
    # =============================================================================
    
    analyse_par_station = []
    toutes_dates_manquantes = []
    
    print(f"\n{'='*80}")
    print(f" ANALYSE DÉTAILLÉE PAR STATION")
    print(f"{'='*80}")
    
    for i, station in enumerate(sorted(stations), 1):
        print(f"\n[{i:2d}/{nb_stations}] 📡 Analyse de {station.upper()}")
        
        # Données réelles pour cette station
        station_data = data[data['Station_meteo'] == station]
        nb_occurrences_reelles = len(station_data)
        
        # Dates présentes pour cette station
        dates_presentes = set(station_data['Date'].dt.date)
        
        # Identifier les dates manquantes
        dates_manquantes_station = []
        for date in all_dates:
            if date.date() not in dates_presentes:
                dates_manquantes_station.append({
                    'Station_meteo': station,
                    'Date_manquante': date.date(),
                    'Annee': date.year,
                    'Mois': date.month,
                    'Semaine_ISO': date.isocalendar().week,
                    'Jour_semaine': date.strftime('%A')
                })
        
        # Statistiques pour cette station
        nb_manquantes = len(dates_manquantes_station)
        pourcentage_completude = (nb_occurrences_reelles / nb_jours_theorique * 100)
        pourcentage_manquant = (nb_manquantes / nb_jours_theorique * 100)
        
        # Ajouter à l'analyse globale
        analyse_par_station.append({
            'Station_meteo': station,
            'Occurrences_theoriques': nb_jours_theorique,
            'Occurrences_reelles': nb_occurrences_reelles,
            'Occurrences_manquantes': nb_manquantes,
            'Pourcentage_completude': round(pourcentage_completude, 1),
            'Pourcentage_manquant': round(pourcentage_manquant, 1),
            'Date_min_station': station_data['Date'].min().date() if len(station_data) > 0 else None,
            'Date_max_station': station_data['Date'].max().date() if len(station_data) > 0 else None
        })
        
        # Ajouter les dates manquantes à la liste globale
        toutes_dates_manquantes.extend(dates_manquantes_station)
        
        # Affichage des résultats pour cette station
        print(f"    Occurrences réelles    : {nb_occurrences_reelles:,}")
        print(f"    Occurrences manquantes : {nb_manquantes:,}")
        print(f"    Complétude            : {pourcentage_completude:.1f}%")
        
        if nb_manquantes > 0:
            print(f"     Données incomplètes   : {pourcentage_manquant:.1f}% manquant")
        else:
            print(f"    Station complète !")
    
    # =============================================================================
    # CRÉATION DES DATAFRAMES DE RÉSULTATS
    # =============================================================================
    
    df_analyse_globale = pd.DataFrame(analyse_par_station)
    df_dates_manquantes = pd.DataFrame(toutes_dates_manquantes)
    
    # =============================================================================
    # STATISTIQUES GLOBALES
    # =============================================================================
    
    nb_total_theorique = nb_jours_theorique * nb_stations
    nb_total_reel = df_analyse_globale['Occurrences_reelles'].sum()
    nb_total_manquant = df_analyse_globale['Occurrences_manquantes'].sum()
    
    completude_globale = (nb_total_reel / nb_total_theorique * 100)
    
    statistiques = {
        'nb_stations': nb_stations,
        'nb_jours_theorique_par_station': nb_jours_theorique,
        'nb_total_theorique': nb_total_theorique,
        'nb_total_reel': nb_total_reel,
        'nb_total_manquant': nb_total_manquant,
        'completude_globale_pct': round(completude_globale, 1),
        'stations_completes': (df_analyse_globale['Pourcentage_completude'] == 100.0).sum(),
        'stations_incompletes': (df_analyse_globale['Pourcentage_completude'] < 100.0).sum()
    }
    
    # =============================================================================
    # AFFICHAGE DES RÉSULTATS GLOBAUX
    # =============================================================================
    
    print(f"\n{'='*80}")
    print(f" RÉSULTATS GLOBAUX")
    print(f"{'='*80}")
    print(f" Stations analysées        : {nb_stations}")
    print(f" Période d'analyse         : {date_min_global.date()} → {date_max_global.date()}")
    print(f" Jours théoriques/station  : {nb_jours_theorique}")
    print(f" Total théorique           : {nb_total_theorique:,} occurrences")
    print(f" Total réel                : {nb_total_reel:,} occurrences")
    print(f" Total manquant            : {nb_total_manquant:,} occurrences")
    print(f" Complétude globale        : {completude_globale:.1f}%")
    print(f" Stations complètes        : {statistiques['stations_completes']}")
    print(f" Stations incomplètes      : {statistiques['stations_incompletes']}")
    
    # =============================================================================
    # ANALYSES COMPLÉMENTAIRES
    # =============================================================================
    
    if len(df_dates_manquantes) > 0:
        print(f"\n{'='*80}")
        print(f" ANALYSES COMPLÉMENTAIRES")
        print(f"{'='*80}")
        
        # Top 5 des stations les plus problématiques
        top_problematiques = df_analyse_globale.nlargest(5, 'Occurrences_manquantes')
        print(f"\n TOP 5 stations avec le plus de données manquantes :")
        for _, row in top_problematiques.iterrows():
            print(f"   {row['Station_meteo']:<20} : {row['Occurrences_manquantes']:,} jours manquants ({row['Pourcentage_manquant']:.1f}%)")
        
        # Analyse par année si applicable
        if 'Annee' in df_dates_manquantes.columns:
            manquants_par_annee = df_dates_manquantes['Annee'].value_counts().sort_index()
            print(f"\n Répartition des données manquantes par année :")
            for annee, nb_manquant in manquants_par_annee.items():
                print(f"   {annee} : {nb_manquant:,} jours manquants")
    
    # =============================================================================
    # RETOUR DES RÉSULTATS
    # =============================================================================
    
    resultats = {
        'analyse_globale': df_analyse_globale,
        'dates_manquantes': df_dates_manquantes,
        'statistiques': statistiques,
        'periode_analyse': (date_min_global, date_max_global)
    }
    
    print(f"\n{'='*80}")
    print(f" ANALYSE TERMINÉE - Résultats disponibles dans le dictionnaire retourné")
    print(f"{'='*80}")
    
    return resultats



def analyser_jours_manquants(df):
    """
    Analyse simple des jours manquants par station météo
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame avec colonnes 'Date' et 'Station_meteo'
    
    Returns:
    --------
    dict avec 3 tableaux :
        - 'total_par_station' : nombre total de jours manquants par station
        - 'detail_jours_manquants' : liste complète des jours manquants
        - 'par_annee_station' : nombre de jours manquants par année et station
    """
    
    # Copie et conversion de la date
    df_work = df.copy()
    df_work['Date'] = pd.to_datetime(df_work['Date'])
    
    # Période globale
    date_min = df_work['Date'].min()
    date_max = df_work['Date'].max()
    
    # Toutes les dates théoriques
    toutes_dates = pd.date_range(start=date_min, end=date_max, freq='D')
    stations = df_work['Station_meteo'].unique()
    
    print(f" Période : {date_min.date()} → {date_max.date()}")
    print(f" Stations : {len(stations)}")
    print(f" Jours théoriques : {len(toutes_dates)}")
    
    # 1. TABLEAU : Jours manquants par station
    total_par_station = []
    
    # 2. TABLEAU : Détail de tous les jours manquants  
    detail_jours_manquants = []
    
    for station in stations:
        # Dates présentes pour cette station
        dates_station = set(df_work[df_work['Station_meteo'] == station]['Date'].dt.date)
        
        # Identifier les dates manquantes
        dates_manquantes = [d for d in toutes_dates if d.date() not in dates_station]
        
        # Total par station
        total_par_station.append({
            'Station_meteo': station,
            'Jours_manquants': len(dates_manquantes)
        })
        
        # Détail des jours manquants
        for date_manq in dates_manquantes:
            detail_jours_manquants.append({
                'Station_meteo': station,
                'Date_manquante': date_manq.date(),
                'Annee': date_manq.year,
                'Semaine': date_manq.isocalendar().week,
                'Jour_semaine': date_manq.strftime('%A')
            })
    
    # Convertir en DataFrames
    df_total = pd.DataFrame(total_par_station).sort_values('Jours_manquants', ascending=False)
    df_detail = pd.DataFrame(detail_jours_manquants).sort_values(['Station_meteo', 'Date_manquante'])
    
    # 3. TABLEAU : Par année et station
    if len(df_detail) > 0:
        df_par_annee = df_detail.groupby(['Station_meteo', 'Annee']).size().reset_index(name='Jours_manquants')
        df_par_annee = df_par_annee.pivot(index='Station_meteo', columns='Annee', values='Jours_manquants').fillna(0)
    else:
        df_par_annee = pd.DataFrame()
    
    # Résultats
    resultats = {
        'total_par_station': df_total,
        'detail_jours_manquants': df_detail, 
        'par_annee_station': df_par_annee
    }
    
    # Affichage rapide
    print(f"\n RÉSULTATS :")
    print(f"   • Total jours manquants : {len(df_detail)}")
    print(f"   • Stations affectées : {df_total[df_total['Jours_manquants'] > 0].shape[0]}")
    
    return resultats


def analyser_valeurs_manquantes_stations(df, colonnes=None, titre_heatmap="Pourcentage de valeurs manquantes par station météo"):
    """
    Analyse simple des valeurs manquantes par station météo avec heatmap
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame avec les données météo
    colonnes : list, optional
        Liste des colonnes à analyser. Par défaut: ['Pluie','Humidite_max', 'Humidite_min', 'Tmax', 'Tmin', 'Tmoy']
    titre_heatmap : str, optional
        Titre pour la heatmap
    
    Returns:

    --------
    dict avec :
        - 'tableau_complet' : DataFrame avec toutes les analyses
        - 'tableau_pourcentages' : DataFrame pour visualisation
        - 'heatmap' : objet heatmap
        - 'synthese' : dict avec moyennes par colonne
    """
    
    # Colonnes par défaut
    if colonnes is None:
        colonnes = ['Pluie','Humidite_max', 'Humidite_min', 'Tmax', 'Tmin', 'Tmoy']
    
    # Analyser chaque station
    resultats = []
    
    for station in df['Station_meteo'].unique():
        data_station = df[df['Station_meteo'] == station]
        nb_occurences = len(data_station)
        
        # Résultat pour cette station
        station_result = {'Station_meteo': station, 'Nombre_occurrences': nb_occurences}
        
        # Pour chaque colonne
        for colonne in colonnes:
            nb_manquants = data_station[colonne].isna().sum()
            pourcentage_manquant = (nb_manquants / nb_occurences * 100).round(1)
            station_result[f'{colonne}_manquant_%'] = pourcentage_manquant
            station_result[f'{colonne}_manquant_nb'] = nb_manquants
        
        resultats.append(station_result)
    
    # Créer le DataFrame complet
    df_complet = pd.DataFrame(resultats).sort_values('Station_meteo')
    
    # Calculer le score moyen de problème
    colonnes_pct = [col for col in df_complet.columns if col.endswith('_manquant_%')]
    df_complet['Score_probleme_moyen'] = df_complet[colonnes_pct].mean(axis=1).round(1)
    
    # Créer le DataFrame pour la heatmap
    colonnes_pour_viz = ['Station_meteo'] + colonnes_pct
    df_viz = df_complet[colonnes_pour_viz].set_index('Station_meteo')
    df_viz.columns = [col.replace('_manquant_%', '') for col in df_viz.columns]
    
    # Créer la heatmap
    heatmap_fig = plot_heatmap(
        data=df_viz,
        title=titre_heatmap,
        xlabel="Weather variables", 
        ylabel="Weather stations",
        cmap="Reds",
        cbar_label="missing %",
        fmt=".1f",
        figsize=(10, 6)
    )
    
    # Synthèse par colonne
    synthese = {}
    for colonne in colonnes:
        col_pct = f'{colonne}_manquant_%'
        moyenne = df_complet[col_pct].mean()
        synthese[colonne] = round(moyenne, 1)
    
    # Affichage rapide
    print("ANALYSE DES VALEURS MANQUANTES PAR STATION")
    print("=" * 60)
    print(f"Stations analysées : {len(df_complet)}")
    print(f"Variables analysées : {len(colonnes)}")
    
    # Top 3 colonnes les plus problématiques
    moyennes_triees = sorted(synthese.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 3 variables les plus manquantes :")
    for colonne, moyenne in moyennes_triees[:3]:
        print(f"   {colonne:<15} : {moyenne:>6.1f}%")
    
    # Top 3 stations les plus problématiques
    top_stations = df_complet.nlargest(3, 'Score_probleme_moyen')
    print(f"\nTop 3 stations les plus problématiques :")
    for _, row in top_stations.iterrows():
        print(f"   {row['Station_meteo']:<15} : {row['Score_probleme_moyen']:>6.1f}%")
    
    # Retourner tout
    return {
        'tableau_complet': df_complet,
        'tableau_pourcentages': df_viz,
        'heatmap': heatmap_fig,
        'synthese': synthese
    }


#################################################### Base cercos #########################################################################

def analyser_completude_cercos(df_cercos, afficher_graphique=True, afficher_resultats=True):
    """
    Analyse la complétude des données cercos par zone de traitement.
    
    Args:
        df_cercos: DataFrame contenant les données cercos
        afficher_graphique: bool, afficher le graphique des occurrences manquantes
        afficher_resultats: bool, afficher les résultats numériques
    
    Returns:
        DataFrame: Tableau de synthèse avec les statistiques de complétude par zone
    """
    # Calculer automatiquement le nombre de semaines réelles pour 2025
    max_semaine_2025 = df_cercos[df_cercos['Annee'] == 2025]['Semaine'].max()
    nb_semaines_theorique = 52 + 52 + max_semaine_2025  # 2023 + 2024 + 2025 réel
    
    # Analyse par zone
    zone_analysis = df_cercos.groupby('Zone_traitement').agg({
        'Post_observation': 'nunique',
        'Annee': 'count'
    }).rename(columns={
        'Post_observation': 'nb_postes_uniques',
        'Annee': 'nb_occurrences_totales'
    })
    
    # Calculs de complétude
    zone_analysis['nb_occurrences_theorique'] = zone_analysis['nb_postes_uniques'] * nb_semaines_theorique
    zone_analysis['occurrences_manquantes'] = zone_analysis['nb_occurrences_theorique'] - zone_analysis['nb_occurrences_totales']
    zone_analysis['pourcentage_completude'] = (zone_analysis['nb_occurrences_totales'] / 
                                             zone_analysis['nb_occurrences_theorique'] * 100).round(2)
    
    # Trier par pourcentage de complétude
    resultats = zone_analysis.sort_values('pourcentage_completude')
    
    # Affichage optionnel des informations
    if afficher_resultats:
        print(f"Dernière semaine 2025 détectée: {max_semaine_2025}")
        print(f"Nombre total de semaines théoriques: {nb_semaines_theorique}")
        print("\nTableau de complétude par zone:")
        print(resultats)
    
    # Graphique optionnel
    if afficher_graphique:
        # Afficher les occurrences manquantes
        plot_bar(resultats.reset_index(), x='Zone_traitement', y='occurrences_manquantes', 
                 title='Nombre de semaines manquantes par zone de traitement pour tout ses postes d\'observation',
                 xlabel='Zone de traitement', ylabel='Occurrences manquantes', rotation=45)
        
        # Afficher aussi les valeurs numériques
        print("\nOccurrences manquantes par zone :")
        print(resultats['occurrences_manquantes'].sort_values(ascending=False))
    
    return resultats


def analyser_valeurs_manquantes_par_zone(df, 
                                        colonne_zone='Zone_traitement',
                                        colonne_poste='Post_observation', 
                                        indicateurs=None,
                                        afficher_details=True,
                                        top_zones_problematiques=5,
                                        max_postes_par_zone=3):
    """
    Analyse complète des valeurs manquantes par zone de traitement.
    
    Args:
        df: DataFrame à analyser
        colonne_zone: nom de la colonne contenant les zones de traitement
        colonne_poste: nom de la colonne contenant les postes d'observation
        indicateurs: liste des indicateurs à analyser (si None, détection automatique)
        afficher_details: afficher les détails dans la console
        top_zones_problematiques: nombre de zones problématiques à détailler
        max_postes_par_zone: nombre max de postes à afficher par zone
    
    Returns:
        dict: dictionnaire contenant tous les résultats de l'analyse
    """
    
    # Définir les indicateurs par défaut si non spécifiés
    if indicateurs is None:
        indicateurs = ['Nff_moyen', 'Nfr_moyen', 'Pjfn_moyen', 'Pjft_moyen', 'Etat_devolution_moy', 'Dp_moy']
    
    if afficher_details:
        print("=== ANALYSE DES VALEURS MANQUANTES PAR ZONE DE TRAITEMENT ===")
    
    # Créer un DataFrame pour stocker les résultats
    resultats_manquants = []
    
    # Pour chaque zone de traitement
    for zone in df[colonne_zone].unique():
        # Filtrer les données pour cette zone
        data_zone = df[df[colonne_zone] == zone]
        nb_occurences = len(data_zone)
        nb_postes_uniques = data_zone[colonne_poste].nunique()
        
        # Calculer le pourcentage de valeurs manquantes pour chaque indicateur
        zone_result = {
            colonne_zone: zone, 
            'Nombre_occurrences': nb_occurences,
            'Nb_postes_uniques': nb_postes_uniques
        }
        
        for indicateur in indicateurs:  
            if indicateur in df.columns:
                nb_manquants = data_zone[indicateur].isna().sum()
                pourcentage_manquant = (nb_manquants / nb_occurences * 100).round(1)
                zone_result[f'{indicateur}_manquant_%'] = pourcentage_manquant
                zone_result[f'{indicateur}_manquant_nb'] = nb_manquants
        
        resultats_manquants.append(zone_result)
    
    # Convertir en DataFrame
    df_manquants_par_zone = pd.DataFrame(resultats_manquants)
    df_manquants_par_zone = df_manquants_par_zone.sort_values(colonne_zone)
    
    # Colonnes de pourcentages
    colonnes_pourcentages = [colonne_zone, 'Nombre_occurrences', 'Nb_postes_uniques'] + [col for col in df_manquants_par_zone.columns if col.endswith('_manquant_%')]
    
    if afficher_details:
        print("Analyse des valeurs manquantes par zone de traitement :")
        print("=" * 80)
        print("\nPourcentages de valeurs manquantes par zone et par indicateur :")
        print(df_manquants_par_zone[colonnes_pourcentages].to_string(index=False))
    
    # Calculer la moyenne des pourcentages manquants par indicateur
    moyennes_manquantes = {}
    for indicateur in indicateurs:
        if indicateur in df.columns:
            col_pourcentage = f'{indicateur}_manquant_%'
            if col_pourcentage in df_manquants_par_zone.columns:
                moyenne = df_manquants_par_zone[col_pourcentage].mean()
                moyennes_manquantes[indicateur] = moyenne
    
    # Trier par pourcentage décroissant
    moyennes_triees = sorted(moyennes_manquantes.items(), key=lambda x: x[1], reverse=True)
    
    if afficher_details:
        print("\n" + "=" * 80)
        print("SYNTHÈSE DES INDICATEURS LES PLUS PROBLÉMATIQUES :")
        print("=" * 80)
        print("Moyenne des valeurs manquantes par indicateur (toutes zones confondues) :")
        for indicateur, moyenne in moyennes_triees:
            print(f"{indicateur:<25} : {moyenne:>6.1f}%")
    
    # Calculer un score de "problème" pour chaque zone
    colonnes_pourcentages_calc = [col for col in df_manquants_par_zone.columns if col.endswith('_manquant_%')]
    df_manquants_par_zone['Score_probleme_moyen'] = df_manquants_par_zone[colonnes_pourcentages_calc].mean(axis=1).round(1)
    
    # Trier par score décroissant
    zones_problematiques = df_manquants_par_zone.sort_values('Score_probleme_moyen', ascending=False)
    
    if afficher_details:
        print("\n" + "=" * 80)
        print("ZONES AVEC LE PLUS DE PROBLÈMES :")
        print("=" * 80)
        print("Zones classées par score de problème (moyenne des % manquants) :")
        print(zones_problematiques[[colonne_zone, 'Nombre_occurrences', 'Nb_postes_uniques', 'Score_probleme_moyen']].to_string(index=False))
    
    # Détail des postes concernés par les valeurs manquantes
    details_postes = {}
    
    if afficher_details:
        print("\n" + "=" * 80)
        print("DÉTAIL DES POSTES AVEC VALEURS MANQUANTES :")
        print("=" * 80)
    
    for zone in zones_problematiques.head(top_zones_problematiques)[colonne_zone]:
        if afficher_details:
            print(f"\n--- Zone: {zone} ---")
        
        data_zone = df[df[colonne_zone] == zone]
        
        # Identifier les postes avec des valeurs manquantes
        postes_avec_na = data_zone[data_zone[indicateurs].isna().any(axis=1)][colonne_poste].unique()
        
        zone_details = {
            'postes_avec_na': list(postes_avec_na),
            'detail_par_poste': {}
        }
        
        if len(postes_avec_na) > 0:
            if afficher_details:
                print(f"Postes avec valeurs manquantes: {', '.join(postes_avec_na)}")
            
            # Détail par poste
            for poste in postes_avec_na[:max_postes_par_zone]:
                data_poste = data_zone[data_zone[colonne_poste] == poste]
                na_counts = data_poste[indicateurs].isna().sum()
                na_indicators = [ind for ind, count in na_counts.items() if count > 0]
                
                zone_details['detail_par_poste'][poste] = {
                    'indicateurs_manquants': na_indicators,
                    'nb_valeurs_manquantes': na_counts[na_counts > 0].sum()
                }
                
                if na_indicators and afficher_details:
                    print(f"  → {poste}: {', '.join(na_indicators)} ({na_counts[na_counts > 0].sum()} valeurs manquantes)")
        else:
            if afficher_details:
                print("Aucun poste avec valeurs manquantes")
        
        details_postes[zone] = zone_details
    
    # Préparer les données pour le heatmap
    colonnes_pourcentages_viz = [colonne_zone] + [col for col in df_manquants_par_zone.columns if col.endswith('_manquant_%')]
    df_viz = df_manquants_par_zone[colonnes_pourcentages_viz].set_index(colonne_zone)
    
    # Nettoyer les noms de colonnes pour l'affichage
    df_viz.columns = [col.replace('_manquant_%', '') for col in df_viz.columns]
    
    # Retourner tous les résultats dans un dictionnaire
    resultats_cercos = {
        'df_manquants_par_zone': df_manquants_par_zone,
        'tableau_pourcentages': df_manquants_par_zone[colonnes_pourcentages],
        'moyennes_par_indicateur': dict(moyennes_triees),
        'zones_problematiques': zones_problematiques,
        'details_postes': details_postes,
        'heatmap_data': df_viz,  # Données pour la heatmap dans un objet séparé
        'indicateurs_analyses': indicateurs,
        'resume_stats': {
            'nb_zones_total': len(df[colonne_zone].unique()),
            'nb_postes_total': len(df[colonne_poste].unique()),
            'indicateur_plus_problematique': moyennes_triees[0][0] if moyennes_triees else None,
            'zone_plus_problematique': zones_problematiques.iloc[0][colonne_zone] if len(zones_problematiques) > 0 else None
        }
    }
    
    return resultats_cercos

########### test statistique #######################

def analyser_normalite_indicateurs(df, indicateurs_a_tester=None):
    """
    Test de normalité simple avec recommandations
    
    Effectue un test de Shapiro-Wilk sur les indicateurs spécifiés pour évaluer 
    leur normalité et recommande le type de tests statistiques appropriés.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données à analyser
    indicateurs_a_tester : list, optional
        Liste des noms de colonnes à tester. Si None, utilise les indicateurs 
        cercos par défaut : ['Pjft_moyen', 'Pjfn_moyen', 'Nff_moyen', 
        'Nfr_moyen', 'Etat_devolution_moy', 'Dp_moy']
    
    Returns:
    --------
    dict
        Dictionnaire contenant pour chaque indicateur :
        - 'p_value' : float, p-value du test de Shapiro-Wilk
        - 'normal' : bool, True si la distribution est normale (p > 0.05)
    
    Prints:
    -------
    - Résultats détaillés pour chaque indicateur
    - Recommandations de tests statistiques
    - Tableau de synthèse des résultats
    
    Examples:
    ---------
    >>> # Utilisation avec les indicateurs par défaut
    >>> resultats = analyser_normalite_indicateurs(df_cercos)
    
    >>> # Utilisation avec des indicateurs personnalisés
    >>> indicateurs_meteo = ['Pluie_sum', 'Tmax_mean', 'Tmin_mean', 'Humidite_max_mean']
    >>> resultats_meteo = analyser_normalite_indicateurs(df_meteo, indicateurs_meteo)
    
    """
    if indicateurs_a_tester is None:
        # Détection automatique des colonnes numériques si aucune liste fournie
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclure les colonnes d'identifiants
        exclude_cols = ['Annee', 'Semaine']
        indicateurs_a_tester = [col for col in numeric_columns if col not in exclude_cols]
        
        # Si aucune colonne numérique détectée, utiliser les indicateurs cercos par défaut
        if not indicateurs_a_tester:
            indicateurs_a_tester = ['Pjft_moyen', 'Pjfn_moyen', 'Nff_moyen', 'Nfr_moyen', 'Etat_devolution_moy', 'Dp_moy']
    
    # Vérifier que les colonnes existent dans le DataFrame
    colonnes_existantes = [col for col in indicateurs_a_tester if col in df.columns]
    colonnes_manquantes = [col for col in indicateurs_a_tester if col not in df.columns]
    
    if colonnes_manquantes:
        print(f"ATTENTION: Colonnes non trouvées dans le DataFrame: {colonnes_manquantes}")
    
    if not colonnes_existantes:
        print("ERREUR: Aucune colonne valide trouvée pour l'analyse.")
        return {}
    
    print("TEST DE NORMALITÉ DES INDICATEURS\n")
    print(f"Colonnes analysées: {len(colonnes_existantes)}")
    print(f"Méthode: Test de Shapiro-Wilk (H0: distribution normale)\n")
    
    resultats = {}
    
    for indicateur in colonnes_existantes:
        # Nettoyer les données (supprimer les valeurs manquantes)
        data_clean = df[indicateur].dropna()
        
        if len(data_clean) < 3:
            print(f"{indicateur}:")
            print(f"  ERREUR: Pas assez de données valides ({len(data_clean)} valeurs)")
            print()
            continue
            
        try:
            stat, p_value = shapiro(data_clean)
            normal = p_value > 0.05
            resultats[indicateur] = {'p_value': p_value, 'normal': normal}
            
            print(f"{indicateur}:")
            print(f"  N = {len(data_clean)} observations")
            print(f"  Statistique W = {stat:.4f}")
            print(f"  P-value = {p_value:.6f}")
            print(f"  Distribution normale: {'OUI' if normal else 'NON'}")
            print(f"  Tests recommandés: {'Paramétriques (ANOVA, t-test)' if normal else 'Non-paramétriques (Kruskal-Wallis, Dunn)'}")
            print()
            
        except Exception as e:
            print(f"{indicateur}:")
            print(f"  ERREUR lors du calcul: {str(e)}")
            print()
    
    if resultats:
        # Résumé global
        normaux = sum(1 for r in resultats.values() if r['normal'])
        total = len(resultats)
        
        print("=" * 50)
        print("RÉSUMÉ GLOBAL:")
        print(f"Indicateurs avec distribution normale: {normaux}/{total}")
        print(f"Pourcentage de normalité: {(normaux/total)*100:.1f}%")
        
        if normaux == 0:
            print("RECOMMANDATION GÉNÉRALE: Utiliser des tests NON-PARAMÉTRIQUES")
        elif normaux == total:
            print("RECOMMANDATION GÉNÉRALE: Utiliser des tests PARAMÉTRIQUES")
        else:
            print("RECOMMANDATION GÉNÉRALE: Utiliser des tests NON-PARAMÉTRIQUES (pour la cohérence)")
        
        # Tableau de synthèse
        df_synthese = pd.DataFrame([
            {
                'Indicateur': ind,
                'P_value': f"{res['p_value']:.6f}",
                'Normal': 'OUI' if res['normal'] else 'NON',
                'Tests_recommandés': 'Paramétriques' if res['normal'] else 'Non-paramétriques'
            }
            for ind, res in resultats.items()
        ])
        
        print("\nTABLEAU DE SYNTHÈSE:")
        print(df_synthese.to_string(index=False))
    
    return resultats


def analyser_zones_similaires_simple(df, indicateur, colonne_groupement='Zone_traitement'):
    """
    Identifier les zones/stations avec des comportements similaires en utilisant le test de Dunn
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    indicateur : str
        Nom de l'indicateur à analyser
    colonne_groupement : str, optional
        Colonne utilisée pour grouper ('Zone_traitement' ou 'Station_meteo')
        Par défaut : 'Zone_traitement'
    
    Returns:
    --------
    list : Liste des groupes similaires avec leurs moyennes
    """
    import scikit_posthocs as sp
    import pandas as pd
    
    # Préparer les données
    data_melted = df[[colonne_groupement, indicateur]].dropna()
    
    if len(data_melted[colonne_groupement].unique()) < 2:
        print(f"Pas assez de {colonne_groupement} avec des données pour effectuer le test")
        return []
    
    try:
        # Test de Dunn avec correction de Bonferroni
        dunn_results = sp.posthoc_dunn(
            data_melted, 
            val_col=indicateur, 
            group_col=colonne_groupement, 
            p_adjust='bonferroni'
        )
        
        # Calculer les moyennes par groupe
        moyennes_groupes = data_melted.groupby(colonne_groupement)[indicateur].mean()
        
        # Identifier les groupes similaires (p >= 0.05)
        groupes_similaires = []
        groupes_uniques = dunn_results.index.tolist()
        
        for i, groupe1 in enumerate(groupes_uniques):
            for j, groupe2 in enumerate(groupes_uniques):
                if i < j:  # Éviter les doublons et auto-comparaisons
                    p_value = dunn_results.loc[groupe1, groupe2]
                    if p_value >= 0.05:  # Similaires
                        moy1 = moyennes_groupes[groupe1]
                        moy2 = moyennes_groupes[groupe2]
                        groupes_similaires.append(
                            f"{groupe1} ≈ {groupe2} ({moy1:.1f} vs {moy2:.1f})"
                        )
        
        return groupes_similaires
        
    except Exception as e:
        print(f"Erreur lors du test de Dunn : {str(e)}")
        return []



def test_homogeneite_inter_groupes(df, variable_groupement, indicateur='Pjft_moyen'):
    """
    Test ANOVA généralisé pour comparer les moyennes entre différents groupes
    H0: Les moyennes entre groupes sont égales (homogénéité inter-groupes)
    H1: Au moins un groupe a une moyenne différente
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame contenant les données
    variable_groupement : str
        Nom de la colonne utilisée pour grouper (ex: 'Zone_traitement', 'Station_meteo')
    indicateur : str
        Indicateur à analyser
    """
    from scipy.stats import f_oneway
    
    groupes = df[variable_groupement].unique()
    
    if len(groupes) < 2:
        return None, f"Pas assez de {variable_groupement} pour le test"
    
    # Créer les groupes
    groupes_donnees = []
    groupes_valides = []
    
    for groupe in groupes:
        data_groupe = df[df[variable_groupement] == groupe][indicateur].dropna()
        if len(data_groupe) > 0:
            groupes_donnees.append(data_groupe)
            groupes_valides.append(groupe)
    
    if len(groupes_donnees) < 2:
        return None, "Pas assez de groupes valides"
    
    # Test ANOVA
    f_stat, p_value = f_oneway(*groupes_donnees)
    
    # Calculs descriptifs
    moyennes_groupes = [groupe.mean() for groupe in groupes_donnees]
    ecarts_types_groupes = [groupe.std() for groupe in groupes_donnees]
    
    return {
        'variable_groupement': variable_groupement,
        'indicateur': indicateur,
        'nb_groupes': len(groupes_donnees),
        'groupes_testes': groupes_valides,
        'f_statistic': f_stat,
        'p_value': p_value,
        'significatif': 'Oui' if p_value < 0.05 else 'Non',
        'moyennes_par_groupe': dict(zip(groupes_valides, moyennes_groupes)),
        'ecarts_types_par_groupe': dict(zip(groupes_valides, ecarts_types_groupes)),
        'interpretation': 'Groupes hétérogènes' if p_value < 0.05 else 'Groupes homogènes'
    }



def plot_distributions_par_zone(df, colonne_groupe='Zone_traitement', colonnes_numeriques=None):
    """
    Fonction pour afficher les distributions numériques par zone de traitement ou station météo
    en utilisant la fonction plot_numeric_distributions existante.
    
    Parameters:
    -----------
    df : DataFrame
        Le DataFrame contenant les données
    colonne_groupe : str, default 'Zone_traitement'
        La colonne pour grouper les données ('Zone_traitement' ou 'Station_meteo')
    colonnes_numeriques : list, optional
        Liste des colonnes numériques à analyser. Si None, utilise toutes les colonnes numériques
    
    Returns:
    --------
    dict : Dictionnaire avec les groupes comme clés et les DataFrames filtrés comme valeurs
    """
    
    # Vérifier que la colonne de groupement existe
    if colonne_groupe not in df.columns:
        print(f"Erreur : La colonne '{colonne_groupe}' n'existe pas dans le DataFrame")
        return None
    
    # Obtenir les valeurs uniques de la colonne de groupement
    groupes = df[colonne_groupe].unique()
    groupes = sorted([g for g in groupes if pd.notna(g)])  # Supprimer les NaN et trier
    
    print(f"=== ANALYSE DES DISTRIBUTIONS PAR {colonne_groupe.upper()} ===")
    print(f"Nombre de {colonne_groupe.lower()}s trouvé(s) : {len(groupes)}")
    print(f"Liste des {colonne_groupe.lower()}s : {groupes}")
    
    # Créer un dictionnaire pour stocker les DataFrames filtrés
    donnees_par_groupe = {}
    
    # Filtrer les données pour chaque groupe
    for groupe in groupes:
        df_filtre = df[df[colonne_groupe] == groupe].copy()
        donnees_par_groupe[groupe] = df_filtre
        print(f"\n{colonne_groupe} '{groupe}' : {df_filtre.shape[0]} lignes")
    
    return donnees_par_groupe

def afficher_toutes_distributions(donnees_par_groupe, colonnes_numeriques=None):
    """
    Affiche les distributions pour TOUTES les zones/groupes
    
    Parameters:
    -----------
    donnees_par_groupe : dict
        Dictionnaire retourné par plot_distributions_par_zone
    colonnes_numeriques : list, optional
        Liste des colonnes numériques à analyser
    """
    
    print(f"\n=== DISTRIBUTIONS POUR TOUTES LES ZONES ===")
    print(f"Nombre total de groupes : {len(donnees_par_groupe)}")
    
    for i, (nom_groupe, df_groupe) in enumerate(donnees_par_groupe.items(), 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(donnees_par_groupe)}] ZONE : {nom_groupe}")
        print(f"Nombre d'observations : {df_groupe.shape[0]}")
        print(f"{'='*60}")
        
        # Utiliser votre fonction existante pour chaque groupe
        plot_numeric_distributions(df_groupe, columns=colonnes_numeriques)

def afficher_distribution_groupe(donnees_par_groupe, nom_groupe, colonnes_numeriques=None):
    """
    Affiche les distributions pour un groupe spécifique
    
    Parameters:
    -----------
    donnees_par_groupe : dict
        Dictionnaire retourné par plot_distributions_par_zone
    nom_groupe : str
        Nom du groupe à afficher
    colonnes_numeriques : list, optional
        Liste des colonnes numériques à analyser
    """
    
    if nom_groupe not in donnees_par_groupe:
        print(f"Erreur : Le groupe '{nom_groupe}' n'existe pas.")
        print(f"Groupes disponibles : {list(donnees_par_groupe.keys())}")
        return
    
    df_groupe = donnees_par_groupe[nom_groupe]
    
    print(f"\n=== DISTRIBUTIONS POUR : {nom_groupe} ===")
    print(f"Nombre d'observations : {df_groupe.shape[0]}")
    
    # Utiliser votre fonction existante
    plot_numeric_distributions(df_groupe, columns=colonnes_numeriques)





##############################################################################  Intrant ###################################################################



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
    print(f"Lignes ajoutées : {len(lignes_manquantes)}")
    print(f"Taille finale de df_intrant_complet : {df_intrant_complet.shape}")

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
    print(df_intrant_complet.isna().sum())
    
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
            print(f"Périodes manquantes pour la zone {zone} :")
            print(manquantes)

    # Afficher le message si aucune période manquante n'a été trouvée
    if not manquantes_trouvees:
        print("Toutes les périodes sont présentes pour toutes les zones.")

    # Vérifier les années et semaines globales
    annees_cercos = df_cercos['Annee'].unique()
    annees_intrant = df_intrant_complet['Annee'].unique()

    semaines_cercos = df_cercos['Semaine'].unique()
    semaines_intrant = df_intrant_complet['Semaine'].unique()

    # Vérification des années
    annees_manquantes = set(annees_cercos) - set(annees_intrant)
    if not annees_manquantes:
        print("Toutes les années de df_cercos sont présentes dans df_intrant_complet.")
    else:
        print(f"Années manquantes : {annees_manquantes}")

    # Vérification des semaines
    semaines_manquantes = set(semaines_cercos) - set(semaines_intrant)
    if not semaines_manquantes:
        print("Toutes les semaines de df_cercos sont présentes dans df_intrant_complet.")
    else:
        print(f"Semaines manquantes : {semaines_manquantes}")
    
    return df_intrant_complet