import sys
import os
# Ajoute le dossier parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import streamlit as st
import pandas as pd
from src.preparation_base import *
from src.plot_fonction import *
from src.analyse_fonction import *
from src.streamlit_func import * 
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

path_data_brute, path_data_traite = get_data_paths()  


st.markdown("""
    <h1 style='text-align: center;'>Partie analyse des données</h1>
    <div style='
        background-color: #e3f0fc;    
        border: 1px solid #b6d4f4;
        padding: 18px 28px;
        margin-bottom: 28px;
        margin-top: 12px;
        font-size: 18px;
        color: #1e293b;
        width: 100%;
        text-align: center;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(30,41,59,0.03);
    '>
        Cette section propose une analyse approfondie de chaque base de données, avec des statistiques descriptives présentées sous forme de tableaux, des informations détaillées sur chaque zone de traitement ou station météo, ainsi que des visualisations interactives pour chaque indicateur clé.
    </div>
""", unsafe_allow_html=True)

# 1. Sélection de la base
base = st.selectbox("Choisir la base à analyser", ["Cercos", "Météo", "Intrant", "Finale"])
if base == "Cercos":
    df = load_data(path=path_data_traite + "/df_cercos_traite_mod_zone.parquet", database_name="cercos") 
elif base == "Météo":
    df = load_data(path=path_data_traite + "/df_meteoQlick_traite_mod_zone.parquet", database_name="qlik")
elif base == "Intrant":
    df = load_data(path=path_data_traite + "/df_intrant_complet_mod_zone.parquet", database_name="intrant_comp")
elif base == "Finale":
    df = load_data(path=path_data_traite + "/merge_final_mod_zone.parquet", database_name="merge1")

# 2. Onglets pour stats et graphiques
tab1, tab2, tab3 = st.tabs(["Statistiques descriptives", "Graphiques", "Corrélations"])








with tab1:
    if base == "Cercos" or base == "Intrant":
        with st.expander("ℹ️ Infos sur les zones de traitement", expanded=False):
            if "Zone_traitement" in df.columns:
                nb_zones = df["Zone_traitement"].nunique()
                st.markdown(f"**Nombre de zones de traitement :** {nb_zones}")
                zones = sorted(df["Zone_traitement"].dropna().unique())
                st.markdown(f"**Liste des zones de traitement :** {', '.join(str(z) for z in zones)}")
            else:
                st.info("Colonne 'Zone_traitement' absente de la base.")

    elif base == "Météo":
        with st.expander("ℹ️ Infos sur les stations météo", expanded=False):
            if "Station_meteo" in df.columns:
                nb_stations = df["Station_meteo"].nunique()
                st.markdown(f"**Nombre de stations météo :** {nb_stations}")
                stations = sorted(df["Station_meteo"].dropna().unique())
                st.markdown(f"**Liste des stations météo :** {', '.join(str(s) for s in stations)}")
            else:
                st.info("Colonne 'Station_meteo' absente de la base.")

    elif base == "Finale":
        with st.expander("ℹ️ Infos synthétiques sur la base finale", expanded=False):
            if "Zone_traitement" in df.columns:
                nb_zones = df["Zone_traitement"].nunique()
                st.markdown(f"**Nombre total de zones de traitement :** {nb_zones}")
                zones = sorted(df["Zone_traitement"].dropna().unique())
                st.markdown(f"**Liste des zones de traitement :** {', '.join(str(z) for z in zones)}")
            if "Station_meteo" in df.columns:
                nb_stations = df["Station_meteo"].nunique()
                st.markdown(f"**Nombre total de stations météo :** {nb_stations}")
                stations = sorted(df["Station_meteo"].dropna().unique())
                st.markdown(f"**Liste des stations météo :** {', '.join(str(s) for s in stations)}")

    # Afficher la table de statistiques descriptives globale uniquement pour la base "Finale"
    if base == "Finale":
        st.subheader("Statistiques descriptives sur l'ensemble de la base")
        st.write(df.describe())

    # Statistiques descriptives groupées selon la base sélectionnée
    if base == "Cercos" or base == "Intrant":
        # Pour Cercos et Intrant : groupby par Zone_traitement
        zone_col = None
        for colname in ["Zone_traitement"]:
            if colname in df.columns:
                zone_col = colname
                break
        if zone_col:
            st.subheader("Statistiques descriptives par zone de traitement")
            exclude_cols = [zone_col, "Annee", "Semaine", "Post_observation", "Long", "Lat"]
            num_cols = [col for col in df.select_dtypes(include="number").columns if col not in exclude_cols]
            if num_cols:
                indicateur = st.selectbox("Choisir l'indicateur à afficher", num_cols)
                try:
                    describe_group = df.groupby(zone_col)[[indicateur]].describe().loc[:, (slice(None), ["count", "mean", "std", "min", "max"])]
                    # Renommage des colonnes pour plus de clarté
                    rename_map = { 
                        f"{indicateur}_count": "Nombre d'observation",
                        f"{indicateur}_mean": "Moyenne",
                        f"{indicateur}_std": "Ecart type",
                        f"{indicateur}_min": "Min",
                        f"{indicateur}_max": "Max"
                    }
                    describe_group.columns = [rename_map.get(f"{col[0]}_{col[1]}", col[1]) for col in describe_group.columns]
                    st.dataframe(describe_group, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors du groupby describe : {e}")
            else:
                st.info("Aucun indicateur numérique disponible pour la description groupée.")
        else:
            st.warning("Colonne  'Zone_traitement' non trouvée dans le DataFrame.")
    elif base == "Météo":
        # Pour Météo : groupby par Station_meteo, possibilité de filtrer l'indicateur
        station_col = None
        for colname in ["Station_meteo"]:
            if colname in df.columns:
                station_col = colname
                break
        if station_col:
            st.subheader("Statistiques descriptives par station météo")
            try:
                # Colonnes à prendre en compte uniquement
                meteo_cols = [
                    'Pluie_sum', 'Humidite_max_mean', 'Humidite_min_mean', 'Tmax_mean', 'Tmin_mean',
                    'Pluviometrie_4semaines'
                ]
                num_cols = [col for col in meteo_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
                if not num_cols:
                    st.warning("Aucune des colonnes sélectionnées n'est présente ou numérique dans la base Météo.")
                else:
                    # Ajoute le selectbox pour choisir l'indicateur météo à afficher
                    indicateur = st.selectbox("Choisir l'indicateur météo à afficher", num_cols)
                    describe_group = df.groupby(station_col)[[indicateur]].describe().loc[:, (slice(None), ["count", "mean", "std", "min", "max"])]
                    # Renommage des colonnes pour clarté (pas de doublons)
                    rename_map = {
                        f"{indicateur}_count": "Nombre d'observation",
                        f"{indicateur}_mean": "Moyenne",
                        f"{indicateur}_std": "Ecart type",
                        f"{indicateur}_min": "Min",
                        f"{indicateur}_max": "Max"
                    }
                    describe_group.columns = [rename_map.get(f"{col[0]}_{col[1]}", col[1]) for col in describe_group.columns]
                    st.dataframe(describe_group, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur lors du groupby describe : {e}") 
        else:
            st.warning("Colonne  'Station_meteo' non trouvée dans le DataFrame.")


    # Afficher les graphiques pour toutes les bases
    st.markdown("---")
    st.subheader("Visualisation des distributions des variables:")

    # Histogramme (inchangé)
    non_numeric_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(non_numeric_cols) > 0:
        col = st.selectbox("**Variable non numérique à visualiser**", non_numeric_cols)
        value_counts = df[col].value_counts().sort_values(ascending=False)
    # S'assurer que px est importé au niveau global
    import plotly.express as px
    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        labels={'x': col, 'y': 'Effectif'},
        title=f"Répartition de {col}",
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # --- Distribution des variables numériques d'intérêt avec sélection interactive ---
    st.markdown("---")
    st.subheader("Distribution des variables numériques d'intérêt")
    colonnes_interet = [
        'Annee', 'Semaine', 'Pluie_sum', 'Humidite_max_mean', 'Humidite_min_mean', 'Tmax_mean', 'Tmin_mean',
        'Pluviometrie_4semaines', 'Tmoy_mean', 'Nff_moyen', 'Nfr_moyen', 'Pjfn_moyen', 'Pjft_moyen',
        'Etat_devolution_moy', 'Dp_moy','Long', 'Lat','Quantite', 'Quantite_huile', 'Nb_jours_entre_2_traitements', 'Nb_jours_2mm_tr'
    ]
    colonnes_numeriques = [col for col in colonnes_interet if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    # Déterminer la colonne de regroupement selon la base
    if base == "Cercos" or base == "Intrant":
        colonne_groupe = "Zone_traitement"
        colonne_groupe_label = "zone de traitement"
    elif base == "Météo":
        colonne_groupe = "Station_meteo"
        colonne_groupe_label = "station météo"
    else:
        colonne_groupe = None
        colonne_groupe_label = ""

    if colonne_groupe and colonne_groupe in df.columns:
        # Correction des accords pour les labels
        if colonne_groupe_label == "zone de traitement":
            global_label = "Globale (L'ensemble des zones de traitement)" 
        elif colonne_groupe_label == "station météo":
            global_label = "Globale (L'ensemble des stations météo)"
        else:
            global_label = f"Globale (L'ensemble des {colonne_groupe_label}s)"
        mode = st.radio(
            f"**Afficher la distribution globale ou par {colonne_groupe_label}**",
            [global_label, f"Par {colonne_groupe_label}"]
        )

        if mode.startswith("Globale"):
            if colonnes_numeriques:
                fig = plot_numeric_distributions(df, columns=colonnes_numeriques)
                st.pyplot(fig)
            else:
                st.info("Aucune variable numérique d'intérêt disponible pour la distribution.")
        else:
            groupes = sorted(df[colonne_groupe].dropna().unique())
            groupe_choisi = st.selectbox(f"Sélectionner une {colonne_groupe_label} :", groupes)
            df_groupe = df[df[colonne_groupe] == groupe_choisi]
            if colonnes_numeriques:
                fig = plot_numeric_distributions(df_groupe, columns=colonnes_numeriques)
                st.pyplot(fig)
            else:
                st.info(f"Aucune variable numérique d'intérêt disponible pour la distribution dans cette {colonne_groupe_label}.")
    else:
        if colonnes_numeriques:
            fig = plot_numeric_distributions(df, columns=colonnes_numeriques)
            st.pyplot(fig)
        else:
            st.info("Aucune variable numérique d'intérêt disponible pour la distribution.")


with tab2:
    if base == "Cercos":
        st.subheader("Distribution des indicateurs clés (Cercos)")
        dist_boxplot_indicateurs_cercos(df)

        # Boxplots par année (indicateur en y, année en x)
        st.markdown("---")
        st.subheader("Distribution des indicateurs par année")
        dist_boxplot_indicateurs_annee_cercos(df)

        # Troisième graphique interactif : boxplot par zone de traitement et année pour un indicateur sélectionnable
        st.markdown("---")
        st.subheader("Boxplot interactif par zone de traitement et année")
        dist_boxplot_interactive_cercos(df, plot_boxp_interactive)


    elif base == "Intrant":

        # --- Boxplot interactif par variable et par zone de traitement ---
        dist_ind_intr_zone(df)
 
        # --- Système de sélection flexible pour la quantité (moyenne/somme) par catégorie/zone, intrant/huile ---
        st.markdown("---")
        st.subheader("Quantité d'intrant ou d'huile par catégorie ou zone de traitement")
        dist_ind_intr_group(df, plot_intrant_moyen)

    # --- Barplot Occurence du traitement (sélection semaine ou zone) ---
        st.markdown("---")
        st.subheader("Occurence du traitement par année, semaine ou zone de traitement")
        st.markdown("Ce graphique permet d'identifier les semaines ou les zones les plus traitées chaque année.")
        dist_occ_group(df, base, occurence_traitement)
        
        st.markdown("---")

        # --- HEATMAP INTERACTIVE (quantité totale d'intrant ou d'huile par zone et année) ---
        st.subheader("Carte de chaleur : quantité totale d'intrant ou d'huile par zone et année")
        dist_heatmap_group(df)
 
    
    if base == "Météo":
        st.subheader("Distribution des indicateurs clés ")
        dist_indic_meteo(df)
        
        st.subheader("Distribution des indicateurs météo ")
        dist_indic_meteo_grouped(df) 


with tab3:
    # Liste des colonnes d'intérêt pour la corrélation
    colonnes_interet = [
        'Annee', 'Semaine', 'Pluie_sum', 'Humidite_max_mean', 'Humidite_min_mean', 'Tmax_mean', 'Tmin_mean',
        'Pluviometrie_4semaines', 'Tmoy_mean', 'Nff_moyen', 'Nfr_moyen', 'Pjfn_moyen', 'Pjft_moyen',
        'Etat_devolution_moy', 'Dp_moy','Long', 'Lat', 'Quantite', 'Quantite_huile', 'Nb_jours_entre_2_traitements', 'Nb_jours_2mm_tr'
    ]
    # On garde seulement les colonnes numériques présentes dans le DataFrame
    colonnes_presentes = [col for col in colonnes_interet if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if len(colonnes_presentes) < 2:
        st.warning("Pas assez de colonnes numériques communes pour afficher une matrice de corrélation.")
    else:
        df_corr = df[colonnes_presentes]
        st.markdown(f"**Représentation des corrélations entre les variables numériques de la base : {base}**")
        if base == "Finale":
            fig = plot_correlation_matrix(df_corr, figsize=(18, 14), annot=True, fmt='.2f')
        else:
            fig = plot_correlation_matrix(df_corr, figsize=(8, 8), annot=True, fmt='.2f')
        st.pyplot(fig)


