import sys
import os
# Ajoute le dossier parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd

import pandas as pd
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.preparation_base import *
from src.plot_fonction import *
from src.analyse_fonction import *
import warnings
warnings.filterwarnings("ignore")
import missingno as msno 
import sidetable as stb
import missingno as msno
from itertools import product
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st

##################################################### Page analyse #####################################################

############################ Base cercos #############################

def dist_boxplot_indicateurs_cercos(df):
    """
    Affiche des boxplots Plotly pour la distribution de plusieurs indicateurs Cercos sur l'ensemble des années.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes des indicateurs à visualiser.

    Returns:
        None. Affiche les graphiques directement dans Streamlit.
    """

    indicateurs = ["Etat_devolution_moy", 'Pjft_moyen', 'Nff_moyen', 'Nfr_moyen', 'Dp_moy', "Pjfn_moyen"]
    # Boxplots simples (toutes années confondues)
    fig = make_subplots(rows=3, cols=2, subplot_titles=indicateurs)
    for i, indicateur in enumerate(indicateurs):
        row = (i // 2) + 1
        col = (i % 2) + 1
        if indicateur in df.columns:
            box = go.Box(
                y=df[indicateur],
                boxmean=True,
                name=indicateur,
                marker=dict(color="blue"),
                showlegend=False
            )
            fig.add_trace(box, row=row, col=col)
    fig.update_layout(
        height=900, width=1200,
        title={'text': "Distribution des indicateurs (toutes années)", 'x': 0.5, 'xanchor': 'center'},
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


def dist_boxplot_indicateurs_annee_cercos(df):
    """
    Affiche des boxplots Plotly pour la distribution de plusieurs indicateurs Cercos, séparés par année.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes des indicateurs et la colonne 'Annee'.

    Returns:
        None. Affiche les graphiques directement dans Streamlit.
    """

    indicateurs = ["Etat_devolution_moy", 'Pjft_moyen', 'Nff_moyen', 'Nfr_moyen', 'Dp_moy', "Pjfn_moyen"]
    if "Annee" in df.columns:
        fig_year = make_subplots(rows=3, cols=2, subplot_titles=indicateurs)
        for i, indicateur in enumerate(indicateurs):
            row = (i // 2) + 1
            col = (i % 2) + 1
            if indicateur in df.columns:
                box = go.Box(
                    y=df[indicateur],
                    x=df["Annee"].astype(str),
                    boxmean=True,
                    name=indicateur,
                    marker=dict(color="blue"),
                    showlegend=False
                )
                fig_year.add_trace(box, row=row, col=col)
        fig_year.update_layout(
            height=900, width=1200,
            title={'text': "Distribution des indicateurs par année", 'x': 0.5, 'xanchor': 'center'},
            showlegend=False
        )
        st.plotly_chart(fig_year, use_container_width=True)
    else:
        st.info("Colonne 'Annee' absente de la base.")



def dist_boxplot_interactive_cercos(df, plot_boxp_interactive):
    """
    Affiche un boxplot interactif Plotly pour un indicateur Cercos sélectionné, par zone et par année.

    Args:
        df (pd.DataFrame): DataFrame contenant les indicateurs, la zone et l'année.
        plot_boxp_interactive (function): Fonction de génération du boxplot interactif.

    Returns:
        None. Affiche le graphique dans Streamlit.
    """
    indicateurs_interactifs = ['Nff_moyen', 'Nfr_moyen', 'Pjfn_moyen', 'Pjft_moyen', 'Etat_devolution_moy', 'Dp_moy']
    indicateur_choisi = st.selectbox("Sélectionner l'indicateur Cercos à visualiser :", indicateurs_interactifs, index=3)
    if "Zone_traitement" in df.columns and "Annee" in df.columns and indicateur_choisi in df.columns:
        try:
            fig_interactive = plot_boxp_interactive(
                df, "Zone_traitement", indicateur_choisi, hue="Annee", rotation=70,
                title=f"Distribution de {indicateur_choisi} par zone de traitement et par année"
            )
            if fig_interactive is not None:
                # Si la fonction retourne une figure Plotly
                import plotly.graph_objects as go
                if isinstance(fig_interactive, go.Figure):
                    st.plotly_chart(fig_interactive, use_container_width=True)
                else:
                    # Si la fonction affiche déjà le graphique, rien à faire
                    pass
            else:
                st.info("Aucun graphique généré par plot_boxp_interactive.")
        except Exception as e:
            st.error(f"Erreur lors de l'affichage du boxplot interactif : {e}")
    else:
        st.info("Colonnes nécessaires absentes de la base.")









############################ Base intrants #############################
def dist_ind_intr_zone(df):
    """
    Affiche des boxplots interactifs pour visualiser la distribution des indicateurs d'intrants
    (quantité, huile, jours entre traitements, etc.) par zone de traitement, avec sélection dynamique
    des variables et des zones via Streamlit.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes d'intrants et les informations de zone.

    Returns:
        None. Affiche les graphiques directement dans Streamlit.
    """
    st.subheader("Distribution des indicateurs Intrant par zone de traitement ")
    try:
        df_intrant = df.copy()
        import plotly.graph_objects as go
        import plotly.express as px
        # Variables disponibles
        variable_options = {
            "Quantité d'intrant": "Quantite",
            "Quantité d'huile": "Quantite_huile",
            "Nb jours entre 2 traitements": "Nb_jours_entre_2_traitements",
            "Nb jours 2mm tr": "Nb_jours_2mm_tr"
        }
        # Filtrer les variables présentes dans le DataFrame
        available_vars = {k: v for k, v in variable_options.items() if v in df_intrant.columns}
        # Sélection des variables à afficher
        selected_var = st.selectbox(
            "Choisir la variable à afficher :",
            list(available_vars.keys()),
            index=0
        )
        # Sélection des zones de traitement
        if "Zone_traitement" in df_intrant.columns:
            all_zones = sorted(df_intrant["Zone_traitement"].dropna().unique())
            # Ajout d'une option "Toutes les zones" pour faciliter la sélection
            zone_options = ["Toutes les zones"] + all_zones
            selected_zones = st.multiselect(
                "Choisir les zones de traitement à afficher :",
                zone_options,
                default=["Toutes les zones"]
            )
            # Si "Toutes les zones" est sélectionné ou rien n'est sélectionné, on prend tout
            if ("Toutes les zones" in selected_zones) or (not selected_zones):
                selected_zones = all_zones
            else:
                # On retire l'option "Toutes les zones" si elle est sélectionnée avec d'autres
                selected_zones = [z for z in selected_zones if z != "Toutes les zones"]
        else:
            st.warning("Colonne 'Zone_traitement' absente de la base.")
            selected_zones = []

        # Affichage des boxplots pour chaque variable sélectionnée
        if selected_var and selected_zones:
            var_col = available_vars[selected_var]
            # On veut aussi l'année et la semaine pour les outliers
            needed_cols = ["Zone_traitement", var_col]
            if "Annee" in df_intrant.columns:
                needed_cols.append("Annee")
            if "Semaine" in df_intrant.columns:
                needed_cols.append("Semaine")
            df_plot = df_intrant[df_intrant["Zone_traitement"].isin(selected_zones)][needed_cols].dropna(subset=[var_col])
            if df_plot.empty:
                st.info("Aucune donnée pour les zones sélectionnées.")
            else:
                # Boxplot par zone (couleur par zone) avec possibilité d'enlever les zones via la légende
                fig = go.Figure()
                color_map = px.colors.qualitative.Plotly
                zone_colors = {zone: color_map[i % len(color_map)] for i, zone in enumerate(selected_zones)}
                for i, zone in enumerate(selected_zones):
                    df_zone = df_plot[df_plot["Zone_traitement"] == zone]
                    fig.add_trace(go.Box(
                        y=df_zone[var_col],
                        name=str(zone),
                        marker_color=zone_colors[zone],
                        boxmean=True,
                        boxpoints='outliers',
                        line=dict(width=2),
                        width=0.5,
                        # Outliers customdata for hover
                        customdata=np.stack([
                            df_zone["Annee"].values if "Annee" in df_zone.columns else [None]*len(df_zone),
                            df_zone["Semaine"].values if "Semaine" in df_zone.columns else [None]*len(df_zone)
                        ], axis=-1) if ("Annee" in df_zone.columns and "Semaine" in df_zone.columns) else None,
                        hovertemplate=(
                            f"Zone: {zone}<br>{selected_var}: %{{y}}" +
                            ("<br>Année: %{customdata[0]}<br>Semaine: %{customdata[1]}" if ("Annee" in df_zone.columns and "Semaine" in df_zone.columns) else "")
                        ),
                        visible=True
                    ))
                fig.update_layout(
                    showlegend=True,
                    legend_title_text="Zone de traitement",
                    xaxis_title="Zone de traitement",
                    yaxis_title=selected_var,
                    boxmode='group',
                    height=500,
                    width=900,
                    margin=dict(l=40, r=20, t=60, b=120),
                    title=f"Distribution de {selected_var} par zone de traitement"
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=12))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Veuillez sélectionner une variable et au moins une zone de traitement.")
    except Exception as e:
        st.info(f"Impossible d'afficher les boxplots interactifs : {e}")


def dist_ind_intr_group(df, plot_intrant_moyen):
    """
    Affiche des graphiques interactifs (barplots) pour explorer la distribution des intrants agricoles
    (quantité d'intrant ou d'huile) selon différents regroupements et types d'agrégation, avec options
    dynamiques pour l'utilisateur dans Streamlit.

    L'utilisateur peut :
        - Choisir la variable à analyser (quantité d'intrant ou quantité d'huile)
        - Sélectionner le regroupement (par catégorie d'intrant ou par zone de traitement)
        - Choisir le type d'agrégation (moyenne ou somme totale)

    Selon l'agrégation choisie :
        - Moyenne : affiche un barplot de la valeur moyenne par groupe (via plot_intrant_moyen)
        - Somme totale : affiche un barplot de la somme totale par groupe et par année

    Args:
        df (pd.DataFrame): DataFrame contenant les données d'intrants, les catégories, zones et années.
        plot_intrant_moyen (function): Fonction externe pour générer le barplot des moyennes par groupe.

    Returns:
        None. Affiche les graphiques directement dans Streamlit.
    """
    # Options pour l'utilisateur
    value_options = {
        "Quantité d'intrant": ("Quantite", "Quantité d'intrant"),
        "Quantité d'huile": ("Quantite_huile", "Quantité d'huile")
    }
    group_options = {
        "Par catégorie d'intrant": ("Intrant_revu", "Catégorie d'intrant"),
        "Par zone de traitement": ("Zone_traitement", "Zone de traitement")
    }
    agg_options = {
        "Moyenne": "mean", 
        "Somme totale": "sum"
    }
    # Sélection utilisateur
    selected_value_label = st.radio(
        "Choisir la variable à afficher :",
        list(value_options.keys()),
        horizontal=True
    )
    selected_group_label = st.radio(
        "Grouper par :",
        list(group_options.keys()),
        horizontal=True
    )
    selected_agg_label = st.radio(
        "Type d'agrégation :",
        list(agg_options.keys()),
        horizontal=True
    )
    val_col, var_label = value_options[selected_value_label]

    if val_col == "Quantite":
        unite = "kg/ha"
    elif val_col == "Quantite_huile":
        unite = "L/ha"
    else:
        unite = ""

    cat_col, x_label = group_options[selected_group_label]
    agg_func = agg_options[selected_agg_label]
    # Vérification des colonnes nécessaires
    if cat_col in df.columns and val_col in df.columns and "Intrant_revu" in df.columns:
        df_bar = df[df["Intrant_revu"] != "Aucun traitement"].copy()
        if len(df_bar) == 0:
            st.info("Aucune donnée d'intrant hors 'Aucun traitement' à afficher.")
        else:
            # Calculer l'agrégation demandée
            if agg_func == "mean":
                y_label = f"{var_label} moyenne ({unite})"
                fig_bar = plot_intrant_moyen(
                    df_bar,
                    cat_col=cat_col,
                    val_col=val_col,
                    title=f"{y_label} par {x_label}",
                    x_label=x_label,
                    y_label=y_label
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                y_label = f"{var_label} totale ({unite})"
                # Calculer la somme par groupe
                df_sum = df_bar.groupby([cat_col, "Annee"])[val_col].sum().reset_index()
                fig_sum = px.bar(
                    df_sum,
                    x=cat_col,
                    y=val_col,
                    color="Annee",
                    text= val_col,
                    labels={cat_col: x_label, val_col: y_label},
                    title=f"{y_label} par {x_label}"
                )
                fig_sum.update_traces(texttemplate='%{text:.1f}', textposition='auto',textfont_size=10)
                st.plotly_chart(fig_sum, use_container_width=True)
    else:
        st.info(f"Colonnes '{cat_col}', '{val_col}' et/ou 'Intrant_revu' absentes de la base.")


def dist_occ_group(df, base, occurence_traitement):
    """
    Affiche un graphique interactif de l'occurrence des traitements agricoles selon différents regroupements,
    avec options dynamiques pour l'utilisateur dans Streamlit.

    L'utilisateur peut :
        - Choisir le regroupement de l'occurrence (par semaine ou par zone de traitement)
        - Visualiser l'occurrence des traitements hors "Aucun traitement" par année

    Args:
        df (pd.DataFrame): DataFrame contenant les données d'intrants, les zones, semaines et années.
        base (str): Type de base à analyser (doit être "Intrant" pour activer la fonction).
        occurence_traitement (function): Fonction externe générant le graphique d'occurrence.

    Returns:
        None. Affiche le graphique directement dans Streamlit.
    """
    # Options de sélection
    occ_group_options = {
        "Par semaine": ("Semaine", "Semaine", "Occurence du traitement par semaine et année"),
        "Par zone de traitement": ("Zone_traitement", "Zone de traitement", "Occurence du traitement par zone de traitement et année")
    }
    selected_occ_group = st.radio(
        "Grouper l'occurrence par :",
        list(occ_group_options.keys()),
        horizontal=True
    )
    zone_col, xlabel, occ_title = occ_group_options[selected_occ_group]
    # Vérification des colonnes nécessaires
    if base == "Intrant" and "Annee" in df.columns and zone_col in df.columns and "Intrant_revu" in df.columns:
        fig_occ = occurence_traitement(
            df,
            zone_col=zone_col,
            year_col="Annee",
            filter_col="Intrant_revu",
            filter_value="Aucun traitement",
            title=occ_title,
            xlabel=xlabel,
            ylabel="Occurence"
        )
        if fig_occ is not None:
            st.plotly_chart(fig_occ, use_container_width=True)
        else:
            st.info("Aucune donnée d'intrant hors 'Aucun traitement' à afficher pour l'occurrence.")
    else:
        st.info("Colonnes nécessaires absentes ou base non adaptée pour ce graphique.")



def dist_heatmap_group(df):
    """
    Affiche une heatmap interactive Plotly de la quantité totale d'intrant ou d'huile appliquée,
    croisée par zone de traitement et par année, avec annotations et palette de couleurs personnalisée.

    L'utilisateur peut :
        - Choisir la variable à afficher sur la heatmap (quantité d'intrant ou d'huile)
        - Visualiser la somme totale par zone et par année (hors "Aucun traitement")

    Fonctionnalités :
        - Les valeurs sont agrégées (somme) par zone et année.
        - Les valeurs sont annotées dans chaque case de la heatmap.
        - La palette de couleurs est adaptée pour une meilleure lisibilité.
        - Affichage direct dans Streamlit.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes 'Zone_traitement', 'Annee', la variable sélectionnée,
                           et 'Intrant_revu'.

    Returns:
        None. Affiche la heatmap dans Streamlit.
    """
    heatmap_value_options = {
        "Quantité d'intrant": ("Quantite", "Quantité totale d'intrant"),
        "Quantité d'huile": ("Quantite_huile", "Quantité totale d'huile")
    }
    selected_heatmap_value_label = st.radio(
        "Choisir la variable à afficher sur la heatmap :",
        list(heatmap_value_options.keys()),
        horizontal=True
    )
    heatmap_val_col, heatmap_var_label = heatmap_value_options[selected_heatmap_value_label]
    # Déduire l'unité
    if heatmap_val_col == "Quantite":
        unite = "kg/ha"
    elif heatmap_val_col == "Quantite_huile":
        unite = "L/ha"
    else:
        unite = ""

    # Vérification des colonnes nécessaires
    if (
        "Zone_traitement" in df.columns
        and "Annee" in df.columns
        and heatmap_val_col in df.columns
        and "Intrant_revu" in df.columns
    ):
        df_heatmap = df[df["Intrant_revu"] != "Aucun traitement"].copy()
        if len(df_heatmap) == 0:
            st.info("Aucune donnée d'intrant hors 'Aucun traitement' à afficher pour la heatmap.")
        else:
            # Calcul du pivot pour la heatmap
            heatmap_pivot = df_heatmap.pivot_table(
                index="Zone_traitement",
                columns="Annee",
                values=heatmap_val_col,
                aggfunc="sum",
                fill_value=0
            )
            # Affichage avec plotly.graph_objects pour personnaliser les couleurs et les annotations
            z = heatmap_pivot.values
            x = heatmap_pivot.columns.astype(str)
            y = heatmap_pivot.index.astype(str)
            # Palette custom YlOrRd adoucie (max moins foncé)
            colorscale = [
                [0.0, "#ffffe5"],   # blanc-jaune très clair
                [0.2, "#ffe08b"],   # jaune pâle
                [0.4, "#fec44f"],   # jaune-orangé
                [0.6, "#fe9929"],   # orange
                [0.8, "#ec7014"],   # orange foncé
                [1.0, "#f8310e"]    # rouge-orangé plus clair (adouci)
            ]
            # Annotations (valeurs dans chaque case)
            annotations = []
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    annotations.append(
                        dict(
                            x=x[j],
                            y=y[i],
                            text=f"{z[i, j]:.0f}",
                            font=dict(color="black", size=13),
                            showarrow=False
                        )
                    )
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale=colorscale,
                colorbar=dict(title=heatmap_var_label),
                hovertemplate="Zone: %{y}<br>Année: %{x}<br>Quantité: %{z}<extra></extra>",
                zmin=0
            ))
            fig_heatmap.update_layout(
                title=f"{heatmap_var_label} par zone de traitement et année ({unite})",
                xaxis_title="Année",
                yaxis_title="Zone de traitement",
                annotations=annotations,
                height=600,
                width=950,
                margin=dict(l=40, r=20, t=60, b=120)
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Colonnes nécessaires absentes pour la heatmap.")






##################################################### Base Météo #####################################################
def dist_indic_meteo(df_meteo):
    """
    Affiche des boxplots Plotly pour visualiser la distribution des principaux indicateurs météorologiques.

    L'utilisateur peut :
        - Afficher la distribution de tous les indicateurs météo en même temps (sous-graphiques)
        - Ou sélectionner un seul indicateur météo à visualiser

    Fonctionnalités :
        - Les indicateurs pris en compte sont : Pluie, Humidité max/min, Température max/min/moyenne
        - Seules les colonnes présentes et non vides sont affichées
        - Affichage interactif et mise en forme adaptée pour Streamlit

    Args:
        df_meteo (pd.DataFrame): DataFrame contenant les colonnes météo à visualiser.

    Returns:
        None. Affiche les graphiques directement dans Streamlit.
    """
    indicateurs_meteo = [
        'Pluie_sum', 'Humidite_max_mean', 'Humidite_min_mean',
        'Tmax_mean', 'Tmin_mean', 'Tmoy_mean'
    ]

    # Filtrer les indicateurs présents et numériques
    present_cols = [col for col in indicateurs_meteo if col in df_meteo.columns and df_meteo[col].dropna().size > 0]
    if not present_cols:
        st.warning("Aucun indicateur météo valide pour la distribution.")
        return

    # Option d'affichage : tous ou un seul indicateur à la fois
    choix = st.radio(
        "Afficher la distribution de :", 
        ["Tous les indicateurs", "Un seul indicateur"]
    )

    if choix == "Un seul indicateur":
        indicateur = st.selectbox("Choisir l'indicateur météo", present_cols)
        fig = go.Figure()
        fig.add_trace(
            go.Box(
                y=df_meteo[indicateur],
                name=indicateur,
                boxmean=True,
                fillcolor='rgba(31, 119, 180, 0.6)',
                line=dict(color='rgba(31, 119, 180, 1)', width=1.5),
                marker=dict(color='rgba(31, 119, 180, 1)', size=5),
                whiskerwidth=0.8
            )
        )
        fig.update_layout(
            height=500,
            width=700,
            title_text=f"Distribution de {indicateur}",
            title_x=0.5,
            title_xanchor="center",
            showlegend=False,
            template='plotly_white',
            plot_bgcolor='white'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Création d'une figure avec des sous-graphiques
        n = len(present_cols)
        rows = 2
        cols = 3
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"Distribution de {ind}" for ind in present_cols]
        )

        for i, indicateur in enumerate(present_cols):
            row = i // cols + 1
            col = i % cols + 1
            fig.add_trace(
                go.Box(
                    y=df_meteo[indicateur],
                    name=indicateur,
                    boxmean=True,
                    fillcolor='rgba(31, 119, 180, 0.6)',
                    line=dict(color='rgba(31, 119, 180, 1)', width=1.5),
                    marker=dict(color='rgba(31, 119, 180, 1)', size=5),
                    whiskerwidth=0.8
                ),
                row=row, col=col
            )

        fig.update_layout(
            height=800,
            width=2200,
            title_text="Distribution des indicateurs météorologiques",
            title_x=0.5,
            showlegend=False,
            template='plotly_white',
            plot_bgcolor='white'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
        st.plotly_chart(fig, use_container_width=True)



def dist_indic_meteo_grouped(df_meteo):
    """
    Affiche des boxplots Plotly pour visualiser la distribution des indicateurs météorologiques,
    groupés selon différents critères sélectionnables par l'utilisateur dans Streamlit.

    L'utilisateur peut :
        - Grouper la distribution par année, par intensité de pluie, ou par année + intensité de pluie.
        - Visualiser la distribution de plusieurs indicateurs météo (pluie, humidité, températures).

    Fonctionnalités :
        - Affiche des sous-graphiques pour chaque indicateur météo.
        - Utilise des couleurs distinctes pour chaque groupe (année ou intensité de pluie).
        - Affichage interactif et mise en forme adaptée pour Streamlit.
        - Légende dynamique selon le regroupement choisi.

    Args:
        df_meteo (pd.DataFrame): DataFrame contenant les colonnes météo à visualiser,
                                 ainsi que 'Annee' et/ou 'Intensite_pluie'.

    Returns:
        None. Affiche les graphiques directement dans Streamlit.
    """

    indicateurs_meteo = [
        'Pluie_sum', 'Humidite_max_mean', 'Humidite_min_mean',
        'Tmax_mean', 'Tmin_mean', 'Tmoy_mean'
    ]
    
    groupby_option = st.radio(
        "Grouper la distribution par :",
        ("Année", "Intensité de pluie", "Année + Intensité de pluie")
    )

    if groupby_option == "Année":
        annees = df_meteo['Annee'].unique()
        colors = {
            2023: 'rgb(31, 119, 180)',  # Bleu
            2024: 'rgb(255, 127, 14)',  # Orange
            2025: 'rgb(44, 160, 44)'    # Vert
        }
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f"Distribution de {ind}" for ind in indicateurs_meteo],
            horizontal_spacing=0.07, vertical_spacing=0.12
        )
        for i, indicateur in enumerate(indicateurs_meteo):
            row = i // 3 + 1
            col = i % 3 + 1
            for annee in annees:
                df_annee = df_meteo[df_meteo['Annee'] == annee]
                fig.add_trace(
                    go.Box(
                        y=df_annee[indicateur],
                        name=str(annee),
                        boxmean=True,
                        showlegend=(i == 0),
                        legendgroup=str(annee),
                        marker_color=colors.get(annee, 'gray'),
                        line=dict(color=colors.get(annee, 'gray'))
                    ),
                    row=row, col=col
                )
        fig.update_layout(
            height=700,
            width=1200,
            title_text="Distribution des indicateurs météorologiques par année",
            title_x=0.5,
            title_xanchor="center",
            template='plotly_white',
            legend_title_text='Année',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.18,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=40, r=40, t=60, b=100),
        )
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        st.plotly_chart(fig, use_container_width=True)

    elif groupby_option == "Intensité de pluie":
        intensites = df_meteo['Intensite_pluie'].unique()
        colors = {
            'Calme': 'rgb(31, 119, 180)',
            'Faible': 'rgb(44, 160, 44)',
            'Forte': 'rgb(255, 127, 14)',
            'Très forte': 'rgb(214, 39, 40)',
            'Alerte': 'rgb(148, 0, 211)'
        }
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f"Distribution de {ind}" for ind in indicateurs_meteo],
            horizontal_spacing=0.07, vertical_spacing=0.12
        )
        for i, indicateur in enumerate(indicateurs_meteo):
            row = i // 3 + 1
            col = i % 3 + 1
            for intensite in intensites:
                df_intensite = df_meteo[df_meteo['Intensite_pluie'] == intensite]
                fig.add_trace(
                    go.Box(
                        y=df_intensite[indicateur],
                        name=str(intensite),
                        boxmean=True,
                        showlegend=(i == 0),
                        legendgroup=str(intensite),
                        marker_color=colors.get(intensite, 'gray'),
                        line=dict(color=colors.get(intensite, 'gray'))
                    ),
                    row=row, col=col
                )
        fig.update_layout(
            height=700,
            width=1200,
            title_text="Distribution des indicateurs météorologiques par intensité de pluie",
            title_x=0.5,
            title_xanchor="center",
            template='plotly_white',
            legend_title_text='Intensité de pluie',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.18,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=40, r=40, t=60, b=100),
        )
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        st.plotly_chart(fig, use_container_width=True)

    else:  # "Année + Intensité de pluie"
        colors = {
            'Calme': 'rgb(31, 119, 180)',
            'Faible': 'rgb(44, 160, 44)',
            'Forte': 'rgb(255, 127, 14)',
            'Très forte': 'rgb(214, 39, 40)',
            'Alerte': 'rgb(148, 0, 211)'
        }
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f"{ind}" for ind in indicateurs_meteo],
            horizontal_spacing=0.08, vertical_spacing=0.15
        )
        for i, indicateur in enumerate(indicateurs_meteo):
            row = i // 3 + 1
            col = i % 3 + 1
            temp_fig = px.box(
                df_meteo,
                x="Annee",
                y=indicateur,
                color="Intensite_pluie",
                category_orders={"Annee": sorted(df_meteo["Annee"].unique())},
                color_discrete_map=colors,
                points="outliers"
            )
            for trace in temp_fig.data:
                trace.showlegend = (i == 0)  # Légende seulement sur le 1er subplot
                trace.legendgroup = trace.name
                fig.add_trace(trace, row=row, col=col)
            fig.update_xaxes(title_text="Année", row=row, col=col, tickangle=45)
            fig.update_yaxes(title_text=indicateur, row=row, col=col)
        fig.update_layout(
            height=800,
            width=1600,
            title_text="Distribution des indicateurs météorologiques par année et Intensité de pluie",
            title_x=0.5,
            title_xanchor="center",
            boxmode="group",  # <<<<<< clé pour affichage groupé
            legend_title="Intensité de pluie",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.17,
                xanchor="center",
                x=0.5,
                font=dict(size=15),
                bgcolor='rgba(0,0,0,0)'
            ),
            plot_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=160)
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
        st.plotly_chart(fig, use_container_width=True)


###################################################### Page prédiction ######################################################

def visualiser_importance_variables_streamlit(
    base_path,
    target_col_name,
    zone,
    is_future=False,
    n_top=25,
    key=None,
    horizon=None
):
    """
    Visualise les variables les plus importantes du modèle pour une zone donnée dans Streamlit.
    - Affiche les 25 variables les plus importantes sous forme de graphique interactif.
    - Utilise importance.csv ou importance_futur.csv selon le paramètre is_future.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if horizon is not None:
        save_dir = os.path.join(project_root, "data", f"models_data_{horizon}", f"models_{target_col_name}_{zone}")
    else:
        save_dir = os.path.join(project_root, "data", "models_data", f"models_{target_col_name}_{zone}")
    importance_filename = "importance_futur.csv" if is_future else "importance.csv"
    importance_path = os.path.join(save_dir, importance_filename)
    
    if not os.path.exists(importance_path):
        st.warning(f"Le fichier '{importance_filename}' n'est pas présent pour la zone {zone}.")
        return

    importance_df = pd.read_csv(importance_path)
    if importance_df.empty or 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
        st.error("Le fichier d'importance n'est pas valide ou ne contient pas les colonnes attendues.")
        return

    # Prendre les n_top variables les plus importantes
    top_df = importance_df.sort_values(by="importance", ascending=False).head(n_top)

    fig = px.bar(
        top_df,
        x="importance",
        y="feature",
        orientation='h',
        title=f"Top {n_top} variables les plus importantes<br>Zone: {zone} | Modèle {'futur' if is_future else 'test'}",
        labels={"importance": "Importance", "feature": "Variable"},
        color="importance",
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=500,
        margin=dict(l=80, r=30, t=60, b=70),
        title_x=0.3
    )
    
    st.plotly_chart(fig, use_container_width=True)


def visualiser_zone_streamlit_plotly(
    base_path,
    target_col_name,
    zone,
    pred_col='y_pred',  
    train_pred_filename='predictions_train.csv',
    test_filename='predictions.csv',
    step_ticks=10,
    horizon= None
):
    """
    Visualise dans Streamlit (avec Plotly) l'évolution temporelle de la variable cible et des prédictions
    pour une zone donnée, en affichant à la fois les résultats sur le train et le test.

    Affiche :
        - L'historique réel (train) et ses prédictions.
        - Les valeurs réelles (test) et les prédictions associées.
        - Les intervalles de confiance 95% si disponibles.
        - Une ligne verticale pour le début de la période de test.
        - Les métriques de performance (RMSE, MAE) pour train et test dans le titre.

    Fonctionnalités :
        - Lecture automatique des fichiers de prédiction pour la zone et la cible.
        - Gestion des erreurs de fichiers ou de colonnes manquantes.
        - Affichage interactif et mise en forme adaptée pour Streamlit.

    Args:
        base_path (str): Chemin de base du projet.
        target_col_name (str): Nom de la variable cible.
        zone (str): Identifiant de la zone à visualiser.
        pred_col (str, optionnel): Nom de la colonne de prédiction (défaut 'y_pred').
        train_pred_filename (str, optionnel): Nom du fichier CSV des prédictions sur le train.
        test_filename (str, optionnel): Nom du fichier CSV des prédictions sur le test.
        step_ticks (int, optionnel): Espacement des dates affichées sur l'axe X.

    Returns:
        None. Affiche le graphique interactif directement dans Streamlit.
    """    

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if horizon is not None:
        save_dir = os.path.join(project_root, "data", f"models_data_{horizon}", f"models_{target_col_name}_{zone}")
    else:
        save_dir = os.path.join(project_root, "data", "models_data", f"models_{target_col_name}_{zone}")
    train_pred_path = os.path.join(save_dir, train_pred_filename)
    test_path = os.path.join(save_dir, test_filename)

    if not os.path.exists(train_pred_path):
        st.warning(f"Le fichier '{os.path.basename(train_pred_path)}' n'est pas présent pour la zone {zone}. Merci de lancer le bouton de test du modèle (test.py) en bas de la page.")
        return
    if not os.path.exists(test_path):
        st.warning(f"Le fichier '{os.path.basename(test_path)}' n'est pas présent pour la zone {zone}. Merci de lancer le bouton de test du modèle (test.py) en bas de la page.")
        return

    train_pred_df = pd.read_csv(train_pred_path)
    test_df = pd.read_csv(test_path)

    if 'model_name' in test_df.columns:
        model_name = str(test_df['model_name'].iloc[0])
    else:
        model_name = "N/A"

    # Vérification colonne ds
    for df_name, df in zip(['train_pred_df', 'test_df'], [train_pred_df, test_df]):
        if "ds" not in df.columns:
            st.error(f"Colonne 'ds' manquante dans {df_name}.")
            return

    # Conversion en datetime pour l'axe x
    train_pred_df['ds'] = pd.to_datetime(train_pred_df['ds'])
    test_df['ds'] = pd.to_datetime(test_df['ds'])

    # Colonne y_pred sur train
    if pred_col not in train_pred_df.columns:
        pred_cols_pred_train = [c for c in train_pred_df.columns if c not in ['unique_id', 'ds', 'Annee', 'Semaine', 'y']]
        if len(pred_cols_pred_train) == 1:
            pred_col_pred_train = pred_cols_pred_train[0]
        else:
            st.error(f"Colonne de prédiction non trouvée dans {list(train_pred_df.columns)}")
            return
    else:
        pred_col_pred_train = pred_col

    try:
        rmse_train = mean_squared_error(train_pred_df['y'], train_pred_df[pred_col_pred_train], squared=False)
        mae_train = mean_absolute_error(train_pred_df['y'], train_pred_df[pred_col_pred_train])
    except Exception as e:
        st.error(f"Erreur calcul métriques train : {e}")
        return

    # pour le test
    if pred_col not in test_df.columns:
        pred_cols = [c for c in test_df.columns if c not in ['unique_id', 'ds', 'Annee', 'Semaine', 'y']]
        if len(pred_cols) == 1:
            pred_col = pred_cols[0]
        else:
            st.error(f"Colonne de prédiction non trouvée dans {list(test_df.columns)}")
            return

    try:
        y_pred = test_df[pred_col]
        y_true = test_df['y']
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
    except Exception as e:
        st.error(f"Erreur calcul métriques test : {e}")
        return

    fig = go.Figure()

    # Courbe historique train (y), issue de predictions_train.csv uniquement
    fig.add_trace(go.Scatter(
        x=train_pred_df['ds'],
        y=train_pred_df['y'],
        mode='lines+markers',
        name='Historique (train)',
        line=dict(color='#888888', width=1.5),  
        marker=dict(symbol='circle', size=4),   
        opacity=0.7,
        hovertemplate='ds: %{x}<br>y: %{y}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=train_pred_df['ds'],
        y=train_pred_df[pred_col_pred_train],
        mode='lines+markers',
        name='Prédiction (train)',
        line=dict(color='orange', width=1.5, dash='dash'),  
        marker=dict(symbol='square', size=4), 
        hovertemplate='ds: %{x}<br>y_pred: %{y}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=test_df['ds'],
        y=y_true,
        mode='lines+markers',
        name='Réel (test)',
        line=dict(color='blue', width=1.5),   
        marker=dict(symbol='circle', size=4), 
        hovertemplate='ds: %{x}<br>y: %{y}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=test_df['ds'],
        y=y_pred,
        mode='lines+markers',
        name='Prédiction (test)',
        line=dict(color='red', width=1.5, dash='dash'),  
        marker=dict(symbol='square', size=4),            
        hovertemplate='ds: %{x}<br>y_pred: %{y}<extra></extra>'
    ))

    if 'ic_hi_95' in test_df.columns and 'ic_lo_95' in test_df.columns:
        fig.add_trace(go.Scatter(
            x=test_df['ds'],
            y=test_df['ic_hi_95'],
            mode='lines',
            name="IC 95% haut",
            line=dict(color='black', dash='dot', width=1),
            hovertemplate='ds: %{x}<br>IC 95% haut: %{y}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=test_df['ds'],
            y=test_df['ic_lo_95'],
            mode='lines',
            name="IC 95% bas",
            line=dict(color='black', dash='dot', width=1),
            hovertemplate='ds: %{x}<br>IC 95% bas: %{y}<extra></extra>'
        ))

    # Ligne verticale début test
    fig.add_shape(
        type="line",
        x0=test_df['ds'].iloc[0],
        x1=test_df['ds'].iloc[0],
        y0=0,
        y1=1,
        xref='x',
        yref='paper',
        line=dict(color='green', width=1, dash='dot')
    )
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        name='Début test',
        line=dict(color='green', width=1, dash='dot'),
        showlegend=True
    ))

    # -- ESPACEMENT DES DATES SUR L'AXE X avec "ds" --
    x_all = (
        list(train_pred_df['ds'].unique()) +
        [x for x in test_df['ds'].unique() if x not in train_pred_df['ds'].unique()]
    )
    x_all = sorted(set(x_all))
    tickvals = x_all[::step_ticks]
    ticktext = [x.strftime('%Y-%m-%d') for x in tickvals]

    fig.update_layout(
        title={
            'text': (
            f"<b>Zone {zone} | Modèle: {model_name}</b><br>"
            f"<span style='display:block; font-size:12px; font-weight:normal;'>"
            f"Train - RMSE: {rmse_train:.3f} | MAE: {mae_train:.3f}&nbsp; | &nbsp;"
            f"Test - RMSE: {rmse:.3f} | MAE: {mae:.3f}"
            f"</span>"
        ),
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        title_font_size=22,
        xaxis_title='Date',
        yaxis_title=target_col_name,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.42,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            title=dict(font=dict(size=13))
        ),
        hovermode='x unified',
        margin=dict(l=40, r=30, t=60, b=110),
        width=1200,
        height=550
    )
    fig.update_xaxes(
        tickangle=45,
        tickfont=dict(size=11),
        type='date',
        tickvals=tickvals,
        ticktext=ticktext
    )
    fig.update_yaxes(tickfont=dict(size=12), gridcolor='rgba(0,0,0,0.1)')

    st.plotly_chart(fig, use_container_width=False)

 
def visualiser_futur_zone_streamlit_plotly(
    base_path,
    target_col_name,
    zone,
    futur_pred_filename='predictions_futur.csv',
    train_pred_futur_filename='predictions_train_futur.csv',
    pred_col='y_pred',
    step_ticks=10,
    horizon=None
):
    """
    Visualise dans Streamlit (avec Plotly) l'historique utilisé pour la prédiction future
    et les prédictions futures pour une zone de traitement donnée.

    Affiche :
        - L'historique réel (train) et ses prédictions (issues de predictions_train_futur.csv).
        - Les prédictions futures (issues de predictions_futur.csv).
        - Les intervalles de confiance 95% si disponibles.
        - Une ligne verticale pour le début de la période de prévision future.
        - Les métriques de performance (RMSE, MAE) pour le train dans le titre.

    Fonctionnalités :
        - Lecture automatique des fichiers de prédiction pour la zone et la cible.
        - Gestion des erreurs de fichiers ou de colonnes manquantes.
        - Affichage interactif et mise en forme adaptée pour Streamlit.
        - Pas de vérité connue pour le futur, seules les prédictions sont affichées.

    Args:
        base_path (str): Chemin de base du projet.
        target_col_name (str): Nom de la variable cible.
        zone (str): Identifiant de la zone à visualiser.
        futur_pred_filename (str, optionnel): Nom du fichier CSV des prédictions futures.
        train_pred_futur_filename (str, optionnel): Nom du fichier CSV du train utilisé pour la prédiction future.
        pred_col (str, optionnel): Nom de la colonne de prédiction (défaut 'y_pred').
        step_ticks (int, optionnel): Espacement des dates affichées sur l'axe X.

    Returns:
        None. Affiche le graphique interactif directement dans Streamlit.
    """

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if horizon is not None:
        save_dir = os.path.join(project_root, "data", f"models_data_{horizon}", f"models_{target_col_name}_{zone}")
    else:
        save_dir = os.path.join(project_root, "data", "models_data", f"models_{target_col_name}_{zone}")
    train_pred_futur_path = os.path.join(save_dir, train_pred_futur_filename)
    fut_pred_path = os.path.join(save_dir, futur_pred_filename)

    if not os.path.exists(train_pred_futur_path):
        st.warning(f"Le fichier '{os.path.basename(train_pred_futur_path)}' n'est pas présent pour la zone {zone}. Merci de lancer le bouton de prédiction future (model_future.py) en bas de la page.")
        return
    if not os.path.exists(fut_pred_path):
        st.warning(f"Le fichier '{os.path.basename(fut_pred_path)}' n'est pas présent pour la zone {zone}. Merci de lancer le bouton de prédiction future (model_future.py) en bas de la page.")
        return
    
    train_pred_futur_df = pd.read_csv(train_pred_futur_path)
    fut_pred_df = pd.read_csv(fut_pred_path)
    if 'model_name' in fut_pred_df.columns:
        model_name = str(fut_pred_df['model_name'].iloc[0])
    else:
        model_name = "N/A"

    # Vérification colonne ds
    for df_name, df in zip(['train_pred_futur_df', 'fut_pred_df'], [train_pred_futur_df, fut_pred_df]):
        if "ds" not in df.columns:
            st.error(f"Colonne 'ds' manquante dans {df_name}.")
            return

    # Conversion en datetime pour l'axe x
    train_pred_futur_df['ds'] = pd.to_datetime(train_pred_futur_df['ds'])
    fut_pred_df['ds'] = pd.to_datetime(fut_pred_df['ds'])

    # Colonne y_pred sur train_pred_futur
    if pred_col not in train_pred_futur_df.columns:
        pred_cols_pred_train = [c for c in train_pred_futur_df.columns if c not in ['unique_id', 'ds', 'Annee', 'Semaine', 'y']]
        if len(pred_cols_pred_train) == 1:
            pred_col_pred_train = pred_cols_pred_train[0]
        else:
            st.error(f"Colonne de prédiction non trouvée dans {list(train_pred_futur_df.columns)}")
            return
    else:
        pred_col_pred_train = pred_col

    # Calcul des métriques train (uniquement sur le train, pas sur le futur)
    try:
        rmse_train = mean_squared_error(train_pred_futur_df['y'], train_pred_futur_df[pred_col_pred_train], squared=False)
        mae_train = mean_absolute_error(train_pred_futur_df['y'], train_pred_futur_df[pred_col_pred_train])
    except Exception as e:
        st.error(f"Erreur calcul métriques train : {e}")
        return

    # Pour le futur, pas de vérité connue
    fig = go.Figure()

    # Historique (train)
    fig.add_trace(go.Scatter(
        x=train_pred_futur_df['ds'],
        y=train_pred_futur_df['y'],
        mode='lines+markers',
        name='Historique (train)',
        line=dict(color='#888888', width=1.5),
        marker=dict(symbol='circle', size=4),
        opacity=0.7,
        hovertemplate='ds: %{x}<br>y: %{y}<extra></extra>'
    ))

    # Prédiction (train)
    fig.add_trace(go.Scatter(
        x=train_pred_futur_df['ds'],
        y=train_pred_futur_df[pred_col_pred_train],
        mode='lines+markers',
        name='Prédiction (train)',
        line=dict(color='orange', width=1.5, dash='dash'),
        marker=dict(symbol='square', size=4),
        hovertemplate='ds: %{x}<br>y_pred: %{y}<extra></extra>'
    ))

    # Prédiction (futur)
    if pred_col not in fut_pred_df.columns:
        pred_cols = [c for c in fut_pred_df.columns if c not in ['unique_id', 'ds', 'Annee', 'Semaine', 'y_true', 'type_prediction']]
        if len(pred_cols) == 1:
            pred_col_futur = pred_cols[0]
        else:
            st.error(f"Colonne de prédiction non trouvée dans {list(fut_pred_df.columns)}")
            return
    else:
        pred_col_futur = pred_col

    fig.add_trace(go.Scatter(
        x=fut_pred_df['ds'],
        y=fut_pred_df[pred_col_futur],
        mode='lines+markers',
        name='Prédiction (futur)',
        line=dict(color='red', width=1.5, dash='dash'),
        marker=dict(symbol='square', size=4),
        hovertemplate='ds: %{x}<br>y_pred: %{y}<extra></extra>'
    ))

    # Ajout intervalles de confiance 90% si dispo
    if 'ic_hi_95' in fut_pred_df.columns and 'ic_lo_95' in fut_pred_df.columns:
        fig.add_trace(go.Scatter(
            x=fut_pred_df['ds'],
            y=fut_pred_df['ic_hi_95'],
            mode='lines',
            name="IC 95% haut",
            line=dict(color='black', dash='dot', width=1),
            hovertemplate='ds: %{x}<br>IC 95% haut: %{y}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=fut_pred_df['ds'],
            y=fut_pred_df['ic_lo_95'],
            mode='lines',
            name="IC 95% bas",
            line=dict(color='black', dash='dot', width=1),
            hovertemplate='ds: %{x}<br>IC 95% bas: %{y}<extra></extra>'
        ))

    # Ligne verticale début prévision future
    fig.add_shape(
        type="line",
        x0=fut_pred_df['ds'].iloc[0],
        x1=fut_pred_df['ds'].iloc[0],
        y0=0,
        y1=1,
        xref='x',
        yref='paper',
        line=dict(color='green', width=1, dash='dot')
    )
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        name='Début prévision future',
        line=dict(color='green', width=1, dash='dot'),
        showlegend=True
    ))

    # -- ESPACEMENT DES DATES SUR L'AXE X avec "ds" --
    x_all = (
        list(train_pred_futur_df['ds'].unique()) +
        [x for x in fut_pred_df['ds'].unique() if x not in train_pred_futur_df['ds'].unique()]
    )
    x_all = sorted(set(x_all))
    tickvals = x_all[::step_ticks]
    ticktext = [x.strftime('%Y-%m-%d') for x in tickvals]

    fig.update_layout(
        title={
            'text': (
            f"<b>Zone {zone} | Modèle: {model_name}</b><br>"
            f"<span style='display:block; font-size:12px; font-weight:normal;'>"
            f"Train - RMSE: {rmse_train:.3f} | MAE: {mae_train:.3f} &nbsp; | &nbsp; "
            f"Prévisions futures (pas de vérité connue)"
            f"</span>"
        ),
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
        title_font_size=22,
        xaxis_title='Date',
        yaxis_title=target_col_name,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.42,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            title=dict(font=dict(size=13))
        ),
        hovermode='x unified',
        margin=dict(l=40, r=30, t=60, b=110),
        width=1200,
        height=550
    )
    fig.update_xaxes(
        tickangle=45,
        tickfont=dict(size=11),
        type='date',
        tickvals=tickvals,
        ticktext=ticktext
    )
    fig.update_yaxes(tickfont=dict(size=12), gridcolor='rgba(0,0,0,0.1)')

    st.plotly_chart(fig, use_container_width=False)


def plot_indicator_alert(df_train, df_futur, indicateur, seuil_val):
    """
    Affiche dans Plotly l'évolution d'un indicateur avec alertes et intervalles de confiance, pour l'historique et les prédictions futures.

    Cette fonction trace :
        - L'historique réel (train) de l'indicateur.
        - Les points d'alerte historiques (dépassement du seuil).
        - Les prédictions futures (y_pred).
        - Les points d'alerte futurs (dépassement du seuil sur la borne IC correspondante).
        - Les intervalles de confiance 95% (haut et bas) sous forme de lignes et de zone remplie.
        - La ligne de seuil sur toute la période.

    La logique d'alerte dépend de l'indicateur :
        - Pour "Dp_moy" et "Etat_devolution_moy" : alerte si valeur > seuil.
        - Pour les autres indicateurs : alerte si valeur < seuil.

    Args:
        df_train (pd.DataFrame): Historique d'apprentissage, doit contenir 'ds', 'y' et éventuellement 'ic_hi_95', 'ic_lo_95'.
        df_futur (pd.DataFrame): Prédictions futures, doit contenir 'ds', 'y_pred' et éventuellement 'ic_hi_95', 'ic_lo_95'.
        indicateur (str): Nom de l'indicateur à tracer.
        seuil_val (float): Valeur du seuil d'alerte.

    Returns:
        tuple: (fig, points_alerte_train, points_alerte_futur)
            - fig (go.Figure): Figure Plotly générée.
            - points_alerte_train (pd.DataFrame): Points historiques en alerte.
            - points_alerte_futur (pd.DataFrame): Points futurs en alerte.
    """

    # Copie pour ne pas modifier les DataFrames d'origine
    df_train = df_train.copy()
    df_futur = df_futur.copy()

    fig = go.Figure()
    
    # Définir les alertes (logique différente selon l'indicateur)
    if indicateur in ["Dp_moy", "Etat_devolution_moy"]:
        df_train['Alerte'] = df_train['y'] > seuil_val
        if 'ic_hi_95' in df_futur.columns:
            df_futur['Alerte'] = df_futur['ic_hi_95'] > seuil_val
        else:
            df_futur['Alerte'] = False
        alert_label_ic = "Alerte IC (haut)"
        alerte_ic_y_col = 'ic_hi_95'
    else:
        df_train['Alerte'] = df_train['y'] < seuil_val
        if 'ic_lo_95' in df_futur.columns:
            df_futur['Alerte'] = df_futur['ic_lo_95'] < seuil_val
        else:
            df_futur['Alerte'] = False
        alert_label_ic = "Alerte IC (bas)"
        alerte_ic_y_col = 'ic_lo_95'
    
    # 1. Historique
    fig.add_trace(go.Scatter(
        x=df_train['ds'],
        y=df_train['y'],
        mode='lines+markers',
        name='Historique',
        line=dict(color='blue', width=1.5)
    ))
    
    # 2. Alertes historiques
    points_alerte_train = df_train[df_train['Alerte']]
    if not points_alerte_train.empty:
        fig.add_trace(go.Scatter(
            x=points_alerte_train['ds'],
            y=points_alerte_train['y'],
            mode='markers',
            name='Alerte (hist.)',
            marker=dict(color='red', size=6)
        ))
    
    # 3. Prédiction future
    fig.add_trace(go.Scatter(
        x=df_futur['ds'],
        y=df_futur['y_pred'],
        mode='lines+markers',
        name='Prédiction',
        line=dict(color='indigo', width=1.5)
    ))
    
    # 4. Points d'alerte futurs (sur la borne concernée)
    points_alerte_futur = df_futur[df_futur['Alerte']]
    if not points_alerte_futur.empty and alerte_ic_y_col in df_futur.columns:
        fig.add_trace(go.Scatter(
            x=points_alerte_futur['ds'],
            y=points_alerte_futur[alerte_ic_y_col],
            mode='markers',
            name=alert_label_ic,
            marker=dict(color='orange', size=8, symbol='diamond')
        ))
    
    # 5. Intervalles de confiance : deux traces visibles + zone remplie
    if 'ic_hi_95' in df_futur.columns and 'ic_lo_95' in df_futur.columns:
        # Borne haute
        fig.add_trace(go.Scatter(
            x=df_futur['ds'],
            y=df_futur['ic_hi_95'],
            mode='lines',
            name='IC 95% haut',
            line=dict(color='gray', dash='dot')
        ))
        # Borne basse (remplissage vers la précédente)
        fig.add_trace(go.Scatter(
            x=df_futur['ds'],
            y=df_futur['ic_lo_95'],
            mode='lines',
            name='IC 95% bas',
            line=dict(color='gray', dash='dot'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.25)'
        ))
    
    # 6. Seuil
    fig.add_trace(go.Scatter(
        x=[df_train['ds'].min(), df_futur['ds'].max()],
        y=[seuil_val, seuil_val],
        mode='lines',
        name=f'Seuil {seuil_val:.2f}',
        line=dict(color='red', dash='dash')
    ))
    
    # Mise en page
    fig.update_layout(
        title=f"Évolution et prévision de {indicateur}",
        xaxis_title='Date',
        yaxis_title=indicateur,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            traceorder="normal"
        ),
        margin=dict(l=40, r=30, t=80, b=40)
    )
    
    return fig, points_alerte_train, points_alerte_futur