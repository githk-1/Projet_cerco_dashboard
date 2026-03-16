import sys
import os
# Ajoute le dossier parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import streamlit as st
import pandas as pd
from src.preparation_base import *
from src.plot_fonction import *
from src.analyse_fonction import *
from src.model_fonction import *
from src.streamlit_func import *
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go

st.set_page_config(layout="wide")


st.markdown("""
    <h1 style='text-align:center; margin-bottom:1rem;'>Système d'alerte multi-indicateur</h1>
    <div style='
        background-color: #e3f0fc;
        border: 1px solid #b6d4f4;
        padding: 18px 28px;
        margin-bottom: 32px;
        font-size: 18px;
        color: #1e293b;
        width: 100%;
        text-align: center;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(30,41,59,0.03);
    '>
        Vous pouvez ajuster les seuils dans la barre latérale,
        les périodes en <span style="color:#b91c1c; font-weight:bold;">alerte</span> sont surlignées en rouge.<br>
        <b>• Sous le seuil</b> (alerte) pour tous les indicateurs sauf <b>Dp_moy</b> et <b>Etat_devolution_moy</b><br>
        <b>• Au-dessus du seuil</b> (alerte) pour <b>Dp_moy</b> et <b>Etat_devolution_moy</b>
    </div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ Information sur l'affichage des alertes", expanded=False):
    st.markdown("Dans nos visualisations, nous utilisons l'ensemble des historiques connus de nos séries, ainsi que nos prédictions pour les 4 semaines à venir, pour lesquelles nous ne disposons pas des valeurs réelles.")


# --- Chemins de données et modèles ---
path_data_brute, path_data_traite = get_data_paths()

# Localisation des modèles
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#Choix de l'horizon de prévision
horizon = st.slider(
    "Choisissez l'horizon de prévision (en semaines)",
    min_value=1, max_value=20, value=4, step=1
)
MODELS_DIR = os.path.join(project_root, "data", f"models_data_{horizon}")

# --- Récupération des zones/indicateurs dynamiquement ---
if not os.path.exists(MODELS_DIR):
    st.error("Le dossier des modèles n'est pas présent pour cet horizon de prédiction. Merci de lancer les fichiers de modélisation dans la page prédiction")
    st.stop()

model_folders = [f for f in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, f))]

indicateurs = set()
zones = set()
model_map = {}

for folder in model_folders:
    if folder.startswith("models_"):
        parts = folder.split("_")
        # Cas particulier pour Etat_devolution
        if parts[1] == "Etat" and parts[2] == "devolution":
            indicateur = parts[1] + "_" + parts[2] + "_" + parts[3]
            zone = "_".join(parts[4:])
        else:
            indicateur = parts[1] + "_" + parts[2]
            zone = "_".join(parts[3:])
        indicateurs.add(indicateur)
        zones.add(zone)
        model_map[(indicateur, zone)] = folder

indicateurs = sorted(list(indicateurs))
zones = sorted(list(zones))

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration du système d'alerte")

    # Sélection de la zone
    selected_zone = st.selectbox("Sélectionnez la zone", zones)

    # Seuils personnalisables pour chaque indicateur (champ numérique direct)
    st.subheader("Saisissez un seuil pour chaque indicateur")
    seuils = {}
    seuil_min, seuil_max = 0.0, 10000.0  
    seuils_defaut = {
        "Dp_moy": 4.0,
        "Etat_devolution_moy": 1000.0,
        "Nff_moyen":12.5,
        "Nfr_moyen": 5.5,
        "Pjfn_moyen": 10.50,
        "Pjft_moyen": 4.0
    }
    for indicateur in indicateurs:
        valeur_defaut = seuils_defaut.get(indicateur, 4.00)
        seuils[indicateur] = st.number_input(
            f"Saisissez le seuil d'alerte pour l'indicateur {indicateur}",
            min_value=float(seuil_min),
            max_value=float(seuil_max),
            value=valeur_defaut,
            step=0.1,
            format="%.2f",
            key=f"input_{indicateur}"
        )

NB_COLS = 2
cols = st.columns(NB_COLS)

for idx, indicateur in enumerate(indicateurs):
    folder = model_map.get((indicateur, selected_zone))
    col = cols[idx % NB_COLS]
    with col:
        st.markdown(f"### {indicateur}")

        if not folder:
            st.info("Pas de données pour cet indicateur/zone.")
            continue

        base_path = os.path.join(MODELS_DIR, folder)
        train_futur_path = os.path.join(base_path, "predictions_train_futur.csv")
        futur_path = os.path.join(base_path, "predictions_futur.csv")

        if not (os.path.exists(train_futur_path) and os.path.exists(futur_path)):
            st.warning("Fichiers 'predictions_train_futur.csv' ou 'predictions_futur.csv' non trouvés.")
            continue

        try:
            df_train = pd.read_csv(train_futur_path)
            df_futur = pd.read_csv(futur_path)
        except Exception as e:
            st.error(f"Erreur lecture fichiers: {e}")
            continue

        # Vérification des colonnes nécessaires
        for colname in ['ds', 'y']:
            if colname not in df_train.columns:
                st.warning(f"Colonne '{colname}' manquante dans predictions_train_futur.csv.")
                continue
        for colname in ['ds', 'y_pred']:
            if colname not in df_futur.columns:
                st.warning(f"Colonne '{colname}' manquante dans predictions_futur.csv.")
                continue

        # Convertir les dates
        df_train['ds'] = pd.to_datetime(df_train['ds'])
        df_futur['ds'] = pd.to_datetime(df_futur['ds'])

        seuil_val = seuils[indicateur]

        # Pour les deux indicateurs spécifiques l'alerte est AU-DESSUS DU SEUIL, sinon SOUS le seuil
        if indicateur in ["Dp_moy", "Etat_devolution_moy"]:
            df_train['Alerte'] = df_train['y'] > seuil_val
            df_futur['Alerte'] = df_futur['y_pred'] > seuil_val
            alert_text = "au-dessus"
        else:
            df_train['Alerte'] = df_train['y'] < seuil_val
            df_futur['Alerte'] = df_futur['y_pred'] < seuil_val
            alert_text = "sous"

        # --- NOUVELLE VISUALISATION PLOTLY ---
        fig, points_alerte_train, points_alerte_futur = plot_indicator_alert(df_train, df_futur, indicateur, seuil_val)
        
        # Mise en page du graphique
        fig.update_layout(
            title=f"Évolution et prévision pour la zone {selected_zone}",
            xaxis_title='Date',
            yaxis_title=indicateur,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=40, r=30, t=60, b=80),
            showlegend=True
        )

        # Affichage du graphique Plotly
        st.plotly_chart(fig, use_container_width=True)

        # Résumé alertes
        nb_alertes_train = points_alerte_train.shape[0]
        nb_alertes_futur = points_alerte_futur.shape[0]
        if nb_alertes_train + nb_alertes_futur > 0:
            st.error(f"⚠️ {nb_alertes_train} période(s) {alert_text} du seuil dans l'historique, {nb_alertes_futur} période(s) {alert_text} du seuil dans le futur")
            with st.expander("Voir périodes en alerte"):
                if nb_alertes_train > 0:
                    st.write("**Train/Historiques :**")
                    for _, row in points_alerte_train.iterrows():
                        st.warning(f"{row['ds'].strftime('%Y-%m-%d')} : {row['y']:.2f}")
                if nb_alertes_futur > 0:
                    st.write("**Futur :**")
                    for _, row in points_alerte_futur.iterrows():
                        st.warning(f"{row['ds'].strftime('%Y-%m-%d')} : {row['y_pred']:.2f}")
        else:
            st.success(f"Aucune période {alert_text} du seuil ({seuil_val:.2f}) dans l'historique ou le futur.")

        # Données tabulaires (optionnel)
        with st.expander("Afficher les données détaillées"):
            st.write("**Données historiques (train)**")
            st.dataframe(df_train[['ds', 'y']], use_container_width=True)
            st.write("**Données prévisions futures**")
            st.dataframe(df_futur[['ds', 'y_pred']], use_container_width=True)
            