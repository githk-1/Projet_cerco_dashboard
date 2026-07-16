import sys
import os
# Ajoute le dossier parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import streamlit as st
from src.preparation_base import *
from src.plot_fonction import *
from src.analyse_fonction import *
from src.model_fonction import *
from src.streamlit_func import *
import warnings
warnings.filterwarnings("ignore")
import subprocess
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide")
  

path_data_brute, path_data_traite = get_data_paths()

st.markdown("""
    <h1 style='text-align: center;'>Partie prédictions</h1>
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
        Cette section comporte la visualisation dynamique des résultats pour la zone sélectionnée ainsi que le lancement des modèles de prévision.
    </div>
""", unsafe_allow_html=True)




project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR_PY = os.path.join(project_root, "fichier_models")
horizon = st.slider(
    "Choisis l'horizon de prévision (en semaines)",
    min_value=1, max_value=20, value=4, step=1
)

MODELS_DIR = os.path.join(project_root, "data", f"models_data_{horizon}")


if not os.path.exists(MODELS_DIR):
    st.error("Le dossier des modèles n'est pas présent pour cet horizon. Merci de cliquer sur le bouton 'Lancer le test du modèle (test.py)' dans la partie exécution des modélisations en bas de la page.")
else:
    # 1. Récupérer tous les dossiers de modèles
    model_folders = [f for f in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, f))]

    # 2. Extraire les indicateurs et zones disponibles
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

    # 3. Sélection dans Streamlit (seulement la zone)
    selected_zone = st.selectbox("Sélectionnez la zone", zones)

    # 4. Onglets pour à date et futur
    tab1, tab2 = st.tabs(["Prédiction sur les données à date", "Prédiction pour les 4 prochaines semaines"])

    NB_COLS = 2

    def show_panel(indicateur, folder, is_futur=False):
        # Chemins des fichiers
        base_path = os.path.join(MODELS_DIR, folder)
        if is_futur:
            train_path = os.path.join(base_path, "train_futur.csv")
            pred_path = os.path.join(base_path, "predictions_futur.csv")
            train_msg = f"Le fichier '{os.path.basename(train_path)}' n'est pas présent. Merci de cliquer sur le bouton  'Lancer la prédiction future (model_futur.py)' en bas de la page."
            pred_msg = f"Le fichier '{os.path.basename(pred_path)}' n'est pas présent. Merci de cliquer sur le bouton  'Lancer la prédiction future (model_futur.py)' en bas de la page."

        else:
            train_path = os.path.join(base_path, "train_data.csv")
            pred_path = os.path.join(base_path, "predictions.csv")
            train_msg = f"Le fichier '{os.path.basename(train_path)}' n'est pas présent. Merci de cliquer sur le bouton 'Lancer le test du modèle (test.py)' en bas de la page."
            pred_msg = f"Le fichier '{os.path.basename(pred_path)}' n'est pas présent. Merci de cliquer sur le bouton 'Lancer le test du modèle (test.py)' en bas de la page."


        try:
            train_df = pd.read_csv(train_path)
        except FileNotFoundError:
            st.warning(train_msg)
            return
        except Exception as e:
            st.error(f"Erreur lors de la lecture de '{os.path.basename(train_path)}' : {e}")
            return

        try:
            pred_df = pd.read_csv(pred_path)
        except FileNotFoundError:
            st.warning(pred_msg)
            return
        except Exception as e:
            st.error(f"Erreur lors de la lecture de '{os.path.basename(pred_path)}' : {e}")
            return
        
        # Affichage du nom de l'indicateur
        st.markdown(f"<div style='text-align:center; font-weight:bold; font-size:20px;'>Indicateur : {indicateur}</div>",
        unsafe_allow_html=True)
        # Affichage du graphique, SANS sous-titre "visualisation graphique"
        if is_futur:
            visualiser_futur_zone_streamlit_plotly(
                base_path=MODELS_DIR,
                target_col_name=indicateur,
                zone=selected_zone,
                pred_col="y_pred",
                horizon=horizon
                
            )
            
            
        else:
            visualiser_zone_streamlit_plotly(
                base_path=MODELS_DIR,
                target_col_name=indicateur,
                zone=selected_zone,
                pred_col="y_pred",
                horizon=horizon
            )

        # Tables dans un expander
        with st.expander("Voir les données d'entraînement & prédiction"):
            if is_futur:
                st.subheader("Prédictions futures")
                st.dataframe(pred_df)

                st.subheader("Données d'entraînement futur")
                st.dataframe(train_df)
                
            else:
                st.subheader("Prédictions")
                st.dataframe(pred_df) 

                st.subheader("Données d'entraînement")
                st.dataframe(train_df)
        
        with st.expander("Voir l'importance des variables"):
            if is_futur : 
                visualiser_importance_variables_streamlit(
                base_path=base_path,
                target_col_name=indicateur,
                zone=selected_zone,
                is_future=True, 
                n_top=25,
                key=f"importance_plot_{selected_zone}_{indicateur}_{'futur' if is_futur else 'test'}_{idx}",
                horizon=horizon 
            )
            
            else:
                visualiser_importance_variables_streamlit(
                base_path=base_path,
                target_col_name=indicateur,
                zone=selected_zone,
                n_top=25,
                key=f"importance_plot_{selected_zone}_{indicateur}_{'futur' if is_futur else 'test'}_{idx}",
                horizon=horizon
            )
                    

    # Onglet 1 : à date 
    with tab1:
        st.markdown("## Prédictions sur les données à date")
        cols = st.columns(NB_COLS)
        for idx, indicateur in enumerate(indicateurs):
            folder = model_map.get((indicateur, selected_zone))
            col = cols[idx % NB_COLS]
            with col:
                if folder:
                    show_panel(indicateur, folder, is_futur=False)
                else:
                    st.warning(f"Aucun modèle trouvé pour l'indicateur {indicateur} et la zone {selected_zone}")


    # Onglet 2 : futur
    with tab2:
        st.markdown("## Prédictions pour les 4 prochaines semaines")
        cols = st.columns(NB_COLS)
        for idx, indicateur in enumerate(indicateurs):
            folder = model_map.get((indicateur, selected_zone))
            col = cols[idx % NB_COLS]
            with col:
                if folder:
                    show_panel(indicateur, folder, is_futur=True)
                else:
                    st.warning(f"Aucun modèle trouvé pour l'indicateur {indicateur} et la zone {selected_zone}")


st.markdown("### Exécution des modélisations")

if not os.path.exists(MODELS_DIR):
    st.info("Vous pouvez créer le dossier et vos modèles en lançant l'optimisation avec le bouton ci-dessous.")

with st.expander("ℹ️ Ordre d'exécution des scripts (cliquez pour voir)"):
    st.markdown("""
    **À savoir**

    - Si un fichier de prédiction ou d'entraînement est manquant, il faut d'abord lancer le fichier **test** (`test.py`).
    - ⚠️ **Attention**: le fichier `test.py` dépend du résultat de l'optimisation.
      Si le test ne fonctionne pas (ou les fichiers sont encore absents), cela signifie probablement que l'optimisation (`train.py`) n'a pas encore été réalisée.
      Dans ce cas, commencez par **lancer l'optimisation** avant de relancer le test.
    - De même, la prédiction future (`model_futur.py`) dépend de l'optimisation.

    **Résumé: Chaque étape dépend de la précédente. Veillez à respecter l'ordre pour que tous les fichiers nécessaires soient bien générés.**
    
    ---
                
    **Ordre d'exécution recommandé pour la modélisation:**

    1️⃣ **Lancer l'optimisation** (`train.py`)  (prend ~2H10)  
    2️⃣ **Lancer le test du modèle** (`test.py`)  (prend ~15min)  
    3️⃣ **Lancer la prédiction future** (`model_futur.py`)  (prend ~15min)
    """)

cols = st.columns(3)

# Bouton pour lancer train.py
with cols[0]:
    if st.button("1. Lancer l'optimisation (train.py)"):
        result = subprocess.run(["python", os.path.join(MODEL_DIR_PY, "train.py"), "--base_path", MODELS_DIR, "--horizon", str(horizon)])
        st.success("L'optimisation a été exécutée avec succès ! Vous pouvez maintenant actualiser la page pour voir les résultats.")
        #st.text(result.stdout)
        if result.stderr:
            st.error(result.stderr)

# Bouton pour lancer test.py
with cols[1]:
    if st.button("2. Lancer le test du modèle (test.py)"):
        result = subprocess.run(["python", os.path.join(MODEL_DIR_PY   , "test.py"), "--base_path", MODELS_DIR, "--horizon", str(horizon)])
        st.success("Le test a été exécutée avec succès ! Vous pouvez maintenant actualiser la page pour voir les résultats.")
        #st.text(result.stdout)
        if result.stderr:
            st.error(result.stderr)

# Bouton pour lancer model_futur.py
with cols[2]:
    if st.button("3. Lancer la prédiction future (model_future.py)"):
        result = subprocess.run(["python", os.path.join(MODEL_DIR_PY, "model_futur.py"), "--base_path", MODELS_DIR, "--horizon", str(horizon)])
        st.success("La prédiction future a été exécutée avec succès ! Vous pouvez maintenant actualiser la page pour voir les résultats.")
        #st.text(result.stdout)
        if result.stderr:
            st.error(result.stderr)
