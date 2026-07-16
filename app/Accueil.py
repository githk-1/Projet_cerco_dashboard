import sys
import os
# Ajoute le dossier parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(
    page_title="Prévision Cercos", 
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <h2 style='text-align: center;'>Bienvenue dans l'application de prévision des tendances de la maladie de la cercosporiose</h2>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.fonctionnalites-section {
    max-width: 700px;
    margin: 0 auto 18px auto;
}
.fonctionnalites-section h3 {
    margin-bottom: 24px;
    text-align: center;
    font-size: 1.8rem;
}
.fonctionnalites-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px 22px;
    justify-items: center;
}
.fonctionnalite-card {
    background-color: #f4f6fb;
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(44,62,80,0.06);
    display: flex;
    align-items: center;
    font-size: 17px;
    width: 100%;
    max-width: 320px;
}
.fonctionnalite-icon {
    font-size: 26px;
    margin-right: 15px;
}
@media (max-width: 700px) {
    .fonctionnalites-section {
        max-width: 99vw;
    }
    .fonctionnalites-grid {
        grid-template-columns: 1fr;
    }
}
</style>
<div class="fonctionnalites-section">
    <h3>Fonctionnalités principales</h3>
    <div class="fonctionnalites-grid">
        <div class="fonctionnalite-card"><span class="fonctionnalite-icon">🗂️</span> Consultation des bases de données (Cercos, Météo, Intrant)</div>
        <div class="fonctionnalite-card"><span class="fonctionnalite-icon">📊</span> Analyse détaillée des bases de données</div>
        <div class="fonctionnalite-card"><span class="fonctionnalite-icon">🤖</span> Lancement des modèles et visualisation des résultats</div>
        <div class="fonctionnalite-card"><span class="fonctionnalite-icon">⏰</span> Alertes personnalisées par zone de traitement</div>
    </div>
</div>
<p style='text-align:center;margin-top:30px;font-size:16px;'>Utilisez le menu à gauche pour naviguer dans l'application.</p>
""", unsafe_allow_html=True)

# Image
# st.image("path_to_image.jpg", caption="Analyse Cercos") 