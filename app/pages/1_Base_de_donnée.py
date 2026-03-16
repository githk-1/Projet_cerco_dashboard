import sys
import os
# Ajoute le dossier parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import streamlit as st
from src.preparation_base import *
from src.plot_fonction import *
from src.analyse_fonction import *
import warnings
warnings.filterwarnings("ignore")
import subprocess

st.set_page_config(layout="wide")


path_data_brute, path_data_traite = get_data_paths()
    
st.markdown("""
    <h1 style='text-align: center;'>Partie base de données </h1>
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
        Cette section présente, pour chaque base, la version brute et la version traitée, détaille l’ensemble des prétraitements réalisés, et offre la possibilité de télécharger les bases traitées au format CSV.
    </div>
""", unsafe_allow_html=True)

bdd_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../orchestration/BDD.py'))

with st.expander("🔄 Actualiser les bases de données & relancer le prétraitement complet", expanded=False):
    st.write("""**Ce bouton permet de :**
- Actualiser toutes les bases de données (importation ou mise à jour des données brutes)
- Relancer l'intégralité du prétraitement (nettoyage, transformation, enrichissement et sauvegarde des bases traitées)
    """)
    if st.button("Lancer l'actualisation des bases de données"):
        try:
            result = subprocess.run([sys.executable, bdd_script_path])#, capture_output=True, text=True)
            st.success("L'actualisation des bases de données a été exécutée avec succès !")
            #st.text(result.stdout) 
            if result.stderr:
                st.error(result.stderr)
        except Exception as e:
            st.error(f"Erreur lors de l'exécution : {e}")



# Création d'onglets
tab1, tab2, tab3, tab4 = st.tabs(["Cercos", "Météo", "Intrant", "Finale"])

with tab1:
    st.subheader("Base brute Cercos")


    df_cercos = load_data(path=path_data_brute + "/base_cerco.xlsx", database_name="cercos")

    # Afficher les données
    st.dataframe(df_cercos, use_container_width=True)
    
    
    # Informations sur les données
    st.info(f"La base Cercos contient {df_cercos.shape[0]} lignes et {df_cercos.shape[1]} colonnes")
    
    
    

    st.markdown("---")
    st.subheader("Base Cercos après traitement")
    df_cercos_traitee = load_data(path=path_data_traite + "/df_cercos_traite_mod_zone.parquet", database_name="cercos")
    st.dataframe(df_cercos_traitee, use_container_width=True)
    st.info(f"Après traitement : {df_cercos_traitee.shape[0]} lignes et {df_cercos_traitee.shape[1]} colonnes")

    with st.expander("Détail des traitements appliqués", expanded=False):
        st.markdown("""
        - Analyse et ajout des semaines manquantes pour chaque poste d'observation
        - Suppression des zones de traitement relatives à Niéky, Badema et Béoumi
        - Ciblage et traitement des valeurs aberrantes
        - Suppression des doublons 
        - Imputation des valeurs manquantes originales et de celles ajoutées (selon un algorithme de similitude)
        - Agrégation de la base au niveau d'observation : Zone de traitement. Nous avons pris la moyenne par semaine de tout les postes afin d'obtenir un niveau d'observation par zone de traitement.
        """)

    csv = df_cercos_traitee.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger les données en CSV",
        data=csv,
        file_name="cercos_data.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("Base brute Météo")
    
    # Utiliser votre fonction load_data pour charger la base Météo
    df_meteo = load_data(path= path_data_brute+"/df_meteoQlick_act.xlsx", database_name= "qlik")
    
    # Afficher les données
    st.dataframe(df_meteo, use_container_width=True)
    
    # Informations sur les données
    st.info(f"La base Météo contient {df_meteo.shape[0]} lignes et {df_meteo.shape[1]} colonnes")

    st.markdown("---")
    st.subheader("Base Météo après traitement")
    df_meteo_traitee = load_data(path=path_data_traite + "/df_meteoQlick_traite_mod_zone.parquet", database_name="qlik")
    st.dataframe(df_meteo_traitee, use_container_width=True)
    st.info(f"Après traitement : {df_meteo_traitee.shape[0]} lignes et {df_meteo_traitee.shape[1]} colonnes")

    with st.expander("Détail des traitements appliqués", expanded=False):
        st.markdown("""
        - Analyse de la qualité des données entre les stations automatiques et manuelles : Station automatique conservées.
        - Suppression des stations météo manuelles ainsi que celles relatives à Niéky, Badema et Vantage Confluence.
        - Analyse et ajout des jours manquants pour chaque station météo.
        - Ciblage et traitement des valeurs aberrantes.
        - Agrégation de la base au format hebdomadaire: transformation de nos indicateur en prenant des moyennes, des sommes et des décile (représentés par un d) pour chaque station météo.
        - Passage du niveau station météo à la maille zone de traitement.
        - Suppression des doublons.
        - Imputation des valeurs manquantes originales et de celles ajoutées (selon un algorithme de similitude).
        - Création de la variable Pluviométrie 4 semaines qui nous informe sur la quantité de pluie sur les 4 dernières semaines.
        - Création de la variable intensité de pluie qui nous donne un état d'alerte selon la valeur de pluviométrie 4 semaines: 
                    

            - Pluviométrie < 50mm -> Calme
            - Pluviométrie > 50mm -> Faible
            - Pluviométrie > 100mm -> Forte
            - Pluviométrie > 150mm -> Très forte
            - Pluviométrie >= 200mm -> Alerte

        """)
    csv = df_meteo_traitee.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger les données en CSV",
        data=csv,
        file_name="meteo_data.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("Base brute Intrant")
    
    # Utiliser votre fonction load_data pour charger la base Intrant
    df_intrant = load_data(path= path_data_brute+"/Intrant.xlsx", database_name= "intrant")
    
    # Afficher les données
    st.dataframe(df_intrant, use_container_width=True)
    
    # Informations sur les données
    st.info(f"La base Intrant contient {df_intrant.shape[0]} lignes et {df_intrant.shape[1]} colonnes")

    st.markdown("---")
    st.subheader("Base Intrant après traitement")
    df_intrant_traitee = load_data(path_data_traite+"/df_intrant_complet_mod_zone.parquet", database_name="intrant_comp")
    st.dataframe(df_intrant_traitee, use_container_width=True)
    st.info(f"Après traitement : {df_intrant_traitee.shape[0]} lignes et {df_intrant_traitee.shape[1]} colonnes")

    with st.expander("Détail des traitements appliqués", expanded=False):
        st.markdown("""
        - Suppression des zones de traitement relatives à Niéky, Badema, Béoumi.
        - Catégorisation des types d'intrant en grandes familles.
        - Ciblage et traitement des valeurs aberrantes.
        - Création de la variable nombre de jours entre 2 mêmes traitements pour prendre en compte les résistances.
        - Agrégation de la base au format hebdomadaire : transformation de nos indicateurs en prenant les sommes des quantités d'intrant et d'huile pour chaque zone de traitement.
        - Création de la variable catégorie qui fait le lien entre les grandes familles d'intrant et la quantité d'huile utilisée sur chaque traitement :
        
            - catégorie 1 : quantité d'huile entre 0 et 2 litre/hectare  
            - catégorie 2 : quantité d'huile entre 2 et 6 litres/hectare  
            - catégorie 3 : quantité d'huile supérieure ou égale à 6 litres/hectare
        
        - Ciblage et suppression des doublons.
        - Mise à niveau de la base pour la jointure : ajout d'une ligne par chaque semaine de la période pour chaque poste.

            - L'objectif est de préparer la jointure de nos bases en ajoutant des lignes qui correspondent aux semaines sans traitement, elles sont remplies avec des 0 au niveau des quantités d'intrant et d'huile utilisées.
        """)

    csv = df_intrant_traitee.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger les données en CSV",
        data=csv, 
        file_name="intrant_data.csv",
        mime="text/csv",
    )

with tab4:
    st.subheader("Base Finale")
    st.markdown("Cette base de données constitue la fusion de l’ensemble des bases traitées, intégrant toutes les informations nécessaires pour l’analyse et la modélisation finale.")

    # Utiliser votre fonction load_data pour charger la base Intrant
    df_merge = load_data(path_data_traite+ "/merge_final_mod_zone.parquet", database_name="merge1")
    
    # Afficher les données
    st.dataframe(df_merge, use_container_width=True)
    
    # Informations sur les données
    st.info(f"La base Intrant contient {df_merge.shape[0]} lignes et {df_merge.shape[1]} colonnes")

    with st.expander("Détail des traitements appliqués", expanded=False):
        st.markdown("""
        - Jointure de l'ensemble de nos base de données pour notre modélisation.
        """)

    csv = df_merge.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger les données en CSV",
        data=csv,
        file_name="final_data.csv",
        mime="text/csv",
    )

