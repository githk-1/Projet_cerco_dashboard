import os
import shutil
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("import_bases.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def get_paths(source_dir=None, project_root=None):
    """
    Retourne les chemins utilisés dans le projet (dossier source, data brute, data traitée).
    Si les paramètres ne sont pas fournis, utilise les chemins par défaut.

    Args:
        source_dir (str, optional): Chemin du dossier source.
        project_root (str, optional): Chemin racine du projet.

    Returns:
        tuple: (source_dir, data_brut_dir, data_traite_dir)
    """
    if source_dir is None:
        source_dir = r"C:\Users\hamici-k\Desktop\Résaux"
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    data_brut_dir = os.path.join(data_dir, "data_brute")
    data_traite_dir = os.path.join(data_dir, "data_traite")
    return source_dir, data_brut_dir, data_traite_dir
# # Chemin du dossier réseau (à adapter selon ton cas)
# source_dir = r"C:\Users\hamici-k\Desktop\Résaux"
# # Chemin du dossier de destination dans ton projet (racine)
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# data_dir = os.path.join(project_root, "data")
# data_brut_dir = os.path.join(data_dir, "data_brute")
# data_traite_dir = os.path.join(data_dir, "data_traite") 

def create_directories(data_brut_dir, data_traite_dir):
    """
    Crée les dossiers de données s'ils n'existent pas.

    Args:
        data_brut_dir (str): Chemin vers le dossier data_brute.
        data_traite_dir (str): Chemin vers le dossier data_traite.
    """
    os.makedirs(data_brut_dir, exist_ok=True)
    logging.info(f"Dossier data brute créé à l'emplacement : {data_brut_dir}")
    os.makedirs(data_traite_dir, exist_ok=True)
    logging.info(f"Dossier data traité créé à l'emplacement : {data_traite_dir}")

# Création des dossiers si besoin
# os.makedirs(data_brut_dir, exist_ok=True)
# logging.info(f"Dossier data brute créé à l'emplacement : {data_brut_dir}")

# os.makedirs(data_traite_dir, exist_ok=True)
# logging.info(f"Dossier data traité créé à l'emplacement : {data_traite_dir}")


def copy_files(source_dir, data_brut_dir):
    """
    Copie les fichiers du dossier source vers le dossier data_brute (écrase si déjà présents).

    Args:
        source_dir (str): Chemin du dossier source.
        data_brut_dir (str): Chemin du dossier data_brute.
    """
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(data_brut_dir, filename)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, dest_file)
            logging.info(f"Copié/Écrasé : {filename} dans {data_brut_dir}")

#  Copie des fichiers (écrase si déjà présent)
# for filename in os.listdir(source_dir):
#     source_file = os.path.join(source_dir, filename)
#     dest_file = os.path.join(data_brut_dir, filename)
#     if os.path.isfile(source_file):
#         shutil.copy2(source_file, dest_file)
#         logging.info(
#             f"Copié/Écrasé : {filename} dans {data_brut_dir}"
#         )

def import_bases(source_dir=None, project_root=None):
    """
    Fonction principale pour orchestrer l'import :
    - Récupère les chemins
    - Crée les dossiers nécessaires
    - Copie les fichiers depuis le dossier source

    Args:
        source_dir (str, optional): Chemin du dossier source.
        project_root (str, optional): Chemin racine du projet.
    """
    source_dir, data_brut_dir, data_traite_dir = get_paths(source_dir, project_root)
    create_directories(data_brut_dir, data_traite_dir)
    copy_files(source_dir, data_brut_dir)
    logging.info("Mise à jour terminée !")

import_bases()
