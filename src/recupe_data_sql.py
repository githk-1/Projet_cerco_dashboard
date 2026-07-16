import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
import warnings
warnings.filterwarnings("ignore")

def connexion_serveur():
    """
    Cette fonction établit une connexion à la base de données SQL Server en utilisant pyodbc.

    Paramètres :
    Aucun

    Renvoie :
    conn (pyodbc.Connection) : L'objet de connexion à la base de données SQL Server.

    Soulève :
    pyodbc.Error : S'il y a une erreur de connexion à la base de données.

    Remarque :
    Les détails de la connexion (serveur, nom d'utilisateur, base de données, mot de passe) sont codés en dur dans la fonction.
    """


    # Connection details
    Serveur='FR1-DVS-DBDEV1.groupecf.net'
    # FR1-DVS-DBDEV1\SIPADEV
    username='sql-sipa-report'
    Database='db_sipa_isoprod_mars'
    password='s1p4r3p0rt'

    # Connection string
    connectionString = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={Serveur};DATABASE={Database};UID={username};PWD={password}'

    # Establish connection
    conn = pyodbc.connect(connectionString)

    return conn
# fonction fermeture de la connexion au serveur
def fermeture_connexion_serveur(conn):
    """
    Ferme la connexion à la base de données SQL Server.

    Paramètres :
    conn (pyodbc.Connection) : L'objet de connexion à la base de données SQL Server.

    Renvoie :
    bool : True si la connexion a été fermée avec succès, False sinon.

    Gestion des exceptions :
    AttributeError : Si l'objet de connexion n'a pas de méthode close(), un message d'erreur est imprimé et la fonction renvoie False.
    Exception : Pour toute autre erreur lors de la fermeture de la connexion, un message d'erreur est imprimé avec les détails de l'exception et la fonction renvoie False.
    """
    try:
        conn.close()
        return True
    except AttributeError:
        # conn n'a pas de méthode close()
        print("L'objet de connexion n'a pas de méthode close.")
        return False
    except Exception as e:
        # Gestion générale des exceptions pour tout autre problème
        print(f"Une erreur s'est produite lors de la fermeture de la connexion : {e}")
        return False


# fonction execution de la requete
# def execute_requete(requete):
#     conn=connexion_serveur()
#     base=pd.read_sql(requete, conn)
#     fermeture_connexion_serveur(conn=conn)
#     return base

def execute_requete(requete):
    """
    Exécute une requête SQL sur la base de données, puis ferme la connexion.

    Paramètres :
    requete (str) : La requête SQL à exécuter.

    Renvoie :
    pandas.DataFrame : Le résultat de la requête sous forme de DataFrame pandas.

    Gestion des exceptions :
    pyodbc.Error : Si une erreur se produit lors de l'exécution de la requête ou de la fermeture de la connexion, une exception de type pyodbc.Error est levée.

    Remarque :
    La fonction `connexion_serveur()` est supposée exister et retourner un objet de connexion à la base de données.
    La fonction `fermeture_connexion_serveur(conn)` est supposée fermer la connexion à la base de données.
    """
    try:
        # Établissement de la connexion au serveur
        conn = connexion_serveur()

        # Exécution de la requête et récupération des résultats dans un DataFrame pandas
        base = pd.read_sql(requete, conn)

        # Fermeture de la connexion
        fermeture_connexion_serveur(conn)

        return base
    except pyodbc.Error as e:
        # Gestion des erreurs de connexion ou d'exécution de requête
        print(f"An error occurred: {e}")
        raise
req_info_indic="""

SELECT
       z.str_libelle_court as Structure,z.dom_libelle as Domaine, 
	   z.sect_libelle as Secteur,z.par_libelle as Parcelle,i.Jour as Date_comptage, i.Annee as Annee_comptage, 
	   i.NumSemaine as Semaine_comptage,i.par_id,i.par_uid,
	   i.Valeur AS Fleurs
FROM vw_report_indic_parcelle i
INNER JOIN vw_report_zone z ON  z.par_id=i.par_id
WHERE i.Annee>2019 AND i.defi_code='sv_comptage_fleur' AND z.str_libelle_court in ('PHP','SCB','GOL','GEL')

"""

requete_meteo = """    
SELECT
m.stamet_id,s.sect_id, z.dom_id, sm.stamet_libelle as Station_meteo,
m.Jour as Date ,z.dom_libelle as Domaine, s.sect_libelle as Secteur, s.sect_surface,
m.defi_code, m.Valeur
FROM vw_report_indic_station_meteo m
INNER JOIN vw_report_secteur s on m.stamet_id=s.stamet_id
INNER JOIN vw_report_zone z on z.sect_id=s.sect_id
INNER JOIN vw_report_station_meteo sm on m.stamet_id= sm.stamet_id
WHERE defi_code in ('sv_pluviom','sv_relative_humidity_min','sv_relative_humidity_max',
'sv_tmoy', 'sv_tmin','sv_tmax','sv_ensol','sv_evapora', 'sv_piche_abri','sv_piche_amps', 'sv_tsol', 'sv_insolation') 
AND Jour >= '2022-01-01' AND m.str_id=11"""

path=r"C:\Users\hamici-k\Desktop\projet_cerco\data\data_brute\base_meteo.parquet"
print(requete_meteo)
base_meteo = execute_requete(requete_meteo)
print(base_meteo.head())                                
base_meteo.to_parquet(path)