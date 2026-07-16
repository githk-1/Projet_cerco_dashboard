# import os
# import pandas as pd
# from openpyxl import Workbook
# from openpyxl.utils.dataframe import dataframe_to_rows
# from openpyxl.styles import Font, Border, Side
# from openpyxl.utils import get_column_letter
# from openpyxl.chart import LineChart, BarChart, Reference, Series
# from openpyxl.chart.legend import Legend
# from openpyxl.chart.axis import ChartLines
# from openpyxl.chart.label import DataLabelList

# def autofit_columns(ws):
#     for column_cells in ws.columns:
#         length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
#         col_letter = get_column_letter(column_cells[0].column)
#         ws.column_dimensions[col_letter].width = length + 2  # +2 pour un peu d'espace

# def add_border_to_table(ws, start_row, start_col, nrows, ncols):
#     border = Border(
#         left=Side(style='thin'),
#         right=Side(style='thin'),
#         top=Side(style='thin'),
#         bottom=Side(style='thin')
#     )
#     for r in range(start_row, start_row + nrows):
#         for c in range(start_col, start_col + ncols):
#             ws.cell(row=r, column=c).border = border

# def add_analysis_chart(ws, df, start_row, start_col):
#     ncols = df.shape[1]
#     nrows = df.shape[0] + 1
#     graph_col = start_col + ncols + 2

#     # Année-Semaine pour l'axe X
#     cat_labels = [f"{int(a)}-{int(s):02d}" for a, s in zip(df["Annee"], df["Semaine"])]
#     for i, val in enumerate(cat_labels):
#         ws.cell(row=start_row + 2 + i, column=graph_col, value=val)
#     cat_col = graph_col

#     col_y_pred_futur = start_col + df.columns.get_loc("y_pred_futur")
#     col_y_pred_test = start_col + df.columns.get_loc("y_pred_test")
#     col_Y = start_col + df.columns.get_loc("Y")
#     col_delta_futur = start_col + df.columns.get_loc("Delta Y future")
#     col_delta_test = start_col + df.columns.get_loc("Delta Y test")

#     data_start = start_row + 2
#     data_end = start_row + 1 + df.shape[0]
#     cats = Reference(ws, min_col=cat_col, min_row=data_start, max_row=data_end)

#     # 1. Chart principal (LineChart)
#     chart = LineChart()
#     chart.title = "Comparaison Y / Prédictions"
#     chart.y_axis.title = "Valeur"
#     chart.x_axis.title = "Année-Semaine"
#     chart.x_axis.textRotation = 45
#     chart.height = 12
#     chart.width = 32

#     # FORCER l'affichage des valeurs sur l'axe Y principal
#     chart.y_axis.delete = False  # Ne pas supprimer l'axe
#     chart.y_axis.visible = True  # Forcer la visibilité
#     chart.y_axis.majorTickMark = "out"  # Graduations principales vers l'extérieur
#     chart.y_axis.minorTickMark = "none"  # Pas de graduations secondaires
#     chart.y_axis.tickLblPos = "nextTo"  # Labels à côté des graduations
#     chart.y_axis.number_format = "0.00"  # Format numérique avec 2 décimales

#     # FORCER l'affichage des valeurs sur l'axe X
#     chart.x_axis.delete = False
#     chart.x_axis.visible = True
#     chart.x_axis.majorTickMark = "out"
#     chart.x_axis.minorTickMark = "none"
#     chart.x_axis.tickLblPos = "nextTo"

#     # Courbes
#     line_colors = {
#         "Y réel": "3050F8",          # bleu
#         "y_pred_test": "FF0000",     # rouge 
#         "y_pred_futur": "00B050"       # vert
#     }
#     courbes_info = [
#         (col_Y, "Y réel", line_colors["Y réel"]),
#         (col_y_pred_test, "y_pred_test", line_colors["y_pred_test"]),
#         (col_y_pred_futur, "y_pred_futur", line_colors["y_pred_futur"])
#     ]
#     for col, label, color in courbes_info:
#         values = Reference(ws, min_col=col, min_row=data_start, max_row=data_end)
#         series = Series(values, title=label)
#         series.graphicalProperties.line.solidFill = color
#         chart.series.append(series)
#     chart.set_categories(cats)

#     # 2. BarChart pour les deltas (axe secondaire)
#     bar_chart = BarChart()
#     bar_chart.type = "col"
#     bar_chart.grouping = "clustered"
#     bar_chart.overlap = 0
#     bar_chart.y_axis.title = "Delta (%)"
#     bar_chart.y_axis.axId = 200
#     bar_chart.y_axis.crosses = "max"

#     # FORCER l'affichage des valeurs sur l'axe Y secondaire (barres)
#     bar_chart.y_axis.delete = False  # Ne pas supprimer l'axe
#     bar_chart.y_axis.visible = True  # Forcer la visibilité
#     bar_chart.y_axis.majorTickMark = "out"  # Graduations principales vers l'extérieur
#     bar_chart.y_axis.minorTickMark = "none"  # Pas de graduations secondaires
#     bar_chart.y_axis.tickLblPos = "nextTo"  # Labels à côté des graduations
#     bar_chart.y_axis.number_format = "0.0"  # Format numérique avec 1 décimale

#     # Barres delta futur : vert (comme la courbe y_pred_futur)
#     values = Reference(ws, min_col=col_delta_futur, min_row=data_start, max_row=data_end)
#     series = Series(values, title="delta_y_futur")
#     series.graphicalProperties.solidFill = line_colors["y_pred_futur"]
#     bar_chart.series.append(series)
#     # Barres delta test : rouge (comme la courbe y_pred_test)
#     values = Reference(ws, min_col=col_delta_test, min_row=data_start, max_row=data_end)
#     series = Series(values, title="delta_y_test")
#     series.graphicalProperties.solidFill = line_colors["y_pred_test"]
#     bar_chart.series.append(series)
#     bar_chart.set_categories(cats)
#     bar_chart.x_axis = chart.x_axis
#     bar_chart.y_axis.majorGridlines = None

#     # Ajouter bar_chart sur axe secondaire
#     chart += bar_chart
#     bar_chart.y_axis.crosses = "max"
#     bar_chart.y_axis.axId = 200
#     bar_chart.y_axis.title = "Delta (%)"
#     chart.y_axis.title = "Valeur"

#     # Légende à l'extérieur du plot
#     chart.legend.position = "r"
#     chart.legend.overlay = False

#     # Place le graphique
#     graph_cell = ws.cell(row=start_row + 2, column=graph_col + 2)
#     ws.add_chart(chart, graph_cell.coordinate)



# # ===========================
# # PARAMÈTRES À MODIFIER :
# # ===========================
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
# root_dir = os.path.join(project_root, "data", f"models_data_{15}")
# zones = [
#     "Bananord", "Banasud", "Brimbo Zone 1", "Brimbo Zone 2", "Broukro",
#     "Fleuve Zone 1", "Fleuve Zone 2", "Sindressou", "Spadi Extension",
#     "Spadi Zone 1", "Spadi Zone 2", "Tiassalé Zone 1", "Tiassalé Zone 2"
# ]
# indicateurs = [
#     "Dp_moy", "Etat_devolution_moy", "Pjft_moyen",
#     "Pjfn_moyen", "Nff_moyen", "Nfr_moyen"
# ]
# output_path = r"C:\Users\hamici-k\Desktop\analyse_previsions_zones.xlsx"
# # ===========================
# # FIN DES PARAMÈTRES
# # ===========================

# def get_model_folder_name(indic, zone):
#     return f"models_{indic}_{zone}"

# def try_read_csv(path):
#     try:
#         return pd.read_csv(path)
#     except Exception:
#         return None

# def write_table_with_title(ws, df, title, start_row, start_col):
#     title_cell = ws.cell(row=start_row, column=start_col, value=title)
#     title_cell.font = Font(bold=True)
#     nrows, ncols = 0, 0
#     for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), start=start_row + 1):
#         is_header = (r_idx == start_row + 1)
#         for c_idx, value in enumerate(row, start=start_col):
#             cell_value = value
#             if not is_header:
#                 # Si ce n'est pas le header, on tente de convertir en numérique
#                 if isinstance(value, str):
#                     try:
#                         if '.' in value:
#                             cell_value = float(value)
#                         else:
#                             cell_value = int(value)
#                     except Exception:
#                         cell_value = value
#             cell = ws.cell(row=r_idx, column=c_idx, value=cell_value)
            
#             # Formater en pourcentage les colonnes Delta (sauf pour Dp_moy)
#             if not is_header and "Delta" in str(df.columns[c_idx - start_col]) and "Dp_moy" not in title:
#                 cell.number_format = "0.00%"
            
#             ncols = max(ncols, c_idx - start_col + 1)
#         nrows += 1
#     add_border_to_table(ws, start_row + 1, start_col, nrows, ncols)
#     return start_row + nrows

# def write_error(ws, message, start_row, start_col):
#     cell = ws.cell(row=start_row, column=start_col, value=message)
#     cell.font = Font(bold=True, color="FF0000")
#     return start_row

# def compute_analysis_table(df_fut, df_pred, indicateur):
#     # Y réel ("y" ou "Y")
#     y_col = "y" if "y" in df_pred.columns else "Y" if "Y" in df_pred.columns else None
#     if y_col is None:
#         raise ValueError("Impossible de trouver la colonne 'y' ou 'Y' dans predictions.csv")
#     merge_cols = ["Annee", "Semaine"]
#     merged = pd.merge(
#         df_fut[merge_cols + ["y_pred"]].rename(columns={"y_pred": "y_pred_futur"}),
#         df_pred[merge_cols + ["y_pred", y_col]].rename(columns={"y_pred": "y_pred_test", y_col: "Y"}),
#         on=merge_cols, how="inner"
#     )
    
#     # Calcul des deltas selon l'indicateur
#     if indicateur == "Dp_moy":
#         # Pour Dp_moy : différence simple (numérique)
#         merged["Delta Y future"] = merged["Y"] - merged["y_pred_futur"]
#         merged["Delta Y test"] = merged["Y"] - merged["y_pred_test"]
#     else:
#         # Pour les autres indicateurs : ratio décimal (sera formaté en % par Excel)
#         merged["Delta Y future"] = merged.apply(
#             lambda row: (row['y_pred_futur']/row['Y']-1) if row['Y'] != 0 else None, axis=1
#         )
#         merged["Delta Y test"] = merged.apply(
#             lambda row: (row['y_pred_test']/row['Y']-1) if row['Y'] != 0 else None, axis=1
#         )
    
#     result = merged[["Annee", "Semaine", "y_pred_futur", "y_pred_test", "Y", "Delta Y future", "Delta Y test"]].copy()
#     return result

# def main():
#     wb = Workbook()
#     if "Sheet" in wb.sheetnames:
#         wb.remove(wb["Sheet"])

#     for zone in zones:
#         ws = wb.create_sheet(title=zone)
#         row_cursor = 1
#         for indic in indicateurs:
#             folder = get_model_folder_name(indic, zone)
#             folder_path = os.path.join(root_dir, folder)
#             pred_path = os.path.join(folder_path, "predictions.csv")
#             fut_path = os.path.join(folder_path, "predictions_futur.csv")

#             # Bloc predictions.csv
#             title1 = f"{indic} - Prédictions"
#             df_pred = try_read_csv(pred_path)
#             if df_pred is not None:
#                 last_row1 = write_table_with_title(ws, df_pred, title1, row_cursor, 1)
#             else:
#                 last_row1 = write_error(ws, f"Fichier absent : {pred_path}", row_cursor, 1)

#             # Bloc predictions_futur.csv (côte à côte)
#             title2 = f"{indic} - Prédictions futures"
#             df_fut = try_read_csv(fut_path)
#             if df_fut is not None:
#                 fut_start_row = row_cursor
#                 fut_start_col = df_pred.shape[1] + 3 if df_pred is not None else 8
#                 write_table_with_title(ws, df_fut, title2, fut_start_row, fut_start_col)
#             else:
#                 fut_start_row = row_cursor
#                 fut_start_col = (df_pred.shape[1] + 3) if df_pred is not None else 8
#                 write_error(ws, f"Fichier absent : {fut_path}", fut_start_row, fut_start_col)

#             # Tableau d'analyse juste en dessous des deux
#             analysis_row = max(last_row1, row_cursor) + 2 # 1 ligne vide
#             if (df_pred is not None) and (df_fut is not None):
#                 try:
#                     df_analysis = compute_analysis_table(df_fut, df_pred, indic)
#                     last_analysis_row = write_table_with_title(ws, df_analysis, f"{indic} - Analyse (futur/test/réel)", analysis_row, 1)
#                     # Ajout du graphique à droite du tableau d'analyse
#                     add_analysis_chart(ws, df_analysis, analysis_row, 1)
#                 except Exception as e:
#                     last_analysis_row = write_error(ws, f"Erreur analyse : {str(e)}", analysis_row, 1)
#             else:
#                 last_analysis_row = write_error(ws, f"Impossible de faire l'analyse (fichiers manquants)", analysis_row, 1)

#             # Sauter 5 lignes après le plus bas
#             row_cursor = max(last_row1, fut_start_row, last_analysis_row) + 15
#         autofit_columns(ws)
#     wb.save(output_path)
#     print(f"Fichier Excel sauvegardé sous : {output_path}")

# if __name__ == "__main__":
#     main()




######################################################################

import os
import pandas as pd
import xlsxwriter

def compute_analysis_table(df_fut, df_pred, indicateur):
    # Y réel ("y" ou "Y")
    y_col = "y" if "y" in df_pred.columns else "Y" if "Y" in df_pred.columns else None
    if y_col is None:
        raise ValueError("Impossible de trouver la colonne 'y' ou 'Y' dans predictions.csv")
    merge_cols = ["Annee", "Semaine"]
    merged = pd.merge(
        df_fut[merge_cols + ["y_pred"]].rename(columns={"y_pred": "y_pred_future"}),
        df_pred[merge_cols + ["y_pred", y_col]].rename(columns={"y_pred": "y_pred_test", y_col: "Y"}),
        on=merge_cols, how="inner"
    )
    
    # Calcul des deltas selon l'indicateur
    if indicateur == "Dp_moy":
        # Pour Dp_moy : différence simple (numérique)
        merged["Delta Y futur"] = merged["Y"] - merged["y_pred_future"]
        merged["Delta Y test"] = merged["Y"] - merged["y_pred_test"]
    else:
        # Pour les autres indicateurs : ratio décimal (sera formaté en % par Excel)
        merged["Delta Y futur"] = merged.apply(
            lambda row: (row['y_pred_future']/row['Y']-1) if row['Y'] != 0 else None, axis=1
        )
        merged["Delta Y test"] = merged.apply(
            lambda row: (row['y_pred_test']/row['Y']-1) if row['Y'] != 0 else None, axis=1
        )
    
    result = merged[["Annee", "Semaine", "y_pred_future", "y_pred_test", "Y", "Delta Y futur", "Delta Y test"]].copy()
    return result
    
def autofit_columns(worksheet, df, start_row, start_col):
    """Auto-ajuste la largeur des colonnes en fonction du contenu"""
    for col_idx, column_name in enumerate(df.columns):
        # Calculer la largeur max nécessaire pour cette colonne
        max_length = len(str(column_name))  # Longueur du header
        
        # Vérifier la longueur de chaque valeur dans la colonne
        for value in df.iloc[:, col_idx]:
            if value is not None:
                value_length = len(str(value))
                max_length = max(max_length, value_length)
        
        # Ajouter un peu d'espace supplémentaire et limiter la largeur max
        column_width = min(max_length + 2, 50)  # Max 50 caractères
        
        # Appliquer la largeur à la colonne
        worksheet.set_column(start_col + col_idx, start_col + col_idx, column_width)


def write_table_with_title(worksheet, df, title, start_row, start_col, workbook):
    # Format pour le titre
    title_format = workbook.add_format({'bold': True, 'font_size': 18})
    # Format pour les bordures
    border_format = workbook.add_format({'border': 1})
    # Format pourcentage
    percent_format = workbook.add_format({'num_format': '0.00%', 'border': 1})
    
    # Écrire le titre
    worksheet.write(start_row, start_col, title, title_format)
    
    # Écrire les headers
    for col, header in enumerate(df.columns):
        worksheet.write(start_row + 1, start_col + col, header, border_format)
    
    # Écrire les données
    for row_idx, row in df.iterrows():
        for col_idx, value in enumerate(row):
            # Formater en pourcentage les colonnes Delta (sauf pour Dp_moy)
            if "Delta" in str(df.columns[col_idx]) and "Dp_moy" not in title:
                worksheet.write(start_row + 2 + row_idx, start_col + col_idx, value, percent_format)
            else:
                worksheet.write(start_row + 2 + row_idx, start_col + col_idx, value, border_format)
    
    # AJOUT : Auto-ajustement des colonnes
    autofit_columns(worksheet, df, start_row, start_col)
    
    return start_row + len(df) + 2

def add_analysis_chart(worksheet, df, start_row, start_col, workbook):
    # Créer le graphique principal (lignes)
    chart1 = workbook.add_chart({'type': 'line'})
    
    # Calculer les plages de données
    data_start_row = start_row + 2
    data_end_row = start_row + 1 + len(df)
    
    # Créer les labels d'axes X personnalisés (Année-Semaine)
    categories = []
    for i in range(len(df)):
        row_data = df.iloc[i]
        annee = int(row_data['Annee'])
        semaine = int(row_data['Semaine'])
        categories.append(f"{annee}-{semaine:02d}")
    
    # Écrire les catégories dans une colonne temporaire
    cat_col = start_col + len(df.columns) + 1
    for i, cat in enumerate(categories):
        worksheet.write(data_start_row + i, cat_col, cat)
    
    # Ajouter les séries de lignes avec des noms COURTS
    # Y réel (bleu)
    chart1.add_series({
        'name': 'Y réel',
        'categories': [worksheet.name, data_start_row, cat_col, data_end_row, cat_col],
        'values': [worksheet.name, data_start_row, start_col + 4, data_end_row, start_col + 4],
        'line': {'color': '#3050F8', 'width': 2}
    })
    
    # y_pred_test (rouge)
    chart1.add_series({
        'name': 'Pred test',  # Plus court
        'categories': [worksheet.name, data_start_row, cat_col, data_end_row, cat_col],
        'values': [worksheet.name, data_start_row, start_col + 3, data_end_row, start_col + 3],
        'line': {'color': '#FF0000', 'width': 2}
    })
    
    # y_pred_future (vert)
    chart1.add_series({
        'name': 'Pred future',  # Plus court
        'categories': [worksheet.name, data_start_row, cat_col, data_end_row, cat_col],
        'values': [worksheet.name, data_start_row, start_col + 2, data_end_row, start_col + 2],
        'line': {'color': '#00B050', 'width': 2}
    })
    
    # Créer le deuxième graphique (barres) séparément
    chart2 = workbook.add_chart({'type': 'column'})
    
    # Delta Y future (vert) - BARRE
    chart2.add_series({
        'name': 'Δ futur',  # Très court avec symbole delta
        'categories': [worksheet.name, data_start_row, cat_col, data_end_row, cat_col],
        'values': [worksheet.name, data_start_row, start_col + 5, data_end_row, start_col + 5],
        'fill': {'color': '#00B050', 'transparency': 50},
        'border': {'color': '#00B050'},
        'y2_axis': True
    })
    
    # Delta Y test (rouge) - BARRE
    chart2.add_series({
        'name': 'Δ test',  # Très court avec symbole delta
        'categories': [worksheet.name, data_start_row, cat_col, data_end_row, cat_col],
        'values': [worksheet.name, data_start_row, start_col + 6, data_end_row, start_col + 6],
        'fill': {'color': '#FF0000', 'transparency': 50},
        'border': {'color': '#FF0000'},
        'y2_axis': True
    })
    
    # Configuration des axes du graphique barres AVANT combine
    chart2.set_y_axis({'visible': False})
    chart2.set_y2_axis({
        'name': 'Delta',
        'visible': True,
        'major_gridlines': {'visible': False}
    })
    
    # Combiner les deux graphiques
    chart1.combine(chart2)
    
    # Configuration du graphique principal
    chart1.set_title({'name': 'Comparaison Y / Prédictions'})
    chart1.set_x_axis({
        'name': 'Année-Semaine',
        'text_axis': True,
        'label_position': 'low'
    })
    chart1.set_y_axis({
        'name': 'Valeur',
        'visible': True
    })
    chart1.set_y2_axis({
        'name': 'Delta',
        'visible': True,
        'major_gridlines': {'visible': False}
    })
    chart1.set_size({'width': 1000, 'height': 480})  # Plus large pour la légende
    chart1.set_legend({
        'position': 'right',  # Retour à droite
        'layout': {
            'x': 0.92,   # Plus à droite
            'y': 0.15,   # Centré verticalement
            'width': 0.15,  # Plus large
            'height': 0.7   # Plus haut
        }
    })
    
    # Calculer la position du graphique
    chart_col = start_col + len(df.columns) + 3
    chart_row = start_row + 2
    
    # Insérer le graphique combiné
    worksheet.insert_chart(chart_row, chart_col, chart1)

def write_error(worksheet, message, start_row, start_col, workbook):
    error_format = workbook.add_format({'bold': True, 'font_color': 'red'})
    worksheet.write(start_row, start_col, message, error_format)
    return start_row

def try_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def get_model_folder_name(indic, zone):
    return f"models_{indic}_{zone}"

# ===========================
# PARAMÈTRES À MODIFIER :
# ===========================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
root_dir = os.path.join(project_root, "data", f"models_data_{12}")
zones = [
    "Bananord", "Banasud", "Brimbo Zone 1", "Brimbo Zone 2", "Broukro",
    "Fleuve Zone 1", "Fleuve Zone 2", "Sindressou", "Spadi Extension",
    "Spadi Zone 1", "Spadi Zone 2", "Tiassalé Zone 1", "Tiassalé Zone 2"
]
indicateurs = [
    "Dp_moy", "Etat_devolution_moy", "Pjft_moyen",
    "Pjfn_moyen", "Nff_moyen", "Nfr_moyen"
]
output_path = r"C:\Users\hamici-k\Desktop\analyse_previsions_zones.xlsx"
# ===========================
# FIN DES PARAMÈTRES
# ===========================

def main():
    workbook = xlsxwriter.Workbook(output_path)
    
    for zone in zones:
        worksheet = workbook.add_worksheet(zone)
        row_cursor = 0
        
        for indic in indicateurs:
            folder = get_model_folder_name(indic, zone)
            folder_path = os.path.join(root_dir, folder)
            pred_path = os.path.join(folder_path, "predictions.csv")
            fut_path = os.path.join(folder_path, "predictions_futur.csv")

            # Bloc predictions.csv
            title1 = f"{indic} - Prédictions"
            df_pred = try_read_csv(pred_path)
            if df_pred is not None:
                last_row1 = write_table_with_title(worksheet, df_pred, title1, row_cursor, 0, workbook)
            else:
                last_row1 = write_error(worksheet, f"Fichier absent : {pred_path}", row_cursor, 0, workbook)

            # Bloc predictions_futur.csv (côte à côte)
            title2 = f"{indic} - Prédictions futures"
            df_fut = try_read_csv(fut_path)
            if df_fut is not None:
                fut_start_row = row_cursor
                fut_start_col = df_pred.shape[1] + 2 if df_pred is not None else 7
                write_table_with_title(worksheet, df_fut, title2, fut_start_row, fut_start_col, workbook)
            else:
                fut_start_row = row_cursor
                fut_start_col = (df_pred.shape[1] + 2) if df_pred is not None else 7
                write_error(worksheet, f"Fichier absent : {fut_path}", fut_start_row, fut_start_col, workbook)

            # Tableau d'analyse juste en dessous
            analysis_row = max(last_row1, row_cursor) + 1
            if (df_pred is not None) and (df_fut is not None):
                try:
                    df_analysis = compute_analysis_table(df_fut, df_pred, indic)
                    last_analysis_row = write_table_with_title(worksheet, df_analysis, f"{indic} - Analyse (futur/test/réel)", analysis_row, 0, workbook)
                    # Ajout du graphique à droite du tableau d'analyse
                    add_analysis_chart(worksheet, df_analysis, analysis_row, 0, workbook)
                except Exception as e:
                    last_analysis_row = write_error(worksheet, f"Erreur analyse : {str(e)}", analysis_row, 0, workbook)
            else:
                last_analysis_row = write_error(worksheet, f"Impossible de faire l'analyse (fichiers manquants)", analysis_row, 0, workbook)

            # Sauter 21 lignes après le plus bas
            row_cursor = max(last_row1, fut_start_row, last_analysis_row) + 21
    
    workbook.close()
    print(f"Fichier Excel sauvegardé sous : {output_path}")

if __name__ == "__main__":
    main()

