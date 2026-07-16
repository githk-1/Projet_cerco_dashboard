import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from datetime import datetime
import plotly.express as px 
import plotly.graph_objects as go
import plotly.express as px

# Nouvelle fonction : occurence_traitement
def occurence_traitement(df, zone_col="Zone_traitement", year_col="Annee", filter_col="Intrant_revu", filter_value="Aucun traitement", title=None, xlabel=None, ylabel="Occurence"):
    """
    Affiche un barplot interactif Plotly de l'occurrence des traitements par zone de traitement et par année.
    Chaque zone (x) a une barre par année (hue).
    
    Parameters:
    -----------
    df : DataFrame
        Le DataFrame contenant les données.
    zone_col : str
        Nom de la colonne pour l'axe X (zone de traitement).
    year_col : str
        Nom de la colonne pour l'année (hue).
    filter_col : str
        Colonne à filtrer pour exclure les "Aucun traitement" (par défaut : "Intrant_revu").
    filter_value : str
        Valeur à exclure (par défaut : "Aucun traitement").
    title : str
        Titre du graphique.
    xlabel : str
        Label de l'axe X.
    ylabel : str
        Label de l'axe Y (par défaut : "Occurence").
    """
    import plotly.express as px
    # Filtrer les données
    df_plot = df[df[filter_col] != filter_value].copy()
    if len(df_plot) == 0:
        return None
    # S'assurer que les colonnes sont des chaînes pour l'affichage
    df_plot[zone_col] = df_plot[zone_col].astype(str)
    df_plot[year_col] = df_plot[year_col].astype(str)
    # Grouper par zone et année
    dist = df_plot.groupby([zone_col, year_col]).size().reset_index(name="Occurence")
    dist = dist.sort_values([zone_col, year_col])
    fig = px.bar(
        dist,
        x=zone_col,
        y="Occurence",
        color=year_col,
        barmode="group",
        title=title if title else f"Occurence du traitement par zone et année (hors '{filter_value}')",
        labels={zone_col: xlabel if xlabel else zone_col, "Occurence": ylabel, year_col: "Année"},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        width=1000,
        yaxis_title=ylabel,
        xaxis_title=xlabel if xlabel else zone_col,
        margin=dict(t=60),
        showlegend=True
    )
    fig.update_traces(marker_line_color='rgb(180,180,180)', marker_line_width=1)
    return fig
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from datetime import datetime
import plotly.express as px 
import plotly.graph_objects as go
import plotly.express as px

def plot_intrant_moyen(df, cat_col, val_col, title, x_label, y_label):
    """
    Crée un barplot interactif Plotly (bleu, labels clairs) pour la quantité moyenne d'intrant par catégorie.
    
    Parameters:
    -----------
    df : DataFrame
        Le DataFrame contenant les données (déjà filtrées si besoin)
    cat_col : str
        Nom de la colonne de catégorie (ex: 'Intrant_revu')
    val_col : str
        Nom de la colonne de valeur (ex: 'Quantite' ou 'Quantite_huile')
    title : str
        Titre du graphique
    x_label : str
        Label de l'axe des x
    y_label : str
        Label de l'axe des y
    """
    # plotly.express doit être importé en haut du fichier, pas ici
    # Calcul de la moyenne par catégorie
    import plotly.express as px
    df_group = df.groupby([cat_col,"Annee"])[val_col].mean().reset_index()
    df_group = df_group.sort_values(val_col, ascending=False)
    fig = px.bar(
        df_group,
        x=cat_col,
        y=val_col,
        text=val_col,
        color= "Annee",
        labels={cat_col: x_label, val_col: y_label},
        color_discrete_sequence=["#1f77b4"]
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='auto',textfont_size=10)
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        width=1000,
        yaxis_title=y_label,
        xaxis_title=x_label,
        margin=dict(t=60),
    )
    return fig




def plot_barp(data, x, y, hue=None, title=None, xlabel=None, ylabel=None, rotation=0):
    """
    Crée un graphique en barres simple avec annotations espacées automatiquement
    
    Parameters:
    -----------
    data : DataFrame
        Le DataFrame contenant les données
    x : str
        Nom de la colonne pour l'axe x
    y : str
        Nom de la colonne pour l'axe y
    hue : str, optional
        Nom de la colonne pour la différenciation par couleur
    title : str, optional
        Titre du graphique
    xlabel : str, optional
        Label de l'axe x
    ylabel : str, optional
        Label de l'axe y
    rotation : int, optional
        Rotation des labels de l'axe x
    """
    
    plt.figure(figsize=(10, 6))
    # Créer le bar plot sans barres d'erreur
    ax = sns.barplot(data=data, x=x, y=y, hue=hue, ci=None)
    fig = ax.get_figure()
    # Ajouter les annotations avec ajustement automatique de la position
    for i, container in enumerate(ax.containers):
        # Augmenter le padding pour chaque groupe de barres
        ax.bar_label(container, fmt='%.1f', padding=3 + (i * 3), size=7)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if hue:
        plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    return fig


def plot_bar(data, x, y, hue=None, title=None, xlabel=None, ylabel=None, rotation=0):
    """
    Crée un graphique à barres interactif avec une palette de couleurs prédéfinie.
    
    Parameters:
    -----------
    data : DataFrame
        Le DataFrame contenant les données
    x : str
        Nom de la colonne pour l'axe X
    y : str
        Nom de la colonne pour l'axe Y
    hue : str, optional
        Nom de la colonne pour la différenciation par couleur
    title : str, optional
        Titre du graphique
    xlabel : str, optional 
        Étiquette de l'axe X
    ylabel : str, optional
        Étiquette de l'axe Y
    rotation : int, optional
        Rotation des étiquettes de l'axe X
    """
    
    fig = px.bar(
        data,
        x=x,
        y=y,
        color=hue,
        title=title,
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(
        xaxis_title=xlabel if xlabel else x,
        yaxis_title=ylabel if ylabel else y,
        title_x=0.5,
        showlegend=True if hue else False,
        xaxis=dict(
            tickangle=rotation,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(220, 220, 220, 0.4)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(220, 220, 220, 0.4)'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(
            family="Arial",
            size=12,
            color='#444444'
        )
    )
    fig.update_traces(
        marker_line_color='rgb(180,180,180)',
        marker_line_width=1
    )
    return fig




def plot_hist(df, x, bins=10, title="Mon titre", xlabel=None, ylabel="Fréquence", couleur="blue", taille=(10,6)):
    """
    Crée un histogramme interactif avec Plotly
    
    Parameters:
    -----------
    df : DataFrame
        Le DataFrame contenant les données
    x : str
        Nom de la colonne à représenter
    bins : int
        Nombre de bins pour l'histogramme
    titre : str
        Titre du graphique
    xlabel : str
        Titre de l'axe x
    ylabel : str
        Titre de l'axe y
    couleur : str
        Couleur de l'histogramme
    taille : tuple
        Dimensions du graphique (largeur, hauteur)
    """
    if xlabel is None:
        xlabel = x

    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[x],
        nbinsx=bins,
        marker_color=couleur,
        marker_line_color='black',
        marker_line_width=1,
        opacity=0.75,
        name=x
    ))

    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=taille[0]*100,
        height=taille[1]*100,
        showlegend=True,
        plot_bgcolor='white'
    )

    # Ajout de la grille
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    fig.show()

    
def plot_boxp(df, x, y, title =None, xlabel=None, ylabel=None, couleur="skyblue", 
              taille=(10, 6), hue=None, rotation=45, style=None):
    """
    Crée un box plot (boîte à moustaches) à partir d'un DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Le DataFrame contenant les données
    x : str
        Nom de la colonne pour l'axe x
    y : str
        Nom de la colonne pour l'axe y
    title : str, optional
        Titre du graphique
    xlabel : str, optional
        Label de l'axe x
    ylabel : str, optional
        Label de l'axe y
    couleur : str, optional
        Couleur du box plot (par défaut "skyblue")
    taille : tuple, optional
        Taille du graphique (largeur, hauteur)
    hue : str, optional
        Colonne pour la subdivision des box plots
    rotation : int, optional
        Rotation des labels de l'axe x en degrés
    style : dict, optional
        Dictionnaire de paramètres de style supplémentaires
    
    Returns:
    --------
    None
    """
    # Vérification des entrées
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un pandas DataFrame")
    if x not in df.columns or y not in df.columns:
        raise ValueError("Les colonnes spécifiées n'existent pas dans le DataFrame")
    
    # Configuration des labels par défaut
    xlabel = x if xlabel is None else xlabel
    ylabel = y if ylabel is None else ylabel
    title = f"Distribution de {y} par {x}" if title is None else title
    
    # Configuration du style
    style_params = {
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    }
    if style is not None:
        style_params.update(style)
        
    # Création du plot
    plt.figure(figsize=taille)
    with plt.style.context(style_params):
        # Création du box plot
        ax = sns.boxplot(x=x, y=y, data=df, hue=hue, color=couleur)
        
        # Personnalisation du graphique
        plt.title(title, pad=20, fontsize=12, fontweight='bold')
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.grid(True, axis='y')
        plt.tick_params(axis='x', rotation=rotation)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Ajustement automatique de la mise en page
        plt.tight_layout()
        
    plt.show()
    
    return ax


def plot_boxp_interactive(df, x, y, title=None, xlabel=None, ylabel=None, 
                         couleur=None, taille=(800, 500), hue=None, 
                         rotation=45, template='plotly_white'):
    """
    Crée un box plot interactif avec Plotly Express.
    
    Parameters:
    -----------
    df : pandas.DataFrame
    Le DataFrame contenant les données
    x : str
        Nom de la colonne pour l'axe x
    y : str
        Nom de la colonne pour l'axe y
    title : str, optional
        Titre du graphique
    xlabel : str, optional
        Label de l'axe x
    ylabel : str, optional
        Label de l'axe y
    couleur : str, optional
        Nom de la colonne pour la couleur des box plots
    taille : tuple, optional
        Taille du graphique (largeur, hauteur)
    hue : str, optional
        Colonne pour la subdivision des box plots (même que couleur)
    rotation : int, optional
        Rotation des labels de l'axe x en degrés
    template : str, optional
        Template Plotly à utiliser

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    import plotly.express as px

    # Vérification des entrées
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un pandas DataFrame")
    if x not in df.columns or y not in df.columns:
        raise ValueError("Les colonnes spécifiées n'existent pas dans le DataFrame")

    # Configuration des labels par défaut
    xlabel = x if xlabel is None else xlabel
    ylabel = y if ylabel is None else ylabel
    titre = f"Distribution de {y} par {x}" if title is None else title

    # Création du box plot interactif
    fig = px.box(
        df,
        x=x,
        y=y,
        color=couleur or hue,  # Utilise couleur ou hue si spécifié
        title=title,
        template=template,
        width=taille[0],
        height=taille[1]
    )

    # Personnalisation du layout
    fig.update_layout(
        title={
            'text': titre,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis={'tickangle': rotation},
        showlegend=True if (couleur or hue) else False,
        plot_bgcolor='white',
        boxgap=0.2,  # Espace entre les box plots
        legend=dict(
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=1.02
    )
    )

    # Ajout de la grille
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.2)'
    )

    return fig





def plot_line_interactive(
    df, 
    x, 
    y, 
    hue=None, 
    title=None, 
    xlabel=None, 
    ylabel=None, 
    taille=(800, 500),
    template='plotly_white',
    line_mode='lines',
    marker_size=6,
    line_width=2,
    grid=True,
    x_type=None,
    y_type=None,
    legend_position='right',
    custom_colors=None
):
    """
    [Documentation précédente maintenue...]
    """
    # Validation des entrées
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un pandas DataFrame")
    
    # Création d'une copie du DataFrame
    df_plot = df.copy()
    
    # Si nous travaillons avec des années, assurons-nous qu'elles sont dans le bon ordre
    if x == 'Annee' or (hue == 'Annee'):
        # Convertir en string mais garder l'ordre
        df_plot['Annee'] = pd.Categorical(
            df_plot['Annee'].astype(str),
            categories=sorted(df_plot['Annee'].unique().astype(str)),
            ordered=True
        )

    # Configuration des labels par défaut
    xlabel = x if xlabel is None else xlabel
    ylabel = y if ylabel is None else ylabel
    title = f"Évolution de {y} en fonction de {x}" if title is None else title

    # [Reste du code identique...]
    legend_positions = {
        'right': dict(yanchor="middle", y=0.5, xanchor="left", x=1.05),
        'top': dict(yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        'bottom': dict(yanchor="top", y=-0.2, xanchor="center", x=0.5),
        'left': dict(yanchor="middle", y=0.5, xanchor="right", x=-0.05)
    }

    # Création du graphique
    fig = px.line(
        df_plot,
        x=x,
        y=y,
        color=hue,
        title=title,
        template=template,
        width=taille[0],
        height=taille[1],
        color_discrete_sequence=custom_colors,
        category_orders={x: sorted(df_plot[x].unique())} if x == 'Annee' else None
    )

    # Mise à jour du layout
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        plot_bgcolor='white',
        legend=legend_positions.get(legend_position, legend_positions['right'])
    )

    # Configuration des traces
    fig.update_traces(
        mode=line_mode,
        marker=dict(size=marker_size),
        line=dict(width=line_width)
    )

    # Si c'est un axe avec des années, forcer le type catégoriel
    if x == 'Annee':
        fig.update_xaxes(type='category', categoryorder='array', 
                        categoryarray=sorted(df_plot[x].unique()))
    elif x_type:
        fig.update_xaxes(type=x_type)
        
    if y_type:
        fig.update_yaxes(type=y_type)

    # Configuration de la grille
    if grid:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    else:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

    return fig


def plot_numeric_distributions(df, figsize=(15, 10), columns=None):
    """
    Plot la distribution de toutes les variables numériques d'un DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
        figsize (tuple): Taille de la figure (width, height)
        columns (list, optional): Liste des noms de colonnes à inclure dans les distributions.
                                 Si None, toutes les colonnes numériques seront utilisées.
    """
    # Si des colonnes spécifiques sont demandées
    if columns is not None:
        # Vérifier que toutes les colonnes demandées existent
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Les colonnes suivantes n'existent pas dans le DataFrame : {missing_cols}")
        
        # Sélectionner les colonnes spécifiées
        selected_data = df[columns]
        
        # Vérifier que les colonnes sélectionnées sont numériques
        numeric_cols = selected_data.select_dtypes(include=['int64', 'float64', 'uint64', 'uint32', 'int32']).columns
        
        if numeric_cols.empty:
            raise ValueError("Aucune des colonnes spécifiées n'est numérique")
        
        if len(numeric_cols) != len(selected_data.columns):
            non_numeric = [col for col in columns if col not in numeric_cols]
            print(f"Attention : Les colonnes non numériques suivantes ont été exclues : {non_numeric}")
    else:
        # Comportement par défaut : sélectionner toutes les colonnes numériques
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'uint64', 'uint32', 'int32']).columns
        
        if numeric_cols.empty:
            raise ValueError("Le DataFrame ne contient aucune colonne numérique")
        
        if len(numeric_cols) != len(df.columns):
            print(f"Attention : {len(df.columns) - len(numeric_cols)} colonnes non numériques ont été exclues des distributions")
    
    # Calcul du nombre de lignes et colonnes pour le subplot
    n_cols = 3
    n_rows = (len(numeric_cols) - 1) // n_cols + 1

    # Sécuriser figsize pour éviter les erreurs matplotlib
    if not (isinstance(figsize, (tuple, list)) and len(figsize) == 2):
        figsize = (15, 10)
    else:
        figsize = (float(figsize[0]), float(figsize[1]))

    # Création de la figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('Distribution des variables numériques', fontsize=16)
    
    # Aplatir axes si une seule ligne
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot pour chaque variable
    for idx, col in enumerate(numeric_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        
        # Histogramme avec KDE
        sns.histplot(data=df, x=col, kde=True, bins=52, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'Distribution de {col}')
        axes[row, col_idx].tick_params(axis='x', rotation=45)
    
    # Supprimer les subplots vides
    for idx in range(len(numeric_cols), n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        fig.delaxes(axes[row, col_idx])
    
    plt.tight_layout()
    return fig





def plot_heatmap(data, title, xlabel, ylabel, cmap="YlOrRd", cbar_label="Valeurs Manquantes", fmt="d", figsize=(12, 8)):
    """
    Génère un heatmap à partir des données fournies.

    Parameters:
        data (DataFrame): Les données pivotées pour le heatmap.
        title (str): Le titre du graphique.
        xlabel (str): Le label de l'axe X.
        ylabel (str): Le label de l'axe Y.
        cmap (str): La palette de couleurs pour le heatmap. Par défaut "YlOrRd".
        cbar_label (str): Le label de la barre de couleur. Par défaut "Valeurs Manquantes".
        fmt (str): Le format des annotations dans le heatmap. Par défaut "d".
        figsize (tuple): La taille de la figure. Par défaut (12, 8).
    """
    plt.figure(figsize=figsize)
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, cbar_kws={'label': cbar_label})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data, method='pearson', figsize=(10, 8), annot=True, fmt='.2f', 
                          cmap='coolwarm', mask_upper=False, title=None, columns=None):
    """
    Plot a correlation matrix using seaborn.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input DataFrame containing the variables to correlate
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman', or 'kendall')
    figsize : tuple, default=(10, 8)
        Figure size (width, height)
    annot : bool, default=True
        Whether to annotate cells with numerical value
    fmt : str, default='.2f'
        Format of the annotations
    cmap : str, default='coolwarm'
        Color map to use
    mask_upper : bool, default=False
        If True, masks the upper triangle of the matrix
    title : str, default=None
        Title of the plot
    columns : list, default=None
        List of column names to include in correlation matrix.
        If None, all numeric columns will be used.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Si des colonnes spécifiques sont demandées
    if columns is not None:
        # Vérifier que toutes les colonnes demandées existent
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Les colonnes suivantes n'existent pas dans le DataFrame : {missing_cols}")
        
        # Sélectionner les colonnes spécifiées
        selected_data = data[columns]
        
        # Vérifier que les colonnes sélectionnées sont numériques
        numeric_data = selected_data.select_dtypes(include=['float64', 'int64','uint64', 'uint32', 'int32'])
        
        if numeric_data.empty:
            raise ValueError("Aucune des colonnes spécifiées n'est numérique")
        
        if len(numeric_data.columns) != len(selected_data.columns):
            non_numeric = [col for col in columns if col not in numeric_data.columns]
            print(f"Attention : Les colonnes non numériques suivantes ont été exclues : {non_numeric}")
    else:
        # Comportement par défaut : sélectionner toutes les colonnes numériques
        numeric_data = data.select_dtypes(include=['float64', 'int64','uint64', 'uint32', 'int32'])
        
        if numeric_data.empty:
            raise ValueError("Le DataFrame ne contient aucune colonne numérique")
        
        if len(numeric_data.columns) != len(data.columns):
            print(f"Attention : {len(data.columns) - len(numeric_data.columns)} colonnes non numériques ont été exclues de la matrice de corrélation")
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr(method=method)
    
    # Create mask for upper triangle if requested
    mask = np.triu(np.ones_like(corr_matrix)) if mask_upper else None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=annot, 
                fmt=fmt,
                cmap=cmap,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .5})
    
    # Set title if provided
    if title:
        plt.title(title)
        
    fig.tight_layout()
    
    return fig
    






def plot_boxp_interactive2(df, x, y, title=None, xlabel=None, ylabel=None, 
                         couleur=None, taille=(800, 500), hue=None, 
                         rotation=45, template='plotly_white', boxmean=True,
                         box_color='rgba(31, 119, 180, 0.6)', outline_color='rgba(31, 119, 180, 1)'):
    """
    Crée un box plot interactif avec Plotly, incluant l'affichage de la moyenne.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Le DataFrame contenant les données
    x : str
        Nom de la colonne pour l'axe x
    y : str
        Nom de la colonne pour l'axe y
    title : str, optional
        Titre du graphique
    xlabel : str, optional
        Label de l'axe x
    ylabel : str, optional
        Label de l'axe y
    couleur : str, optional
        Nom de la colonne pour la couleur des box plots
    taille : tuple, optional
        Taille du graphique (largeur, hauteur)
    hue : str, optional
        Colonne pour la subdivision des box plots (même que couleur)
    rotation : int, optional
        Rotation des labels de l'axe x en degrés
    template : str, optional
        Template Plotly à utiliser
    boxmean : bool or 'sd', optional
        Si True, affiche la moyenne comme un point
        Si 'sd', affiche la moyenne et l'écart-type
    box_color : str, optional
        Couleur de remplissage des boxplots (avec transparence)
    outline_color : str, optional
        Couleur du contour des boxplots (plus foncée)

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    
    # Vérification des entrées
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un pandas DataFrame")
    if x not in df.columns or y not in df.columns:
        raise ValueError("Les colonnes spécifiées n'existent pas dans le DataFrame")

    # Configuration des labels par défaut
    xlabel = x if xlabel is None else xlabel
    ylabel = y if ylabel is None else ylabel
    titre = f"Distribution de {y} par {x}" if title is None else title
    
    # Palette de couleurs avec versions transparentes pour les remplissages
    color_palette = {
        'blue': {'fill': 'rgba(31, 119, 180, 0.6)', 'outline': 'rgba(31, 119, 180, 1)'},
        'orange': {'fill': 'rgba(255, 127, 14, 0.6)', 'outline': 'rgba(255, 127, 14, 1)'},
        'green': {'fill': 'rgba(44, 160, 44, 0.6)', 'outline': 'rgba(44, 160, 44, 1)'},
        'red': {'fill': 'rgba(214, 39, 40, 0.6)', 'outline': 'rgba(214, 39, 40, 1)'},
        'purple': {'fill': 'rgba(148, 103, 189, 0.6)', 'outline': 'rgba(148, 103, 189, 1)'},
        'brown': {'fill': 'rgba(140, 86, 75, 0.6)', 'outline': 'rgba(140, 86, 75, 1)'},
        'pink': {'fill': 'rgba(227, 119, 194, 0.6)', 'outline': 'rgba(227, 119, 194, 1)'},
        'gray': {'fill': 'rgba(127, 127, 127, 0.6)', 'outline': 'rgba(127, 127, 127, 1)'},
        'olive': {'fill': 'rgba(188, 189, 34, 0.6)', 'outline': 'rgba(188, 189, 34, 1)'},
        'cyan': {'fill': 'rgba(23, 190, 207, 0.6)', 'outline': 'rgba(23, 190, 207, 1)'},
    }
    
    plotly_colors = list(color_palette.values())
    
    # Créer une figure
    fig = go.Figure()
    
    # Si une colonne de couleur est spécifiée, créer un box plot pour chaque groupe
    if couleur or hue:
        color_col = couleur or hue
        groups = df[color_col].unique()
        
        for i, group in enumerate(groups):
            group_data = df[df[color_col] == group]
            color_idx = i % len(plotly_colors)
            fill_color = plotly_colors[color_idx]['fill']
            line_color = plotly_colors[color_idx]['outline']
            
            # Ajouter un boxplot pour chaque groupe
            fig.add_trace(go.Box(
                x=group_data[x],
                y=group_data[y],
                name=str(group),
                boxmean=boxmean,  # Affiche la moyenne
                fillcolor=fill_color,  # Couleur de remplissage avec transparence
                line=dict(color=line_color),  # Contour plus foncé
                marker=dict(color=line_color, size=5),  # Points (outliers) de la même couleur
                whiskerwidth=0.8  # Largeur des whiskers
            ))
    else:
        # Si pas de colonne de couleur, créer un seul box plot
        fig.add_trace(go.Box(
            x=df[x],
            y=df[y],
            boxmean=boxmean,  # Affiche la moyenne
            fillcolor=box_color,  # Couleur de remplissage avec transparence
            line=dict(color=outline_color, width=1.5),  # Contour plus foncé
            marker=dict(color=outline_color, size=5),  # Points (outliers) de la même couleur
            whiskerwidth=0.8  # Largeur des whiskers
        ))
    
    # Personnalisation du layout
    fig.update_layout(
        title={
            'text': titre,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis={'tickangle': rotation},
        width=taille[0],
        height=taille[1],
        template=template,
        plot_bgcolor='white',
        boxgap=0.2,  # Espace entre les box plots
        boxgroupgap=0.1,  # Espace entre les groupes de box plots
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02
        ),
        showlegend=True if (couleur or hue) else False
    )

    # Ajout de la grille
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.2)'
    )

    return fig

def plot_boxplots_indic(df, indicators, title="Distribution des indicateurs", 
                             n_cols=4, height=400, width=1400, by_col=None):
    """
    Crée une grille de boxplots interactifs pour visualiser plusieurs indicateurs.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Le DataFrame contenant les données à visualiser
    indicators : list
        Liste des noms de colonnes à visualiser avec des boxplots
    title : str, optional
        Titre global du graphique
    n_cols : int, optional
        Nombre de colonnes dans la grille (défaut: 4)
    height : int, optional
        Hauteur par ligne de boxplots (sera multipliée par le nombre de lignes)
    width : int, optional
        Largeur totale de la figure
    by_col : str, optional
        Colonne à utiliser pour grouper les données (par exemple "Annee" ou "Domaine")
        Si spécifié, les boxplots seront groupés par cette variable
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figure interactive contenant tous les boxplots
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import math
    
    # Calculer le nombre de lignes nécessaires
    n_rows = math.ceil(len(indicators) / n_cols)
    total_height = height * n_rows
    
    # Créer une figure avec des sous-graphiques
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=[f"Distribution de {ind}" for ind in indicators]
    )
    
    # Pour chaque indicateur
    for i, indicator in enumerate(indicators):
        row = i // n_cols + 1  # Déterminer la ligne
        col = i % n_cols + 1   # Déterminer la colonne
        
        if by_col is None:
            # Cas simple: un boxplot par indicateur
            fig.add_trace(
                go.Box(
                    y=df[indicator],
                    name=indicator,
                    boxmean=True,  # Afficher la moyenne
                    fillcolor='rgba(31, 119, 180, 0.6)',  # Couleur de remplissage avec transparence
                    line=dict(color='rgba(31, 119, 180, 1)', width=1.5),  # Contour plus foncé
                    marker=dict(color='rgba(31, 119, 180, 1)', size=5),  # Couleur des points
                    whiskerwidth=0.8  # Largeur des whiskers
                ),
                row=row, col=col
            )
        else:
            # Cas avec regroupement: boxplot groupé par by_col
            # Utiliser des couleurs différentes pour chaque groupe
            colors = [
                ('rgba(31, 119, 180, 0.6)', 'rgba(31, 119, 180, 1)'),   # bleu
                ('rgba(255, 127, 14, 0.6)', 'rgba(255, 127, 14, 1)'),   # orange
                ('rgba(44, 160, 44, 0.6)', 'rgba(44, 160, 44, 1)'),     # vert
                ('rgba(214, 39, 40, 0.6)', 'rgba(214, 39, 40, 1)'),     # rouge
                ('rgba(148, 103, 189, 0.6)', 'rgba(148, 103, 189, 1)'), # violet
                ('rgba(140, 86, 75, 0.6)', 'rgba(140, 86, 75, 1)'),     # marron
                ('rgba(227, 119, 194, 0.6)', 'rgba(227, 119, 194, 1)'), # rose
                ('rgba(127, 127, 127, 0.6)', 'rgba(127, 127, 127, 1)'), # gris
            ]
            
            # Pour chaque valeur unique dans la colonne de regroupement
            for j, group in enumerate(sorted(df[by_col].unique())):
                group_data = df[df[by_col] == group]
                color_idx = j % len(colors)
                fill_color, line_color = colors[color_idx]
                
                # Légende visible uniquement dans le premier subplot
                show_legend = (i == 0)
                
                fig.add_trace(
                    go.Box(
                        y=group_data[indicator],
                        name=str(group),
                        legendgroup=str(group),  # Grouper les éléments de même groupe dans la légende
                        showlegend=show_legend,
                        boxmean=True,  # Afficher la moyenne
                        fillcolor=fill_color,
                        line=dict(color=line_color),
                        marker=dict(color=line_color, size=5),
                        whiskerwidth=0.8
                    ),
                    row=row, col=col
                )
    
    # Mise à jour du layout
    fig.update_layout(
        height=total_height,
        width=width,
        title_text=title,
        title_x=0.5,
        template='plotly_white',
        plot_bgcolor='white',
        boxmode='group' if by_col else 'overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ) if by_col else None
    )
    
    # Mise à jour des axes y pour ajouter des grilles
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)'
    )
    
    return fig