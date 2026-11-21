import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _agrupar_caracteristicas(col: str) -> str:
    """Agrupa las variables """
    if col.startswith('zona_california'):
        return 'Zona Geográfica (Total)'
    elif col.startswith('ocean_proximity'):
        return 'Proximidad Océano (Total)'
    return col


def graficar_importancia_agrupada(modelos: dict, nombre_modelo: str, columnas: list, figsize=(10, 6)):
    """
    Calcula, agrupa y grafica variables para un modelo dado.
    """
    if nombre_modelo not in modelos:
        raise ValueError(f"El modelo '{nombre_modelo}' no existe. Opciones disponibles: {list(modelos.keys())}")

    modelo = modelos[nombre_modelo]

    # Crear DataFrame con importancias
    df_importancias = pd.DataFrame({
        'Caracteristica': columnas,
        'Importancia': modelo.feature_importances_
    }) #

    # Agrupar y sumar importancias
    df_importancias['Grupo'] = df_importancias['Caracteristica'].apply(_agrupar_caracteristicas)
    df_agrupado = (
        df_importancias.groupby('Grupo', as_index=False)['Importancia']
        .sum()
        .sort_values(by='Importancia', ascending=False)
    )

    # Generar gráfico
    plt.figure(figsize=figsize)
    sns.barplot(
        x='Importancia',
        y='Grupo',
        data=df_agrupado,
        palette='viridis',
        hue='Grupo',
        legend=False
    )

    plt.title(f'Importancia Global de Variables Agrupadas - {nombre_modelo}')
    plt.xlabel('Grado de Importancia Total')
    plt.ylabel('Variables / Grupos')
    plt.tight_layout()
    plt.show()

