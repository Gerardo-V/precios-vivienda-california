# src/zonas_utils.py
import pandas as pd


def asignar_zona_california(df):
    """
    Asigna una de las 6 zonas de California con proporciones exactas:
    - Jefferson domina el norte.
    - North California es una franja más compacta debajo.
    - Silicon Valley y West California forman la franja costera continua (en contacto).
    - Central California ocupa todo el interior este sin bloquear la costa.
    - South California cubre adecuadamente el extremo sur.
    """
    zonas = []

    for la, lo in zip(df['latitude'], df['longitude']):
        # 1. Jefferson (Norte superior, amplio y dominante)
        if la >= 39.6:
            zonas.append('Jefferson')

        # 2. North California (Franja norte intermedia más compacta)
        elif la >= 38.3:
            zonas.append('North California')

        # 3. Silicon Valley (Costa centro-norte)
        elif la >= 35.8 and lo < -121.5:
            zonas.append('Silicon Valley')
        elif la >= 35.5 and la < 37 and lo < -120.8:
            zonas.append('Silicon Valley')

        # 4. West California (Costa sur/central, conectada directamente debajo de Silicon Valley)
        elif la >= 33.3 and la < 35.5 and lo < -117.9:
            zonas.append('West California')

        # 5. South California (Extremo sur del estado)
        elif la < 35.5:
            zonas.append('South California')

        # 6. Central California (El gran valle interior y este)
        else:
            zonas.append('Central California')

    df['zona_california'] = zonas
    return df

# ///////////////////////////////////////////////////////////////////////////

import matplotlib.pyplot as plt
import seaborn as sns

def graficar_media_mediana_ubicacion(df, col_ubicacion="zona_california", col_precio="median_house_value"):
    resumen = (
        df.groupby(col_ubicacion)[col_precio]
        .agg(Media="mean", Mediana="median")
        .reset_index()
        .melt(id_vars=col_ubicacion, value_vars=["Media", "Mediana"], var_name="Estadistico", value_name="Precio")
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=resumen, x=col_ubicacion, y="Precio", hue="Estadistico")
    plt.title("Media y Mediana del Precio por Ubicación")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


#//////////////////////////////////////////////////////////////////////////////



def graficar_conteo_objetivo(y, titulo="Distribución de la variable objetivo"):
    conteo = y.value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=conteo.index.astype(str),
        y=conteo.values,
        hue=conteo.index.astype(str),
        palette="viridis",
        legend=False
    )

    plt.title(titulo)
    plt.xlabel("Clase")
    plt.ylabel("Cantidad de registros")
    plt.tight_layout()
    plt.show()