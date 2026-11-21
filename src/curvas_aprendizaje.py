import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, learning_curve


def graficar_curva_aprendizaje(
    estimator,
    X,
    y,
    titulo="Curva de Aprendizaje",
    scoring="f1_macro",
    cv_splits=5,
    n_puntos=5,
):
  cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

  train_sizes, train_scores, val_scores = learning_curve(
      estimator=estimator,
      X=X,
      y=y,
      train_sizes=np.linspace(0.2, 1.0, n_puntos),
      cv=cv,
      scoring=scoring,
      n_jobs=-1,
  )

  train_mean = np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)
  val_mean = np.mean(val_scores, axis=1)
  val_std = np.std(val_scores, axis=1)

  plt.style.use("seaborn-v0_8-whitegrid")
  fig, ax = plt.subplots(figsize=(7.5, 4.5))

  ax.plot(
      train_sizes,
      train_mean,
      "o-",
      color="#D62728",
      label="Entrenamiento (Train)",
  )
  ax.fill_between(
      train_sizes,
      train_mean - train_std,
      train_mean + train_std,
      alpha=0.15,
      color="#D62728",
  )

  ax.plot(
      train_sizes,
      val_mean,
      "o-",
      color="#2CA02C",
      label="Validación cruzada (CV)",
  )
  ax.fill_between(
      train_sizes,
      val_mean - val_std,
      val_mean + val_std,
      alpha=0.15,
      color="#2CA02C",
  )

  ax.set_title(titulo)
  ax.set_xlabel("Tamaño del conjunto de entrenamiento")
  ax.set_ylabel(f"Score ({scoring})")
  ax.legend(loc="best", frameon=True)
  plt.tight_layout()
  plt.show()