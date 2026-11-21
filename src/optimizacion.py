import time
from scipy.stats import randint
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real

DESCRIPCION_PARAMETROS_RF = {
    "n_estimators": "Número de árboles del bosque: más árboles suelen estabilizar la predicción, pero aumentan el tiempo de cómputo.",
    "max_depth": "Profundidad máxima de cada árbol: limita qué tanto se ramifica para evitar que memorice el entrenamiento (sobreajuste).",
    "min_samples_leaf": "Mínimo de muestras requeridas en una hoja final: suaviza las decisiones y evita hojas con muy pocos datos.",
}

DESCRIPCION_PARAMETROS_GB = {
    "n_estimators": "Número de etapas/árboles secuenciales: cada nuevo árbol corrige errores de los anteriores.",
    "learning_rate": "Tasa de aprendizaje: controla cuánto aporta cada árbol nuevo; valores menores requieren más árboles pero generalizan mejor.",
    "max_depth": "Profundidad máxima de cada árbol débil: controla la complejidad de las interacciones entre variables.",
}


def _imprimir_reporte(titulo, tecnica, descripciones, mejores_params, modelo, X_train, y_train, X_test, y_test, tiempo_s):
    y_pred_tr = modelo.predict(X_train)
    y_pred_te = modelo.predict(X_test)

    f1_tr = f1_score(y_train, y_pred_tr, average="macro")
    f1_te = f1_score(y_test, y_pred_te, average="macro")
    rec_te = recall_score(y_test, y_pred_te, average="macro")
    acc_te = accuracy_score(y_test, y_pred_te)

    print(f"=== {titulo} ({tecnica}) ===")
    print(f"Tiempo de ejecución: {tiempo_s:.1f} s\n")

    print("Hiperparámetros ajustados y qué hacen:")
    for param, desc in descripciones.items():
        print(f"  - {param}: {desc}")

    print("\nMejores hiperparámetros encontrados:")
    for k, v in dict(mejores_params).items():
        print(f"  - {k} = {v}")

    print("\nMétricas del mejor modelo (Test):")
    print(f"  - F1-Score (macro): {f1_te:.4f}  (Train: {f1_tr:.4f})")
    print(f"  - Recall (macro):   {rec_te:.4f}")
    print(f"  - Accuracy:         {acc_te:.4f}")


def optimizar_random_forest(X_train, y_train, X_test, y_test, n_iter=12, cv_splits=3, random_state=454, n_jobs=2):
    t0 = time.time()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    param_dist = {
        "n_estimators": randint(100, 260),
        "max_depth": randint(4, 15),
        "min_samples_leaf": randint(2, 12),
    }

    busqueda = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=random_state, n_jobs=n_jobs),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    busqueda.fit(X_train, y_train)

    _imprimir_reporte(
        titulo="Modelo: Random Forest Classifier",
        tecnica="Búsqueda Aleatoria (RandomizedSearchCV)",
        descripciones=DESCRIPCION_PARAMETROS_RF,
        mejores_params=busqueda.best_params_,
        modelo=busqueda.best_estimator_,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        tiempo_s=time.time() - t0,
    )
    return busqueda.best_estimator_, busqueda.best_params_


def optimizar_gradient_boosting(X_train, y_train, X_test, y_test, n_iter=12, cv_splits=3, random_state=454, n_jobs=2):
    t0 = time.time()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    espacio = {
        "n_estimators": Integer(80, 200),
        "learning_rate": Real(0.02, 0.2, prior="log-uniform"),
        "max_depth": Integer(2, 5),
    }

    busqueda = BayesSearchCV(
        estimator=GradientBoostingClassifier(random_state=random_state),
        search_spaces=espacio,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    busqueda.fit(X_train, y_train)

    _imprimir_reporte(
        titulo="Modelo: Gradient Boosting Classifier",
        tecnica="Optimización Bayesiana (BayesSearchCV)",
        descripciones=DESCRIPCION_PARAMETROS_GB,
        mejores_params=busqueda.best_params_,
        modelo=busqueda.best_estimator_,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        tiempo_s=time.time() - t0,
    )
    return busqueda.best_estimator_, busqueda.best_params_