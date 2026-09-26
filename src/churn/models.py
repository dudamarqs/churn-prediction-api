"""Catálogo dos modelos candidatos.

Cada entrada é um Pipeline completo (pré-processamento + estimador), de
modo que treino, validação cruzada e inferência na API sigam exatamente
as mesmas etapas de transformação.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from churn import config
from churn.features import build_preprocessor


def build_pipeline(estimator) -> Pipeline:
    """Encadeia o pré-processador padrão com o estimador informado."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def get_candidates() -> dict[str, Pipeline]:
    """Modelos comparados na Etapa 2, na ordem exigida pelo enunciado.

    `class_weight="balanced"` é aplicado ao baseline e à Random Forest
    para compensar o desbalanceamento (73,5% / 26,5%). O MLPClassifier do
    Scikit-Learn não expõe esse parâmetro — é uma limitação conhecida a
    registrar no Model Card.
    """
    return {
        "logistic_regression": build_pipeline(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=config.RANDOM_STATE,
            )
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=config.RANDOM_STATE,
            )
        ),
        "mlp": build_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                early_stopping=True,
                random_state=config.RANDOM_STATE,
            )
        ),
    }


#: Modelo de referência para a comparação campeão vs baseline.
BASELINE_NAME = "logistic_regression"
