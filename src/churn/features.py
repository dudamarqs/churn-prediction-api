"""Construção do pré-processador de features.

Mantido como um ColumnTransformer do Scikit-Learn para que o ajuste
(fit) aconteça sempre dentro de um Pipeline — evitando vazamento do
conjunto de teste durante a imputação e o escalonamento.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config


def build_preprocessor() -> ColumnTransformer:
    """Devolve o pré-processador aplicado a todos os modelos.

    Numéricas: imputação pela mediana + padronização. O escalonamento é
    indispensável para a Regressão Logística e para o MLPClassifier; é
    inócuo para modelos baseados em árvores, então manter um único
    pré-processador para os três mantém a comparação justa.

    Categóricas: imputação pela moda + one-hot. `handle_unknown="ignore"`
    garante que a API não quebre ao receber uma categoria não vista no
    treino — ela é codificada como zeros em vez de lançar exceção.
    """
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUM_FEATURES),
            ("cat", categorical, config.CAT_FEATURES),
        ],
        remainder="drop",
    )
