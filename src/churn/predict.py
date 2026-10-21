"""Predição com o modelo campeão (Etapa 3).

Responsabilidade única: carregar o Pipeline treinado e transformar dados
brutos de clientes em probabilidade e classe de churn. A API consome este
módulo, mas ele não depende dela — pode ser usado de um notebook ou script.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn import config


def load_model(path: Path | str | None = None) -> Pipeline | None:
    """Carrega o Pipeline treinado; devolve None se o artefato não existir."""
    path = Path(path or config.MODEL_PATH)
    if not path.exists():
        return None
    return joblib.load(path)


def predict_proba(model: Pipeline, customers: dict | list[dict] | pd.DataFrame) -> np.ndarray:
    """Probabilidade de churn para um ou mais clientes.

    Recebe os dados brutos, nos mesmos valores do dataset original ("Yes",
    "Fiber optic", ...): toda a codificação acontece dentro do Pipeline.
    As colunas são reordenadas conforme `config.FEATURES`, então a ordem das
    chaves de entrada não importa.
    """
    if isinstance(customers, dict):
        customers = [customers]
    X = pd.DataFrame(customers)[config.FEATURES]
    return model.predict_proba(X)[:, 1]


def classify(proba: float, threshold: float = config.DECISION_THRESHOLD) -> int:
    """Classe prevista (1 = churn) a partir da probabilidade."""
    return int(proba >= threshold)
