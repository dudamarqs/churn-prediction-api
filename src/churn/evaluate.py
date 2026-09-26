"""Métricas de avaliação definidas na Etapa 1 (ver docs/metrics.md)."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn import config


def precision_at_k(y_true, y_proba, k_percent: float = config.TOP_K_PERCENT) -> float:
    """Métrica de negócio: precisão entre os K% clientes de maior risco.

    Responde "de cada 100 clientes que o time de retenção abordar, quantos
    realmente cancelariam?" — que é o que determina a eficiência do
    orçamento de retenção.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    k = max(1, int(len(y_true) * k_percent))
    top_k_idx = np.argsort(y_proba)[::-1][:k]

    return float(y_true[top_k_idx].mean())


def evaluate(y_true, y_pred, y_proba) -> dict[str, float]:
    """Consolida as métricas técnicas e a métrica de negócio."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision_at_k": precision_at_k(y_true, y_proba),
    }
