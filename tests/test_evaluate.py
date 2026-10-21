"""Testes das métricas de avaliação (Etapa 3)."""

import numpy as np
import pytest

from churn.evaluate import evaluate, precision_at_k


def test_precision_at_k_conta_churn_entre_os_de_maior_risco():
    """Top 30% de 10 clientes = 3 clientes; 2 deles cancelaram."""
    y_true = [1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    y_proba = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

    assert precision_at_k(y_true, y_proba, k_percent=0.3) == pytest.approx(2 / 3)


def test_precision_at_k_nao_depende_da_ordem_das_linhas():
    """O ranking vem das probabilidades, não da posição do cliente na base."""
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    shuffle = np.random.default_rng(42).permutation(len(y_true))

    assert precision_at_k(y_true[shuffle], y_proba[shuffle], k_percent=0.3) == pytest.approx(2 / 3)


def test_precision_at_k_extremos():
    """Ranking perfeito dá 1,0; ranking invertido dá 0,0."""
    y_true = [1, 1, 0, 0]

    assert precision_at_k(y_true, [0.9, 0.8, 0.2, 0.1], k_percent=0.5) == 1.0
    assert precision_at_k(y_true, [0.1, 0.2, 0.8, 0.9], k_percent=0.5) == 0.0


def test_precision_at_k_usa_pelo_menos_um_cliente():
    """Base pequena demais para o K pedido ainda avalia o cliente de maior risco."""
    y_true = [0, 1, 0, 0, 0]
    y_proba = [0.1, 0.9, 0.2, 0.3, 0.4]

    assert precision_at_k(y_true, y_proba, k_percent=0.1) == 1.0


def test_evaluate_devolve_todas_as_colunas_da_tabela_comparativa():
    """As chaves viram colunas de docs/model_comparison.csv."""
    y_true = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    y_proba = np.array([0.9, 0.2, 0.7, 0.4, 0.1, 0.6, 0.3, 0.8, 0.45, 0.05])

    metrics = evaluate(y_true, (y_proba >= 0.5).astype(int), y_proba)

    assert set(metrics) == {"precision", "recall", "f1", "roc_auc", "pr_auc", "precision_at_k"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
