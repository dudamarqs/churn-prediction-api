"""Testes da escolha do modelo campeão (Etapa 3)."""

import joblib
import pandas as pd
import pytest

from churn import config
from churn.models import get_candidates
from churn.train import rank_models


def test_campeao_e_escolhido_pela_validacao_cruzada_nao_pelo_teste():
    """Quando CV e teste discordam, vence quem é melhor na validação cruzada."""
    rows = [
        {"model": "melhor_no_teste", "cv_f1_mean": 0.60, "f1": 0.70},
        {"model": "melhor_na_cv", "cv_f1_mean": 0.65, "f1": 0.60},
        {"model": "pior", "cv_f1_mean": 0.50, "f1": 0.50},
    ]

    ranked = rank_models(rows)

    assert list(ranked["model"]) == ["melhor_na_cv", "melhor_no_teste", "pior"]
    assert ranked.loc[0, "model"] == "melhor_na_cv"


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="modelo ainda nao treinado")
def test_modelo_salvo_e_o_campeao_da_tabela():
    """models/champion.joblib e docs/model_comparison.csv precisam concordar."""
    table = pd.read_csv(config.DOCS_DIR / "model_comparison.csv")
    champion = rank_models(table.to_dict(orient="records")).loc[0, "model"]

    saved = joblib.load(config.MODEL_PATH)
    expected = type(get_candidates()[champion].named_steps["model"])

    assert isinstance(saved.named_steps["model"], expected)
