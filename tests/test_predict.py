"""Testes do módulo de predição (Etapa 3)."""

import numpy as np
import pandas as pd
import pytest

from churn import config, data, predict
from churn.models import BASELINE_NAME, get_candidates


@pytest.fixture(scope="module")
def fitted_model(clean_df: pd.DataFrame):
    """Pipeline pequeno treinado na hora, sem depender de models/champion.joblib."""
    X_train, _, y_train, _ = data.split(clean_df)
    return get_candidates()[BASELINE_NAME].fit(X_train, y_train)


def test_load_model_devolve_none_sem_artefato(tmp_path):
    """Sem modelo treinado, a carga não quebra: a API usa isso para responder 503."""
    assert predict.load_model(tmp_path / "nao_existe.joblib") is None


def test_predict_proba_aceita_um_cliente_ou_varios(fitted_model, sample_customer: dict):
    """Um dict vira uma linha; uma lista vira várias, todas com probabilidade em [0, 1]."""
    um = predict.predict_proba(fitted_model, sample_customer)
    varios = predict.predict_proba(fitted_model, [sample_customer, sample_customer])

    assert um.shape == (1,)
    assert varios.shape == (2,)
    assert np.all((varios >= 0) & (varios <= 1))
    assert varios[0] == pytest.approx(um[0])


def test_predict_proba_ignora_ordem_das_chaves(fitted_model, sample_customer: dict):
    """A ordem dos campos no JSON de entrada não pode mudar a predição."""
    invertido = dict(reversed(list(sample_customer.items())))

    assert predict.predict_proba(fitted_model, invertido)[0] == pytest.approx(
        predict.predict_proba(fitted_model, sample_customer)[0]
    )


def test_predict_proba_igual_ao_pipeline(fitted_model, clean_df: pd.DataFrame):
    """Mesma saída que chamar o Pipeline direto sobre o DataFrame de features."""
    X = clean_df[config.FEATURES].head(20)

    np.testing.assert_allclose(
        predict.predict_proba(fitted_model, X.to_dict(orient="records")),
        fitted_model.predict_proba(X)[:, 1],
    )


def test_classify_respeita_o_corte():
    """Probabilidade igual ao corte conta como churn; abaixo, não."""
    assert predict.classify(config.DECISION_THRESHOLD) == 1
    assert predict.classify(config.DECISION_THRESHOLD - 1e-6) == 0
    assert predict.classify(0.3, threshold=0.25) == 1
