"""Testes da API de inferência (Etapa 3)."""

import pytest
from fastapi.testclient import TestClient

from churn.api.main import app, get_model

client = TestClient(app)


def test_health_retorna_ok():
    """A API responde no /health independentemente de o modelo existir."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejeita_payload_invalido(sample_customer: dict):
    """Categoria fora do domínio deve ser barrada pelo schema (422)."""
    payload = {**sample_customer, "contract": "Contrato inexistente"}

    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.skipif(get_model() is None, reason="modelo ainda nao treinado")
def test_predict_retorna_probabilidade_valida(sample_customer: dict):
    """Com o modelo treinado, /predict devolve uma probabilidade em [0, 1]."""
    response = client.post("/predict", json=sample_customer)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
