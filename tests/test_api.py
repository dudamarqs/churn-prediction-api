"""Testes da API de inferência (Etapa 3)."""

import joblib
from fastapi.testclient import TestClient

from churn.api.main import app, get_model
from churn import config
from churn.api import main

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


def test_predict_retorna_probabilidade_valida(sample_customer: dict):
    """Com o modelo treinado, /predict devolve uma probabilidade em [0, 1]."""
    assert get_model() is not None, "O artefato models/champion.joblib deve existir."
    response = client.post("/predict", json=sample_customer)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)


def test_predict_sem_modelo_retorna_503(monkeypatch, sample_customer: dict):
    monkeypatch.setattr(main, "get_model", lambda: None)
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 503
    assert "Modelo nao encontrado" in response.json()["detail"]


def test_health_sem_modelo(monkeypatch):
    monkeypatch.setattr(main, "get_model", lambda: None)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_modelo_criado_apos_primeira_consulta(monkeypatch, tmp_path):
    path = tmp_path / "champion.joblib"
    monkeypatch.setattr(config, "MODEL_PATH", path)
    main._load_model.cache_clear()
    try:
        assert get_model() is None
        joblib.dump({"artefato": "teste"}, path)
        assert get_model() == {"artefato": "teste"}
        assert get_model() is get_model()
    finally:
        main._load_model.cache_clear()
