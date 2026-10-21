"""API de inferência do modelo campeão (Etapa 3).

Executar da raiz do projeto:

    uvicorn churn.api.main:app --reload
"""

from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from churn import config
from churn.api.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from churn.predict import classify, predict_proba

app = FastAPI(
    title="Churn Prediction API",
    description="Prediz a propensão de churn de clientes de telecom.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def _load_model(path: Path):
    """Mantém em cache somente o artefato carregado com sucesso."""
    return joblib.load(path)


def get_model():
    """Permite encontrar um modelo criado após a primeira chamada à API.

    Para substituir um artefato já carregado, reinicie o processo da API.
    """
    if not config.MODEL_PATH.exists():
        return None
    return _load_model(config.MODEL_PATH)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Verifica se a API está no ar e se o modelo foi carregado."""
    return HealthResponse(status="ok", model_loaded=get_model() is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Retorna a probabilidade de churn para um cliente."""
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo nao encontrado. Execute 'python -m churn.train' primeiro.",
        )

    proba = float(predict_proba(model, customer.model_dump())[0])

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=classify(proba),
    )
