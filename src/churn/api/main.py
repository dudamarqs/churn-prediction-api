"""API de inferência do modelo campeão (Etapa 3).

Executar da raiz do projeto:

    uvicorn churn.api.main:app --reload
"""

from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from churn import config
from churn.api.schemas import CustomerFeatures, HealthResponse, PredictionResponse

app = FastAPI(
    title="Churn Prediction API",
    description="Prediz a propensão de churn de clientes de telecom.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_model():
    """Carrega o Pipeline treinado uma única vez por processo."""
    if not config.MODEL_PATH.exists():
        return None
    return joblib.load(config.MODEL_PATH)


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

    X = pd.DataFrame([customer.model_dump()])[config.FEATURES]
    proba = float(model.predict_proba(X)[0, 1])

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=int(proba >= 0.5),
    )
