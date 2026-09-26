"""Contratos de entrada e saída da API."""

from typing import Literal

from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]


class CustomerFeatures(BaseModel):
    """Dados brutos de um cliente, nos mesmos valores do dataset original.

    O cliente da API envia as categorias como texto ("Yes", "Fiber optic",
    ...); toda a codificação acontece dentro do Pipeline carregado, então
    quem consome a API não precisa conhecer o pré-processamento.
    """

    gender: Literal["Male", "Female"]
    senior_citizen: YesNo
    partner: YesNo
    dependents: YesNo
    tenure_months: int = Field(ge=0, le=100)
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Literal["Yes", "No", "No internet service"]
    online_backup: Literal["Yes", "No", "No internet service"]
    device_protection: Literal["Yes", "No", "No internet service"]
    tech_support: Literal["Yes", "No", "No internet service"]
    streaming_tv: Literal["Yes", "No", "No internet service"]
    streaming_movies: Literal["Yes", "No", "No internet service"]
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float = Field(description="Probabilidade prevista de churn (0 a 1).")
    churn_prediction: int = Field(description="Classe prevista no threshold de 0,5.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
