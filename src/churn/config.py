"""Configuração compartilhada entre notebooks, pipeline de treino e API.

Fonte única de verdade para caminhos, seeds, nomes de colunas e parâmetros
de avaliação. Qualquer módulo ou notebook deve importar daqui em vez de
redefinir esses valores localmente.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
# config.py mora em <raiz>/src/churn/, logo a raiz do projeto está 3 níveis
# acima. Derivar daqui torna os caminhos independentes do diretório de onde
# o código é executado (notebook, pytest, uvicorn ou terminal).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.xlsx"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Artefato do modelo campeão consumido pela API.
MODEL_PATH = MODELS_DIR / "champion.joblib"

# --------------------------------------------------------------------------
# Reprodutibilidade
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --------------------------------------------------------------------------
# Alvo
# --------------------------------------------------------------------------
TARGET = "churn_value"

# --------------------------------------------------------------------------
# Colunas descartadas (ver docs/ml_canvas.md, bloco 7)
# --------------------------------------------------------------------------
# Vazamento de alvo: preenchidas ou calculadas a partir do desfecho.
LEAKAGE_COLS = [
    "Churn Label",    # duplicata textual do alvo
    "Churn Score",    # score de propensão já pronto (IBM)
    "Churn Reason",   # só existe para quem cancelou
    "CLTV",           # composição temporal não documentada
]

# Sem sinal preditivo: identificador ou valor constante na base inteira.
CONSTANT_OR_ID_COLS = [
    "CustomerID",
    "Count",      # constante = 1
    "Country",    # constante = United States
    "State",      # constante = California
]

# Geográficas: alta cardinalidade, risco de overfitting e viés territorial.
GEO_COLS = [
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
]

DROP_COLS = LEAKAGE_COLS + CONSTANT_OR_ID_COLS + GEO_COLS

# --------------------------------------------------------------------------
# Renomeação para snake_case
# --------------------------------------------------------------------------
# Contrato de nomes entre o pré-processamento e o schema da API. Aplicado
# logo após a leitura do arquivo bruto: a partir daí todo o código só
# enxerga snake_case.
RENAME_MAP = {
    "Gender": "gender",
    "Senior Citizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "Tenure Months": "tenure_months",
    "Phone Service": "phone_service",
    "Multiple Lines": "multiple_lines",
    "Internet Service": "internet_service",
    "Online Security": "online_security",
    "Online Backup": "online_backup",
    "Device Protection": "device_protection",
    "Tech Support": "tech_support",
    "Streaming TV": "streaming_tv",
    "Streaming Movies": "streaming_movies",
    "Contract": "contract",
    "Paperless Billing": "paperless_billing",
    "Payment Method": "payment_method",
    "Monthly Charges": "monthly_charges",
    "Total Charges": "total_charges",
    "Churn Value": "churn_value",
}

# --------------------------------------------------------------------------
# Features (nomes pós-rename)
# --------------------------------------------------------------------------
NUM_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
]

CAT_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

FEATURES = NUM_FEATURES + CAT_FEATURES

# --------------------------------------------------------------------------
# Avaliação
# --------------------------------------------------------------------------
# Métrica de negócio Precision@K (ver docs/metrics.md): fração da base
# priorizada pelo time de retenção. Suposição inicial, revisável na Etapa 2.
TOP_K_PERCENT = 0.10

# Corte de probabilidade da classe prevista devolvida pela API. A decisão
# de negócio é por ranking (Precision@K), não por este corte — ver
# docs/model_card.md.
DECISION_THRESHOLD = 0.5
