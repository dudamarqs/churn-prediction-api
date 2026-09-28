# Churn Prediction API

Tech Challenge — Fase 1 (FIAP). Pipeline preditivo de churn de clientes de
uma operadora de telecomunicações, com modelagem clássica (Scikit-Learn) e
disponibilização via API REST (FastAPI).

## Contexto

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado.
Este projeto constrói, do zero, um modelo que classifica clientes com risco
de cancelamento (churn), para apoiar ações de retenção direcionadas dentro
de um orçamento limitado (ver [ML Canvas](docs/ml_canvas.md)).

## Status

- [x] ML Canvas — [`docs/ml_canvas.md`](docs/ml_canvas.md)
- [x] Métricas técnica e de negócio — [`docs/metrics.md`](docs/metrics.md)
- [x] EDA + Baseline (Regressão Logística) — [`notebooks/eda_baseline.ipynb`](notebooks/eda_baseline.ipynb)
- [x] Modelagem comparativa (Random Forest + MLP) — Etapa 2 — [`docs/model_comparison.csv`](docs/model_comparison.csv)
- [x] API FastAPI + testes — Etapa 3
- [ ] Model Card + vídeo STAR — Etapa 4

## Resultado atual

Modelo campeão: **Random Forest** (`models/champion.joblib`), selecionado
por F1-score na classe churn entre os três candidatos (Regressão Logística,
Random Forest, MLP) — ver [`docs/model_comparison.csv`](docs/model_comparison.csv)
para a tabela comparativa completa e [`docs/metrics.md`](docs/metrics.md)
para a justificativa das métricas.

## Estrutura do projeto

```
├── data/
│   ├── raw/                  # dados originais (não modificar)
│   └── processed/            # dados tratados (gerados, não versionados)
├── notebooks/                # experimentação (EDA, baseline)
├── src/churn/                # código produtivo (pacote instalável)
│   ├── api/                  # API FastAPI (/health, /predict)
│   ├── config.py             # caminhos, seeds, colunas, parâmetros
│   ├── data.py                # carga e limpeza do dataset
│   ├── features.py            # pré-processador (ColumnTransformer)
│   ├── models.py               # candidatos (Logística, RF, MLP)
│   ├── train.py                 # treino, comparação e persistência do campeão
│   └── evaluate.py              # métricas técnica e de negócio
├── models/                   # modelo campeão serializado (.joblib)
├── tests/                    # testes automatizados (pytest)
├── docs/                     # ML Canvas, métricas, comparação de modelos, Model Card
├── pyproject.toml
└── README.md
```

## Setup

Requer Python 3.12+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/Mac: source .venv/bin/activate
pip install -e ".[dev]"
```

## Treinar o modelo

Lê `data/raw/Telco_customer_churn.xlsx`, treina os três candidatos com
validação cruzada, salva o campeão em `models/champion.joblib` e a tabela
comparativa em `docs/model_comparison.csv`:

```bash
python -m churn.train
```

## Rodar a API

```bash
uvicorn churn.api.main:app --reload
```

Com a API no ar (documentação interativa em `http://127.0.0.1:8000/docs`):

- `GET /health` — verifica se a API está no ar e se o modelo foi carregado.
- `POST /predict` — recebe os dados de um cliente e retorna a probabilidade
  de churn. Exemplo de payload em `tests/conftest.py` (`sample_customer`).

## Rodar os testes

```bash
pytest
```

## Autoria

Projeto desenvolvido em dupla para o Tech Challenge da Fase 1 (Pós Tech /
FIAP).
