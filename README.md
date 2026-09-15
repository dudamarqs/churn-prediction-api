# Churn Prediction API

Tech Challenge — Fase 1 (FIAP). Pipeline preditivo de churn de clientes de
uma operadora de telecomunicações, com modelagem clássica (Scikit-Learn) e
disponibilização via API REST (FastAPI).

## Contexto

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado.
Este projeto constrói, do zero, um modelo que classifica clientes com risco
de cancelamento (churn), para apoiar ações de retenção direcionadas.

## Status

🚧 Em desenvolvimento — Etapa 1 (Entendimento e Preparação).

- [ ] ML Canvas
- [ ] EDA
- [ ] Baseline (Regressão Logística)
- [ ] Modelagem comparativa (Árvore/Ensemble + MLP) — Etapa 2
- [ ] API FastAPI + testes — Etapa 3
- [ ] Model Card + vídeo STAR — Etapa 4

## Estrutura do projeto

```
├── data/
│   ├── raw/         # dados originais (não modificar)
│   └── processed/   # dados tratados (gerados, não versionados)
├── notebooks/        # experimentação (EDA, baselines)
├── src/               # código produtivo (pré-processamento, API)
├── models/            # modelos treinados serializados (.pkl/.joblib)
├── tests/             # testes automatizados (pytest)
├── docs/              # ML Canvas, Model Card, decisões técnicas
├── requirements.txt
└── README.md
```

## Setup

Instruções completas de instalação e execução serão adicionadas na Etapa 4,
quando a API estiver disponível. Por ora:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Autoria

Projeto desenvolvido em dupla para o Tech Challenge da Fase 1 (Pós Tech /
FIAP).
