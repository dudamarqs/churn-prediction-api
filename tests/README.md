# tests

Testes automatizados com pytest. Mínimo exigido (Etapa 3): teste de uma
função de pré-processamento e teste de status da API (`/health`).

- `test_data.py`: limpeza do dataset e split estratificado.
- `test_api.py`: endpoints `/health` e `/predict`.
- `test_predict.py`: carga do modelo e predição (`churn.predict`).
- `test_evaluate.py`: métricas, incluindo a Precision@K (métrica de negócio).
- `test_train.py`: escolha do campeão pela validação cruzada e coerência
  entre `models/champion.joblib` e `docs/model_comparison.csv`.
