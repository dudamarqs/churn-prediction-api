# notebooks

Notebooks de experimentação (EDA, baselines, comparação de modelos).
Código produtivo (usado pela API) deve ser extraído para `src/`, não vivido
aqui — ver "Boas Práticas Obrigatórias" no enunciado do Tech Challenge.

- `eda_baseline.ipynb` (Etapa 1): volume/qualidade dos dados, distribuição
  do alvo e das features, e baseline de Regressão Logística — usa os
  mesmos módulos de `src/churn/` consumidos pelo treino de produção.
- `model_comparison.ipynb` (Etapa 2): comparação dos três candidatos,
  validação cruzada, busca de hiperparâmetros do Random Forest,
  Precisão@K, importância das features e justificativa do campeão.
