# Model Card — Churn Prediction

> Consolidação da documentação para a entrega. As métricas de modelagem
> são as dos experimentos registrados; a validação local da API e do ambiente
> foi registrada separadamente em `docs/validation.md`.

## Visão geral e uso pretendido

Modelo de classificação binária que estima a probabilidade de um cliente
de telecom cancelar o serviço (`churn_value`), a partir do seu perfil
atual (demografia, contrato, serviços contratados e financeiro).

**Uso pretendido:** gerar um ranking de risco de churn sobre a base ativa,
usado para priorizar um orçamento de retenção limitado — os clientes de
maior risco entram numa lista revisada por um humano do time de
retenção/CRM antes de qualquer ação (ver [ML Canvas](ml_canvas.md), blocos
1 e 3). **Não é** uma automação de disparo de retenção, nem um preditor de
prazo ("vai cancelar em 30 dias") — o dataset é um snapshot estático sem
histórico temporal, então a saída é "risco dado o perfil atual" (ML
Canvas, bloco 5).

**Fora do escopo:** decisão automática sem revisão humana; previsão de
*quando* o cancelamento ocorreria; uso para negar serviço ou tarifar
clientes (o modelo foi desenhado para orientar retenção, não penalização).

## Dados de treino

`Telco_customer_churn.xlsx` (IBM Telco Customer Churn, versão
estendida/Watson Analytics) — 7.043 clientes, snapshot único, sem série
temporal. Classes desbalanceadas: 73,5% não-churn / 26,5% churn. Split
treino/teste 80/20, estratificado por `churn_value`, seed fixa
(`churn.config.RANDOM_STATE = 42`).

**Colunas excluídas** (decisão documentada no [ML Canvas](ml_canvas.md),
bloco 7, e centralizada em `churn.config.DROP_COLS`):

| Motivo | Colunas |
|---|---|
| Vazamento de alvo | `Churn Label`, `Churn Score`, `Churn Reason`, `CLTV` |
| Sem sinal preditivo | `CustomerID`, `Count`, `Country`, `State` |
| Geográficas (alta cardinalidade / viés territorial) | `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude` |

**Features usadas:** 3 numéricas (`tenure_months`, `monthly_charges`,
`total_charges`) + 16 categóricas (demografia, serviços contratados,
contrato, forma de pagamento). Lista completa em `churn.config.FEATURES`.

## Modelo campeão

**Random Forest** (`sklearn.ensemble.RandomForestClassifier`), selecionado
entre três candidatos — Regressão Logística, Random Forest, MLP — pelo
F1-score médio na validação cruzada, na classe churn (métrica técnica
primária definida na Etapa 1, ver [`docs/metrics.md`](metrics.md)). O
conjunto de teste não participa da escolha. Artefato salvo em
`models/champion.joblib`; código de treino em `src/churn/train.py`.

Hiperparâmetros: `n_estimators=300`, `min_samples_leaf=5`,
`class_weight="balanced"`, seed fixa. Validados por busca em grade
(`GridSearchCV`, 24 combinações de `max_depth`, `min_samples_leaf` e
`max_features`, 5-fold CV no treino): a configuração original foi a
melhor, e as 10 melhores ficam a menos de 0,005 de F1 entre si — o
desempenho é estável em relação aos hiperparâmetros. Registro em
[`docs/rf_grid_search.csv`](rf_grid_search.csv).

Pré-processamento (`src/churn/features.py`, igual para os três
candidatos): numéricas com imputação por mediana + padronização;
categóricas com imputação por moda + one-hot (`handle_unknown="ignore"`).

## Performance

Avaliação no conjunto de teste (20%, nunca visto em treino), mais 5-fold
CV estratificada no treino para checar robustez. Tabela completa em
[`docs/model_comparison.csv`](model_comparison.csv); análise visual e
justificativa detalhada em
[`notebooks/model_comparison.ipynb`](../notebooks/model_comparison.ipynb).

| modelo | cv_f1_mean | precision | recall | **f1** | roc_auc | pr_auc | precision@10% |
|---|---|---|---|---|---|---|---|
| **Random Forest (campeão)** | 0,650 | 0,535 | 0,770 | **0,632** | 0,853 | 0,663 | 0,771 |
| Regressão Logística (baseline) | 0,640 | 0,510 | 0,781 | 0,617 | 0,849 | 0,645 | 0,764 |
| MLP | 0,615 | 0,683 | 0,473 | 0,559 | 0,844 | 0,635 | 0,743 |

**Por que o Random Forest:** maior F1 tanto em CV quanto no teste (margem
pequena mas consistente sobre o baseline). Na métrica de negócio, os três
modelos ficam praticamente empatados: o Random Forest lidera ou empata em
8 de 10 valores de K entre 5% e 50%, incluindo o K=10% assumido (0,771),
com diferenças de no máximo 0,03 entre os modelos (ver notebook, seção 6).
O ganho sobre o baseline de Regressão Logística é real, porém modesto.

**O que mais pesa no ranking** (permutation importance no teste, queda no
AUC-PR; notebook, seção 7): `contract` e `tenure_months` lideram, seguidos
de `dependents`, `total_charges` e `internet_service`. `payment_method`
tem importância perto de zero porque seu sinal já está quase todo em
`contract` (78% de quem paga por *Electronic check* tem contrato mensal).

**MLP não escolhido apesar de maior precisão isolada:** seu recall baixo
(0,473) vem do treino sem balanceamento de classes somado ao corte de 0,5
(ver abaixo), não de falta de capacidade de modelagem — seu ROC-AUC
(0,844) fica perto dos demais.

## Limitações conhecidas

- **MLP treinado sem balanceamento de classes:** o `MLPClassifier` não
  tem `class_weight`, e o treino atual não usou a alternativa que ele
  aceita (`sample_weight`). É o único dos três candidatos sem correção
  para o desbalanceamento (73,5%/26,5%), então suas probabilidades ficam
  perto da taxa real de churn e o corte de 0,5 fica alto para ele: no
  teste, classifica 18% dos clientes como churn, contra 26,5% reais, e o
  recall cai para 0,473. A parada antecipada (`early_stopping=True`)
  agrava o efeito: ela escolhe a época pela acurácia, que favorece a
  classe majoritária, e ficou com o modelo da 3ª época. A capacidade de
  ranqueamento não é afetada da mesma forma (ROC-AUC 0,844). Correção
  possível, ainda pendente: treinar o MLP com `sample_weight` balanceado.
- **Busca de hiperparâmetros só no Random Forest:** Regressão Logística e
  MLP usam configurações fixas e razoáveis, não otimizadas. Como a busca
  manteve a configuração original do Random Forest, a comparação entre os
  três segue no mesmo nível de ajuste — mas um MLP ajustado poderia
  reduzir a distância para o campeão.
- **Threshold fixo (0,5) vs. decisão por ranking:** precisão/recall/F1
  reportados usam corte 0,5, mas a decisão de negócio real é por top-K
  (ML Canvas, bloco 3). O Random Forest lidera em F1, mas no ranking os
  três modelos ficam praticamente empatados — a vantagem em F1 não se
  traduz em ganho relevante de Precision@K.
- **Dataset estático, sem histórico temporal:** o modelo estima risco
  "dado o perfil atual", não uma janela de tempo até o cancelamento (ML
  Canvas, bloco 5) — não deve ser interpretado como previsão de prazo.
- **Generalização não testada fora deste dataset:** uma única operadora,
  um único período; comportamento sob distribuição de clientes diferente
  (outra região, outro mix de planos) é desconhecido.

## Possíveis vieses

- Exclusão deliberada de colunas geográficas (`City`, `Zip Code`,
  `Lat Long`) para evitar viés territorial direto — mas variáveis
  correlacionadas indiretamente com geografia/renda (ex.: tipo de
  internet, forma de pagamento) permanecem no modelo e podem carregar
  proxies desse viés. Não quantificado nesta etapa.
- `gender` tem importância aproximadamente nula na análise de
  permutação; esse resultado não demonstra que o modelo nunca use gênero.
  Isso não garante ausência de
  viés — uma feature pouco importante ainda pode ter desempenho desigual
  entre grupos.
- Nenhuma análise de desempenho por subgrupo (ex.: por faixa etária via
  `senior_citizen`, por `gender`) foi feita até aqui. É uma limitação
  documentada e um trabalho futuro, necessário antes de uso operacional.

## Reprodutibilidade

Seeds fixas em `churn.config.RANDOM_STATE` (treino, split e CV).
Instalação: ambiente virtual Python 3.12 e `python -m pip install -e ".[dev]"`.
O ambiente validado está registrado em `docs/validation.md` e
`requirements-lock.txt`. O arquivo serializado exige uma versão compatível
do scikit-learn; use a versão de treino registrada na validação.

Treino: `python -m churn.train`. API: `python -m uvicorn churn.api.main:app`.
Testes: `python -m pytest -q`. Exemplo completo: `docs/sample_customer.json`.
O teste de inferência usa o artefato real e falha caso ele não exista.

`GET /health` retorna o estado do serviço e `model_loaded`; não é garantia
de qualidade preditiva. `POST /predict` retorna a probabilidade arredondada
a quatro casas e a classe com corte de 0,5; entrada inválida retorna 422
e ausência do artefato retorna 503. O pipeline carregado fica em cache:
reinicie a API ao substituir um modelo já carregado.

Não houve novo treinamento nesta consolidação. O ganho observado sobre o
baseline não comprova significância estatística ou impacto de retenção.
O projeto é uma demonstração acadêmica; monitoramento de drift, calibração
das probabilidades e validação externa ficam para evolução posterior.

## Status

- **Etapa 1 (concluído):** dados, features, exclusões e uso pretendido.
- **Etapa 2 (concluído):** performance, comparação e justificativa do
  campeão, busca de hiperparâmetros e importância das features.
- **Etapa 3 (validado):** código modular, 21 testes aprovados e API demonstrada
  no Swagger. Evidências em `docs/validation.md`.
- **Etapa 4 (documentação consolidada):** performance, limitações, possíveis
  vieses e instruções de reprodução. Vídeo STAR e submissão ainda pendentes.
