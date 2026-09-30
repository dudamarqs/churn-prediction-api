# Model Card — Churn Prediction

> Documento vivo, atualizado em etapas junto com o desenvolvimento do
> projeto (ver seção "Status" no fim). Esta versão fecha os fatos da
> Etapa 1 e a performance/campeão da Etapa 2. API e reprodutibilidade
> (Etapa 3) e a revisão final de limitações/vieses (Etapa 4) entram nas
> próximas atualizações.

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
entre três candidatos — Regressão Logística, Random Forest, MLP — por
F1-score na classe churn (métrica técnica primária definida na Etapa 1,
ver [`docs/metrics.md`](metrics.md)). Artefato salvo em
`models/champion.joblib`; código de treino em `src/churn/train.py`.

Hiperparâmetros: `n_estimators=300`, `min_samples_leaf=5`,
`class_weight="balanced"`, seed fixa. Sem busca de hiperparâmetros
(GridSearch/RandomSearch) — valores fixos, não otimizados.

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
pequena mas consistente sobre o baseline), e mantém a maior Precisão@K —
a métrica de negócio — ao longo de toda a faixa razoável de K, não só no
K=10% assumido inicialmente (ver notebook, seção 5). O ganho sobre o
baseline de Regressão Logística é real, porém modesto.

**MLP não escolhido apesar de maior precisão isolada:** seu recall baixo
(0,473) é explicado em parte por uma limitação de biblioteca (ver abaixo),
não só por capacidade de modelagem.

## Limitações conhecidas

- **`MLPClassifier` sem `class_weight`:** o Scikit-Learn não expõe esse
  parâmetro para MLP — é o único dos três candidatos sem correção
  explícita para o desbalanceamento de classes (73,5%/26,5%), o que
  penaliza seu recall.
- **Sem busca de hiperparâmetros:** os três candidatos usam configurações
  fixas e razoáveis, não otimizadas por validação cruzada.
- **Threshold fixo (0,5) vs. decisão por ranking:** precisão/recall/F1
  reportados usam corte 0,5, mas a decisão de negócio real é por top-K
  (ML Canvas, bloco 3). Os dois são coerentes no cenário atual (o
  Random Forest lidera em ambos), mas essa coerência não é garantida se a
  distribuição de scores mudar com novos dados.
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
- Nenhuma análise de desempenho por subgrupo (ex.: por faixa etária via
  `senior_citizen`, por `gender`) foi feita até aqui — fica como item para
  a revisão final da Etapa 4.

## Reprodutibilidade

Seeds fixas em `churn.config.RANDOM_STATE` (treino, split e CV). Detalhes
de ambiente/execução (API, testes) entram nesta seção após a Etapa 3.

## Status

- **Etapa 1 (concluído):** dados, features, exclusões e uso pretendido.
- **Etapa 2 (concluído):** performance, comparação e justificativa do campeão.
- **Etapa 3 (pendente):** reprodutibilidade de ambiente e comportamento via API.
- **Etapa 4 (pendente):** revisão final de limitações e vieses, consolidação.
