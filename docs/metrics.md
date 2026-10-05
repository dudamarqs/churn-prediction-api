# Métrica Técnica e Métrica de Negócio (Etapa 1)

Definição dos critérios usados para avaliar e comparar os modelos na
Etapa 2 (Regressão Logística, Random Forest/Ensemble e MLP).

## Métrica Técnica

**F1-Score da classe churn** é a métrica técnica primária de seleção do
modelo campeão — equilibra precisão e recall sob o desbalanceamento de
classes (73,5% não-churn / 26,5% churn) do dataset, evitando o viés da
acurácia isolada.

Complementada por **AUC-ROC** e **AUC-PR** como métricas secundárias de
avaliação da capacidade de ranqueamento do modelo (AUC-PR é mais
informativa que AUC-ROC sob desbalanceamento). Essas métricas independem
de um threshold fixo, o que importa porque a decisão de negócio (ver ML
Canvas, bloco 3) depende de um ranking de risco, não de uma classificação
binária no corte padrão de 0,5.

## Métrica de Negócio

**Precision@K**: entre os top-K clientes com maior probabilidade de churn
prevista, qual fração realmente apresentaria churn. Mede diretamente a
eficiência do orçamento de retenção (ver ML Canvas, blocos 1 e 9) — evita
desperdiçar ações de retenção em falsos positivos.

O case não define a capacidade real de atendimento do time de retenção,
então **K é parametrizado como uma fração da base** em vez de um número
fixo de clientes. Valor inicial documentado: `TOP_K_PERCENT = 0.10` (top
10%), definido em `src/churn/config.py`. É uma suposição inicial, não um
valor travado — a Etapa 2 pode (e deve) testar outros valores de K conforme
a distribuição real dos scores dos modelos treinados.

Na Etapa 2, a curva de Precision@K para K de 5% a 50%
([`notebooks/model_comparison.ipynb`](../notebooks/model_comparison.ipynb),
seção 6) mostrou os três modelos praticamente empatados (diferenças de no
máximo 0,03): o campeão lidera ou empata em 8 dos 10 valores de K,
incluindo K=10%. Por isso a escolha do campeão é feita pela métrica
técnica (F1 na validação cruzada), e K=10% segue como suposição de
trabalho.

## Ver também

- [ML Canvas](ml_canvas.md) — blocos 1 (Proposta de Valor), 3 (Decisões),
  8 (Avaliação Offline) e 9 (Impacto), de onde essas métricas derivam.
