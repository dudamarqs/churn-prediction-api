# ML Canvas — Previsão de Churn

Canvas de Machine Learning do projeto, construído bloco a bloco para
estruturar o problema de negócio antes da modelagem (Etapa 1: Entendimento
e Preparação).

## Resumo

| # | Bloco | Resumo |
|---|-------|--------|
| 1 | Proposta de Valor | Priorizar um orçamento de retenção limitado pelos clientes de maior risco, não agir sobre todos os "em risco". |
| 2 | Situação Atual | Não existe hoje processo estruturado de retenção baseado em dados (cenário greenfield). |
| 3 | Decisões | Threshold de probabilidade gera lista priorizada; revisão humana decide a ação antes do disparo. |
| 4 | Fazer Previsões | Execução periódica em batch, gerando ranking de risco para toda a base ativa. |
| 5 | Tarefa de Previsão | Classificação binária supervisionada do `Churn Value`, a partir do snapshot atual do cliente. |
| 6 | Fontes de Dados | `Telco_customer_churn.xlsx` (IBM, versão estendida/Watson Analytics), 7.043 clientes, 33 colunas. |
| 7 | Recursos | Variáveis demográficas, de contrato, serviços e financeiras; exclusões documentadas por vazamento/risco. |
| 8 | Avaliação Offline | Precision/Recall/F1/AUC-ROC/AUC-PR como métricas técnicas; Precision@K como métrica de negócio. |
| 9 | Impacto | Eficiência das ações de retenção, medida pela concentração de churners reais entre os priorizados. |

## Stakeholders

| Quem | Papel no projeto | O que precisa do modelo |
|---|---|---|
| Diretoria da operadora | Patrocinadora: pediu o modelo diante da perda acelerada de clientes | Reduzir o churn sem gastar o orçamento de retenção com quem não sairia |
| Time de retenção / CRM | Usuário da lista priorizada; revisa cada cliente e decide a ação (bloco 3) | Uma lista curta e confiável dos clientes de maior risco (Precision@K) |
| Clientes da operadora | Afetados pelas ações de retenção | Ser abordados com ofertas relevantes, nunca penalizados pelo score (ver [Model Card](model_card.md)) |
| Equipe de dados | Constrói, valida e mantém o modelo e a API | Métricas claras para comparar modelos e um pipeline reprodutível |

## 1. Proposta de Valor

O modelo existe para priorizar um orçamento de retenção limitado,
direcionando ações para os clientes com maior risco previsto de churn —
não para agir de forma indiscriminada sobre todo cliente classificado
como "em risco". Por isso, a precisão importa tanto quanto o recall:
dentro de uma restrição orçamentária, é fundamental que os clientes
selecionados para ação realmente estejam em risco de cancelamento
(framing de precisão no topo do ranking, Precision@K).

## 2. Situação Atual

Assume-se que a empresa não possui hoje um processo estruturado de
retenção baseado em dados — trata-se de um cenário greenfield, sem uma
linha de base ou processo anterior formal para comparação direta de
desempenho.

## 3. Decisões

Com base na probabilidade de churn prevista para cada cliente, uma regra
automática seleciona os clientes com score acima de um threshold de
probabilidade definido (calibrado conforme o orçamento/capacidade de
atendimento do time de retenção) e gera uma lista priorizada. Essa lista
passa por revisão humana do time de retenção/CRM antes do disparo da ação
(ex.: contato proativo, oferta de desconto/upgrade), que decide a ação
específica para cada cliente — o modelo não aciona a retenção de forma
totalmente automática.

## 4. Fazer Previsões

O modelo será executado periodicamente em batch, gerando a probabilidade
de churn para os clientes ativos. Os resultados serão utilizados para
criar um ranking de risco e direcionar as ações de retenção conforme a
capacidade disponível.

## 5. Tarefa de Previsão

Dado o perfil de um cliente (dados demográficos, de contrato, serviços
contratados e financeiros), prever a probabilidade de esse cliente
cancelar o serviço (churn), representada pela variável-alvo `Churn Value`
do dataset (binária: 0 = não, 1 = sim). É um problema de classificação
binária supervisionada, com a previsão feita no nível de cliente
individual, a partir do snapshot de dados mais recente disponível sobre
ele.

**Limitação:** o dataset é um snapshot estático, sem histórico temporal
de eventos — por isso a tarefa é "identificar clientes em risco de churn
dado o perfil atual", e não "prever churn dentro de uma janela de tempo
específica" (ex.: 30 dias).

## 6. Fontes de Dados

Dataset único, `Telco_customer_churn.xlsx` (IBM Telco Customer Churn,
versão estendida/Watson Analytics), com 7.043 clientes e 33 colunas —
dados demográficos, de contrato, serviços contratados e financeiros de
uma operadora de telecom. É o dataset sugerido pelo enunciado do case.
Não há integração com fontes externas ou em tempo real; é uma base
histórica estática, já fornecida, sem necessidade de coleta adicional.

## 7. Recursos

Variáveis demográficas (gênero, idade sênior, parceiro(a), dependentes),
de contrato (tipo de contrato, faturamento sem papel, forma de pagamento,
tempo de permanência), serviços contratados (telefone, múltiplas linhas,
internet, segurança online, backup online, proteção de dispositivo,
suporte técnico, streaming TV/filmes) e financeiras (cobrança mensal,
cobrança total).

Excluídas do conjunto de features: `Churn Score` e `Churn Reason`
(vazamento de dados), `Churn Label` (redundante com o target `Churn
Value`), `CustomerID`, `Count`, `Country`, `State` (sem sinal preditivo),
`CLTV` (não há, no case, documentação suficiente sobre sua composição
temporal — risco de vazamento não descartável) e `City`, `Zip Code`,
`Lat Long` (alta cardinalidade, risco de overfitting e viés geográfico).

## 8. Avaliação Offline

Devido ao desbalanceamento entre as classes, a acurácia isoladamente não
será utilizada como principal critério de avaliação. As métricas técnicas
serão Precision, Recall e F1-Score da classe churn, complementadas por
AUC-ROC e AUC-PR, permitindo avaliar tanto a identificação dos clientes
que podem cancelar quanto a capacidade de ranqueamento do modelo. Como o
orçamento de retenção é limitado e os clientes serão priorizados por
risco, a principal métrica de negócio será Precision@K, que mede a
proporção de clientes que realmente apresentariam churn entre aqueles
selecionados para ação.

## 9. Impacto

O modelo busca melhorar a eficiência das ações de retenção ao priorizar,
dentro do orçamento disponível, os clientes com maior risco previsto de
churn. Um falso positivo representa o gasto de uma ação de retenção em um
cliente que não cancelaria, enquanto um falso negativo representa a perda
de uma oportunidade de retenção. O impacto será avaliado pela capacidade
do modelo de concentrar clientes que realmente apresentam churn entre os
clientes priorizados, comparando seu desempenho com uma estratégia de
referência.
