"""Parâmetros de configuração compartilhados entre notebooks e módulos do pipeline."""

# Fração da base priorizada para a métrica de negócio Precision@K.
# Valor inicial documentado em docs/metrics.md; ajustável em experimentos
# futuros (Etapa 2) conforme a distribuição real dos scores dos modelos.
TOP_K_PERCENT = 0.10
