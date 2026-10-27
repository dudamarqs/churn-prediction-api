# Churn Prediction API

Tech Challenge da Fase 1 (Pós Tech FIAP). Pipeline preditivo de churn para
uma operadora de telecomunicações: da análise exploratória ao modelo servido
por uma API REST, com Scikit-Learn e FastAPI.

**Equipe:** Maria Eduarda Marques ([@dudamarqs](https://github.com/dudamarqs))
e Gabriel Silva Jorge ([@3ielzinh](https://github.com/3ielzinh))

**Vídeo de apresentação (método STAR, até 5 min):** [assistir no YouTube](https://youtu.be/PpSDr1HYWPE)

## Entregáveis

| Etapa | Entregável | Onde está |
|---|---|---|
| 1. Entendimento e preparação | ML Canvas e definição das métricas técnica e de negócio | [`docs/ml_canvas.md`](docs/ml_canvas.md), [`docs/metrics.md`](docs/metrics.md) |
| | Notebook de EDA + baseline (Regressão Logística) | [`notebooks/eda_baseline.ipynb`](notebooks/eda_baseline.ipynb) |
| 2. Modelagem e avaliação | Random Forest e MLP comparados ao baseline, com validação cruzada | [`notebooks/model_comparison.ipynb`](notebooks/model_comparison.ipynb) |
| | Tabela comparativa e registro dos experimentos | [`docs/model_comparison.csv`](docs/model_comparison.csv), [`docs/rf_grid_search.csv`](docs/rf_grid_search.csv) |
| | Modelo campeão salvo | [`models/champion.joblib`](models/champion.joblib) |
| 3. Engenharia e API | Código modular de produção | [`src/churn/`](src/churn/) |
| | API FastAPI com `/health` e `/predict` | [`src/churn/api/`](src/churn/api/) |
| | Testes automatizados com pytest | [`tests/`](tests/) |
| 4. Documentação e entrega | Model Card (performance, limitações e vieses) | [`docs/model_card.md`](docs/model_card.md) |
| | README com instalação e execução | este arquivo |
| | Vídeo STAR | link no topo |

## Problema

Uma operadora de telecomunicações está perdendo clientes, e o time de
retenção tem orçamento limitado. O modelo estima a probabilidade de churn de
cada cliente para ordenar a base por risco: os clientes do topo entram numa
lista revisada por uma pessoa do time de retenção antes de qualquer ação
(ver [ML Canvas](docs/ml_canvas.md)).

**Dataset:** IBM Telco Customer Churn (versão estendida), 7.043 clientes,
33 colunas, 26,5% de churn. Arquivo em `data/raw/Telco_customer_churn.xlsx`.

## Resultados

Campeão: **Random Forest**, escolhido pelo maior F1 médio na validação
cruzada (5 folds estratificados, só no treino). O conjunto de teste (20%,
estratificado) ficou reservado para a avaliação final.

| Modelo | F1 (validação cruzada) | F1 (teste) | ROC-AUC (teste) | Precision@10% (teste) |
|---|---|---|---|---|
| **Random Forest (campeão)** | **0,650** | **0,632** | 0,853 | 0,771 |
| Regressão Logística (baseline) | 0,640 | 0,617 | 0,849 | 0,764 |
| MLP | 0,615 | 0,559 | 0,844 | 0,743 |

- **Métrica técnica:** F1 da classe churn, que equilibra precisão e recall
  numa base desbalanceada (só a acurácia enganaria: prever que ninguém
  cancela já acerta 73,5%).
- **Métrica de negócio:** Precision@10%. Dos 10% de clientes com maior risco
  no teste, 77% cancelaram: quase três vezes a taxa média de 26,5%.
- O ganho sobre o baseline é real, porém modesto. Na Precision@K, os três
  modelos ficam praticamente empatados (notebook de comparação, seção 6).
- O que mais pesa no ranking: tipo de contrato e tempo de casa
  (permutation importance, notebook de comparação, seção 7).

## Principais decisões técnicas

- **Vazamento de dados:** removemos `Churn Label`, `Churn Score`,
  `Churn Reason` e `CLTV`, além de identificadores, colunas constantes e
  localização (cidade, CEP e coordenadas, por alta cardinalidade e risco de
  viés geográfico). A lista fica em [`src/churn/config.py`](src/churn/config.py).
- **Pipeline único:** imputação, padronização e one-hot ficam dentro de um
  `Pipeline` do Scikit-Learn, ajustado só com dados de treino em cada fold.
  O mesmo `Pipeline` é salvo em `models/champion.joblib` e usado pela API.
- **Escolha do campeão:** pela validação cruzada, nunca pelo teste.
- **Busca de hiperparâmetros:** 24 combinações para o Random Forest
  (`GridSearchCV`); a configuração usada foi a melhor.
- **Reprodutibilidade:** seeds fixas (`RANDOM_STATE = 42`) no split, na
  validação cruzada e nos modelos; treinar de novo reproduz a tabela acima.
- **Separação de responsabilidades:** notebooks para experimentação; o
  pacote `src/churn` para produção.

## Como executar

### 1. Instalar

Requer Python 3.12 ou superior.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

O modelo treinado já está no repositório: não é preciso treinar para rodar a
API. Para reproduzir exatamente o ambiente usado na validação (Windows,
Python 3.12), use `pip install -r requirements-lock.txt` e depois
`pip install --no-deps -e .`.

### 2. Rodar os testes

```bash
python -m pytest -q
```

São 21 testes: limpeza e split dos dados, métricas, escolha do campeão,
predição e API.

### 3. Subir a API

```bash
python -m uvicorn churn.api.main:app --reload
```

Documentação interativa (Swagger): http://127.0.0.1:8000/docs. Se a porta
8000 estiver bloqueada, como acontece em alguns Windows, acrescente
`--port 8765` e use essa porta nos endereços abaixo.

- `GET /health`: indica se a API está no ar e se o modelo foi carregado
  (`model_loaded`).
- `POST /predict`: recebe os dados de um cliente e devolve
  `churn_probability` e `churn_prediction` (classe no corte de 0,5).

Exemplo com o cliente de [`docs/sample_customer.json`](docs/sample_customer.json):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @docs/sample_customer.json
```

```json
{"churn_probability": 0.889, "churn_prediction": 1}
```

No PowerShell:

```powershell
$payload = Get-Content -Raw docs/sample_customer.json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $payload
```

Dados inválidos (categoria fora do domínio ou campo faltando) retornam
HTTP 422. Sem `models/champion.joblib`, o `/health` responde
`model_loaded: false` e o `/predict` retorna HTTP 503.

A probabilidade é uma estimativa de risco dado o perfil atual, não uma
previsão de quando o cliente vai cancelar. A decisão de negócio usa o
ranking dos clientes de maior risco; a classe no corte de 0,5 não substitui
essa priorização.

### 4. Treinar de novo (opcional)

```bash
python -m churn.train
```

Treina os três modelos, salva o campeão em `models/champion.joblib` e a
tabela comparativa em `docs/model_comparison.csv`.

## Estrutura do projeto

```
├── data/raw/                   # dataset original
├── notebooks/
│   ├── eda_baseline.ipynb      # Etapa 1: EDA + baseline
│   └── model_comparison.ipynb  # Etapa 2: comparação, hiperparâmetros, importância
├── src/churn/                  # código de produção (pacote instalável)
│   ├── config.py               # caminhos, seeds, colunas e parâmetros
│   ├── data.py                 # carga, limpeza e split
│   ├── features.py             # pré-processamento (ColumnTransformer)
│   ├── models.py               # candidatos (Logística, Random Forest, MLP)
│   ├── train.py                # treino, comparação e escolha do campeão
│   ├── evaluate.py             # métricas técnicas e Precision@K
│   ├── predict.py              # predição a partir do modelo salvo
│   └── api/                    # API FastAPI (main.py, schemas.py)
├── models/champion.joblib      # modelo campeão (Pipeline completo)
├── tests/                      # testes com pytest
├── docs/                       # ML Canvas, métricas, Model Card, resultados e validação
├── pyproject.toml              # dependências e configuração
└── requirements-lock.txt       # ambiente exato da validação
```

## Documentação

- [ML Canvas](docs/ml_canvas.md): stakeholders, proposta de valor, decisões, dados e métricas.
- [Métricas](docs/metrics.md): por que F1 e Precision@K.
- [Model Card](docs/model_card.md): performance, limitações e possíveis vieses.
- [Validação local](docs/validation.md): registro da execução dos testes e da API.

## Limitações

- O dataset é um retrato estático: o modelo estima quem está em risco, não
  quando vai cancelar.
- O ganho sobre o baseline é modesto, e o MLP foi o único candidato
  treinado sem balanceamento de classes.
- Não houve análise de desempenho por subgrupo; o uso previsto inclui
  revisão humana antes de qualquer ação.

Detalhes no [Model Card](docs/model_card.md).
