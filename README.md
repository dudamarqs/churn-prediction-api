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
- [x] Modelagem comparativa (Random Forest + MLP) — Etapa 2 — [`notebooks/model_comparison.ipynb`](notebooks/model_comparison.ipynb) + [`docs/model_comparison.csv`](docs/model_comparison.csv)
- [x] API FastAPI + testes — Etapa 3
- [x] Model Card — performance, limitações, vieses e reprodução local
- [ ] Vídeo STAR de até 5 minutos — gravar, publicar e informar o link na entrega

## Resultado atual

Modelo campeão: **Random Forest** (`models/champion.joblib`), selecionado
pelo F1-score médio na validação cruzada (classe churn) entre os três
candidatos (Regressão Logística, Random Forest, MLP) — ver
[`docs/model_comparison.csv`](docs/model_comparison.csv) para a tabela
comparativa completa, [`docs/metrics.md`](docs/metrics.md) para a
justificativa das métricas e o [Model Card](docs/model_card.md) para
performance, limitações e vieses.

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
│   ├── evaluate.py              # métricas técnica e de negócio
│   └── predict.py               # carga do modelo e predição (usado pela API)
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

No PowerShell, a ativação pode ser dispensada: use `.venv\Scripts\python.exe`
no lugar de `python` e `.venv\Scripts\python.exe -m pip` no lugar de `pip`.
O modelo já está incluído no repositório; não é necessário treinar novamente
para demonstrar a API. Para preservar o ambiente validado, use também o
arquivo `requirements-lock.txt` (ambiente Windows, Python 3.12).
Instale com `python -m pip install -r requirements-lock.txt` e depois
`python -m pip install --no-deps -e .`. Em outro sistema, use `.[dev]`.

## Treinar o modelo

Lê `data/raw/Telco_customer_churn.xlsx`, treina os três candidatos com
validação cruzada, salva o campeão em `models/champion.joblib` e a tabela
comparativa em `docs/model_comparison.csv`:

```bash
python -m churn.train
```

## Rodar a API

```bash
python -m uvicorn churn.api.main:app --reload
```

Com a API no ar (documentação interativa em `http://127.0.0.1:8000/docs`):

- `GET /health` — verifica se a API está no ar e se o modelo foi carregado.
- `POST /predict` — recebe os dados de um cliente e retorna a probabilidade
  de churn. Exemplo de payload em `tests/conftest.py` (`sample_customer`).

Se a porta 8000 estiver bloqueada pelo Windows, execute
`python -m uvicorn churn.api.main:app --host 127.0.0.1 --port 8765`
e abra `http://127.0.0.1:8765/docs`. Use essa porta também nas chamadas.

## Rodar os testes

```bash
python -m pytest -q
```

## Autoria

Projeto desenvolvido em dupla para o Tech Challenge da Fase 1 (Pós Tech /
FIAP).

## Demonstração da API

Abra `http://127.0.0.1:8000/docs`, execute `GET /health` e confira
`model_loaded: true`. Em `POST /predict`, cole o conteúdo de
[`docs/sample_customer.json`](docs/sample_customer.json) e clique em Execute.
A resposta contém `churn_probability` e `churn_prediction` (corte de 0,5).

Alternativa pelo PowerShell, em outro terminal na raiz do projeto:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
$payload = Get-Content -Raw docs/sample_customer.json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $payload
```

Categoria inválida ou campo obrigatório ausente retorna HTTP 422.
Sem `models/champion.joblib`, `/health` responde com `model_loaded: false`
e `/predict` retorna HTTP 503. Depois de criar o artefato, uma nova chamada
pode carregá-lo; se substituir um modelo já carregado, reinicie a API.

As probabilidades são estimativas do modelo, não previsões de uma data de
cancelamento. A decisão de negócio usa o ranking top-K com revisão humana;
a classe no corte de 0,5 não substitui essa priorização.

## Entrega final

- Conferir os resultados de execução em [`docs/validation.md`](docs/validation.md): 9 testes passaram, API demonstrada no navegador e métricas do campeão conferidas.
- Gravar o vídeo com o roteiro `docs/roteiro-video-STAR.txt` (até 5 minutos).
- Registrar o link do vídeo após a gravação; ele permanece pendente.
- Garantir acesso do avaliador ao repositório e ao vídeo, conforme a plataforma.
- Submeter os links do repositório e do vídeo. Deploy em nuvem é opcional.
