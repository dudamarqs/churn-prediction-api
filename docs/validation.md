# Validação local da entrega

Data: 13/10/2026. Python 3.12.14 em Windows.

## Resultados

- `python -m pytest -q`: **9 passed**, sem testes ignorados (4,84 s).
- Um aviso de depreciação do cliente httpx/Starlette; não causou falha.
- `/health`: HTTP 200, `status: ok`, `model_loaded: true`.
- `/predict`: HTTP 200, probabilidade **0,889**, classe **1**, usando `docs/sample_customer.json`.
- Categoria inválida: HTTP 422. Ausência do modelo: HTTP 503, coberta por teste.
- Swagger aberto em `http://127.0.0.1:8765/docs`; chamada real no navegador retornou HTTP 200 e os mesmos valores.
- A porta 8000 foi bloqueada pelo Windows nesta máquina (WinError 10013); a porta 8765 funcionou.

## Modelo preservado

Não houve novo treinamento ou alteração do dataset. O artefato original foi carregado com scikit-learn 1.9.1, versão identificada no próprio arquivo e fixada no pyproject.toml.
As métricas recalculadas no mesmo split de teste coincidem com `docs/model_comparison.csv`:

| Métrica | Resultado |
|---|---:|
| Precision | 0,5353159851 |
| Recall | 0,7700534759 |
| F1 | 0,6315789474 |
| ROC-AUC | 0,8525536697 |
| Average Precision (coluna pr_auc) | 0,6626097971 |
| Precision@10% | 0,7714285714 |

## Reproduzir

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m uvicorn churn.api.main:app --host 127.0.0.1 --port 8765
```

Instale Python 3.12 se `py -3.12` não estiver disponível. O lock inclui dependências dos notebooks e uma dependência Windows (pywinpty); é o snapshot do ambiente de validação em Windows. Em outro sistema, instale `.[dev]` e registre o ambiente usado.

## Limites e pendências

- Notebooks não foram executados novamente; as métricas e saídas originais foram preservadas.
- A comparação completa dos três modelos e a CV não foram repetidas; foi conferido o campeão salvo contra o split documentado.
- Análise quantitativa por subgrupo, calibração e validação externa ficam para trabalho futuro.
- Campos da API validam categorias e limites individuais, mas não todas as relações entre serviços.
- Vídeo STAR ainda precisa ser gravado/publicado, e os links submetidos na plataforma.
- Esta cópia foi obtida por ZIP; o histórico de commits permanece no GitHub original. As alterações locais não foram publicadas.

## Atualização — 14/10/2026 (PR #6)

Validação repetida depois de integrar o PR #6 (módulo `churn.predict` e
novos testes), com Python 3.14.7 e scikit-learn 1.9.1 em Windows:

- `python -m pytest -q`: **21 passed**.
- Respostas da API idênticas às da validação anterior para 300 clientes do
  conjunto de teste, para categoria inválida (HTTP 422) e sem modelo
  (`/health` com `model_loaded: false`, `/predict` com HTTP 503).
- `/predict` com `docs/sample_customer.json`: probabilidade **0,889**, classe **1**.
- Os dois notebooks rodaram do início ao fim sem erros.
- `python -m churn.train` reproduziu exatamente `docs/model_comparison.csv`
  (validação cruzada e comparação dos três modelos refeitas).

## Ambiente

- scikit-learn: 1.9.1
- pandas: 2.3.3
- numpy: 2.5.4
- fastapi: 0.143.0
- pydantic: 2.14.1
- pytest: 9.1.1
- httpx: 0.28.1
- joblib: 1.6.0
- uvicorn: 0.54.0
