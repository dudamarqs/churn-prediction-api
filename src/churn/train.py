"""Treino, comparação dos modelos e persistência do campeão (Etapa 2).

Critério de escolha do campeão: maior F1 médio na validação cruzada
estratificada (ver docs/metrics.md).

Executar da raiz do projeto:

    python -m churn.train
"""

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from churn import config, data
from churn.evaluate import evaluate
from churn.models import get_candidates


def cross_validate_f1(pipeline, X, y) -> tuple[float, float]:
    """F1 médio e desvio-padrão em validação cruzada estratificada."""
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def rank_models(rows: list[dict]) -> pd.DataFrame:
    """Ordena os candidatos do melhor para o pior; o primeiro é o campeão.

    O critério é o F1 médio da validação cruzada (só treino). O teste fica
    reservado para a estimativa final: usá-lo na escolha o transformaria em
    mais um conjunto de validação e deixaria as métricas reportadas
    otimistas.
    """
    return pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)


def run() -> pd.DataFrame:
    """Treina os candidatos, compara e salva o campeão em models/."""
    df = data.load_clean()
    X_train, X_test, y_train, y_test = data.split(df)

    rows = []
    fitted = {}

    for name, pipeline in get_candidates().items():
        cv_mean, cv_std = cross_validate_f1(pipeline, X_train, y_train)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        fitted[name] = pipeline
        rows.append({"model": name, "cv_f1_mean": cv_mean, "cv_f1_std": cv_std, **evaluate(y_test, y_pred, y_proba)})

    results = rank_models(rows)

    champion_name = results.loc[0, "model"]
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[champion_name], config.MODEL_PATH, compress=3)

    results.to_csv(config.DOCS_DIR / "model_comparison.csv", index=False)

    print(results.to_string(index=False))
    print(f"\nCampeao: {champion_name} -> {config.MODEL_PATH}")

    return results


if __name__ == "__main__":
    run()
