"""Carga e limpeza do dataset bruto de churn.

Responsabilidade única: transformar o arquivo Excel da IBM em um DataFrame
limpo, com nomes em snake_case e sem colunas de vazamento. Não faz
codificação nem escalonamento — isso é responsabilidade de features.py.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Lê o arquivo Excel bruto, sem nenhuma transformação."""
    return pd.read_excel(path or config.DATA_RAW)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a limpeza mínima acordada na Etapa 1.

    1. Remove colunas de vazamento, identificadores e geográficas.
    2. Renomeia as colunas restantes para snake_case.
    3. Converte `total_charges` para numérico.

    `total_charges` vem como texto porque 11 clientes com `tenure_months = 0`
    têm o campo em branco — são contratos recém-abertos, ainda sem
    faturamento. O valor correto para eles é 0.0, não a mediana da base.
    """
    df = df.drop(columns=config.DROP_COLS, errors="ignore")
    df = df.rename(columns=config.RENAME_MAP)

    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"] = df["total_charges"].fillna(0.0)

    return df


def load_clean(path: Path | str | None = None) -> pd.DataFrame:
    """Atalho para `clean(load_raw(...))`."""
    return clean(load_raw(path))


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino e teste de forma estratificada e reprodutível.

    A estratificação preserva a proporção de churn (~26,5%) nos dois
    conjuntos. A seed vem de config para que baseline (Etapa 1) e modelos
    comparados (Etapa 2) sejam avaliados exatamente sobre o mesmo split.
    """
    X = df[config.FEATURES]
    y = df[config.TARGET]

    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
