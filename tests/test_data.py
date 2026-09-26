"""Testes do pré-processamento (Etapa 3)."""

import pandas as pd

from churn import config, data


def test_clean_remove_colunas_de_vazamento(clean_df: pd.DataFrame):
    """Nenhuma coluna de vazamento pode sobreviver à limpeza."""
    for col in config.LEAKAGE_COLS:
        assert col not in clean_df.columns


def test_clean_converte_total_charges_para_numerico(clean_df: pd.DataFrame):
    """total_charges vem como texto no Excel e precisa virar número sem nulos."""
    assert pd.api.types.is_numeric_dtype(clean_df["total_charges"])
    assert clean_df["total_charges"].isna().sum() == 0


def test_split_e_estratificado_e_reprodutivel(clean_df: pd.DataFrame):
    """O split preserva a proporção de churn e é estável entre execuções."""
    X_train, X_test, y_train, y_test = data.split(clean_df)

    assert len(X_train) + len(X_test) == len(clean_df)
    assert abs(y_train.mean() - y_test.mean()) < 0.01

    _, X_test_again, _, _ = data.split(clean_df)
    assert X_test.index.equals(X_test_again.index)
