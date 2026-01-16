import pandas as pd

from sklearn.utils import resample

from sklearn.model_selection import train_test_split


"""Функция для уменьшения размерности."""


def get_df_resampled(X):
    df_min = X[X['target'] == 1]
    df_maj = X[X['target'] == 0]
    df_downupsample = resample(df_maj, replace=True, n_samples=len(df_min), random_state=42)

    return pd.concat([df_min, df_downupsample], ignore_index=True).sample(frac=1.)


"""
Функция разделяет выборки на df_train и df_test, 
понижает размерность df_train, возвращает обе выборки.
"""


def get_train_test_datasets(df: pd.DataFrame, random_state: int) -> tuple:

    df_train, df_test = train_test_split(df, test_size=0.3, stratify=df['target'], random_state=random_state)

    df_train = get_df_resampled(df_train)

    return df_train, df_test


