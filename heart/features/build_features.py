"""
Функция обработки категориальных признаков
"""
from typing import List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    """Функция обработки категориальных признаков"""
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())

def build_categorical_pipeline() -> Pipeline:
    """Дописанная функция"""
    categorical_pipeline = Pipeline(
        [
            #Added SimpleImputer for Categorical Preprocessing
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline

def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    """Функция обработки numerical признаков"""
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    """Доделанная функция"""
    num_pipeline = Pipeline(
        [
            ("OutlierRemover", OutlierRemover()),
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent"))
        ]
    )
    return num_pipeline


def make_features(transformer: ColumnTransformer, dfx: pd.DataFrame) -> pd.DataFrame:
    """Функция make_feature"""
    return pd.DataFrame(transformer.transform(dfx))


def extract_target(dfx: pd.DataFrame, target_col: List[str]) -> pd.Series:
    """Функция extract_target"""
    target = dfx[target_col].values
    return target


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Класс OutlierRemover"""
    def __init__(self, factor=1.5):
        self.factor = factor

    def outlier_removal(self, xdf: pd.DataFrame):
        """Функция OutlierRemover"""
        xdf = pd.Series(xdf).copy()
        q_1 = xdf.quantile(0.25)
        q_3 = xdf.quantile(0.75)
        iqr = q_3 - q_1
        lower_bound = q_1 - (self.factor * iqr)
        upper_bound = q_3 + (self.factor * iqr)
        xdf.loc[((xdf < lower_bound) | (xdf > upper_bound))] = np.nan
        return pd.Series(xdf)

    def fit(self, abc):
        """Функция fit"""
        return self

    def transform(self, xello) -> pd.DataFrame:
        """Функция transform"""
        return pd.DataFrame(xello).apply(self.outlier_removal)


def build_transformer(
    categorical_features: List[str], numerical_features: List[str]
    ) -> ColumnTransformer:
    """Функция build_transformer"""
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                #[c for c in categorial features],
                list(categorical_features)
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                list(numerical_features),
            )
        ]
    )
    return transformer
