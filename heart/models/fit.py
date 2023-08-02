"""
This is module fit
"""
import pickle
from typing import Union

import pandas as pd
from hydra.utils import instantiate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


SklearnRegressionModel = Union[RandomForestClassifier, LogisticRegression]

def train_model(
    model_params, train_features: pd.DataFrame, target: pd.DataFrame
) -> SklearnRegressionModel:
    """Обучает модель, используя указанные признаки и целевую переменную.

    Параметры:
        - model_params: Параметры модели, которую нужно обучить.
        - train_features: DataFrame с признаками для обучения модели.
        - target: Series с целевой переменной для обучения модели.

    Возвращает:
        Обученная модель (RandomForestClassifier или LogisticRegression).

    """
    model = instantiate(model_params).fit(train_features, target.ravel())#дописано
    return model


def serialize_model(model: SklearnRegressionModel, output: str) -> str:
    """Сериализует модель в файл.

    Параметры:
        - model: Обученная модель (RandomForestClassifier или LogisticRegression).
        - output: Строка, содержащая путь к файлу, в который нужно сохранить модель.

    Возвращает:
        Строка, содержащая путь к сохраненному файлу.

    """
    with open(output, "wb") as baf:
        pickle.dump(model, baf)
    return output

def predict_model(model: SklearnRegressionModel, test_features: pd.DataFrame) -> pd.Series:
    """Применяет обученную модель к тестовым данным и возвращает предсказания.

    Параметры:
        - model: Обученная модель (RandomForestClassifier или LogisticRegression).
        - test_features: DataFrame с признаками для применения модели.

    Возвращает:
        Series с предсказаниями модели.

    """
    predictions = model.predict(test_features)
    return pd.Series(predictions)


def evaluate_model(
    model: SklearnRegressionModel,
    test_features: pd.DataFrame,
    test_target: pd.Series
) -> float:
    """Оценивает качество модели на тестовых данных.

    Параметры:
        - model: Обученная модель (RandomForestClassifier или LogisticRegression).
        - test_features: DataFrame с признаками для оценки модели.
        - test_target: Series с целевой переменной для оценки модели.

    Возвращает:
        Значение метрики качества модели.

    """
    score = model.score(test_features, test_target)
    return score
