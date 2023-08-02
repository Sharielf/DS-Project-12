"""
Модуль, содержащий функции для предсказания и оценки модели.
"""

from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score #roc_auc_score

SklearnRegressionModel = Union[RandomForestRegressor, LogisticRegression]

def predict_model(
    model: SklearnRegressionModel, features: pd.DataFrame
) -> np.ndarray:
    """
    Функция для предсказания целевой переменной с использованием модели.

    Параметры:
        - model: Обученная модель для предсказания.
        - features: DataFrame с признаками, для которых нужно сделать предсказания.

    Возвращает:
        Нumpy массив с предсказанными значениями.

    """
    predicts = model.predict(features)
    return predicts

def evaluate_model(
    predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    """
    Функция для оценки качества предсказаний модели.

    Параметры:
        - predicts: Numpy массив с предсказанными значениями.
        - target: Series с истинными значениями целевой переменной.

    Возвращает:
        Словарь с оценками качества предсказаний, включая точность (accuracy).

    """
    return {
        "accuracy": accuracy_score(target, predicts),
    }
    