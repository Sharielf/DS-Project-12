"""Этот модуль содержит функции обработки категориальных
и числовых признаков, другие классы"""
from typing import Any
from dataclasses import dataclass, field

@dataclass
class RFConfig: #added
    """Класс для конфигурации случайного леса"""
    _target_: str = field(default='sklearn.ensemble.RandomForestClassifier')
    random_state: int = field(default=42)
    max_depth: int = field(default=3)
    n_estimators: int = field(default=100)

@dataclass
class LogregConfig:
    """Класс для логистической регрессии""" 
    _target_: str = field(default='sklearn.linear_model.LogisticRegression')
    penalty: str = field(default='l1')
    solver: str = field(default='liblinear')
    #C: float = field(default=1.0)
    random_state: int = field(default=42)
    max_iter: int = field(default=100)


@dataclass
class ModelConfig:
    """
    Класс для описания модели
    """
    model_name: str
    model_params: Any
    