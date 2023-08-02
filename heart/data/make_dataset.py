"""
Модуль для чтения и разбиения датасета на выборки
"""
# -*- coding: utf-8 -*-
#from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(dataset_path: str) -> pd.DataFrame:
    """
    Read dataset from the given path.

    Args:
        dataset_path: The path to the dataset.

    Returns:
        The dataset as a pandas DataFrame.
    """
    data = pd.read_csv(dataset_path)
    return data

#Added
def split_train_test_data(
        dataset: pd.DataFrame,
        test_size: float,
        random_state: int
    ) -> pd.DataFrame:
    """
    Split the dataset into random train and test subsets.

    Args:
        dataset: The dataset to be split.
        test_size: The proportion of the dataset to include in the test split.
        random_state: The random seed for reproducibility (default: None).

    Returns:
        A tuple containing the train and test subsets as pandas DataFrames.
    """
    train, test = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state
    )
    return train, test
