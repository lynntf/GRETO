"""
Handling for models, e.g., linear models, XGBoost models

TODO: Scikit-learn integration?
"""

import json
import warnings
from typing import Dict, Iterable, Optional

import numpy as np
import xgboost as xgb

from greto.fom_tools import FOM_model


class LinearModel:
    """
    Simple model that applies linear weights
    """

    def __init__(
        self,
        weights: Iterable[float],
        bias: float = 0.0,
        columns: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Create a linear model object:
        weights^T * data[columns] + bias = prediction

        Args:
            - weights: feature weight vector
            - bias: bias term (default is no bias; ranking does not need a bias)
            - columns: feature names
        """
        self.weights = np.array(weights)
        self.bias = bias
        self.columns = columns

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values by applying weights to input data X

        Args:
            - X: input features

        Returns:
            - dot product of features with model weights
        """
        return np.dot(X, self.weights) + self.bias


def save_linear_model(
    w: Iterable[float],
    bias: Optional[float] = None,
    columns: Optional[Iterable[str]] = None,
    filename: Optional[str] = None,
) -> Dict:
    """
    Save a linear model to a json

    Args:
        - w: model feature weights
        - bias: model bias term (not needed for ranking problems)
        - columns: feature names
        - filename: filename to write the model to
    """
    model_dict = {
        "model": "linear",
        "weights": list(w),
        "bias": bias,
        "columns": list(columns) if columns is not None else None,
    }

    if filename is not None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(model_dict, f)

    return model_dict


def load_linear_model(filename: str) -> Dict | LinearModel:
    """
    Read a json linear model
    """
    with open(filename, "r", encoding="utf-8") as f:
        d = json.load(f)
    if d.get("model") != "linear":
        warnings.warn("Loaded model may not be linear")
        return d
    return LinearModel(d["weights"], d["bias"], d["columns"])

def load_linear_FOM_model(filename: str) -> FOM_model:
    """
    Load a linear FOM model into the FOM_model class
    
    Args:
        - filename: filename for saved model
    """
    model = load_linear_model(filename)
    return FOM_model(model.predict, model.columns, model)

def save_xgbranker_model(ranker: xgb.XGBRanker, filename: str) -> None:
    """
    Save a XGBoost Ranker model

    Args:
        - ranker: trained XGBoost Ranker model
        - filename: filename for saved model [`.json` or `.ubj` (binary)]
    """
    ranker.save_model(filename)  # Saved model


def load_xgbranker_model(filename: str) -> xgb.XGBRanker:
    """
    Load an XGBoost Ranker model

    Args:
        - filename: filename for saved model [`.json` or `.ubj` (binary)]
    """
    ranker = xgb.XGBRanker()
    ranker.load_model(filename)
    return ranker

def load_xgbranker_FOM_model(filename:str) -> FOM_model:
    """
    Load an XGBoost Ranker model into the FOM_model class

    Args:
        - filename: filename for saved model [`.json` or `.ubj` (binary)]
    """
    ranker = load_xgbranker_model(filename)
    return FOM_model(ranker.predict, ranker.get_booster().feature_names, ranker)
