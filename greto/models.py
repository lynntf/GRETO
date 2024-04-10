"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Handling for models, e.g., linear models, XGBoost models

TODO: Scikit-learn integration?
"""

import json
import warnings
from typing import Dict, Iterable, List, Optional

import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression

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
        if bias is None:
            bias = 0.0
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
        warnings.warn(f"{__name__}: Loaded model ({filename}) may not be linear")
    return LinearModel(d["weights"], d["bias"], d["columns"])


def load_linear_FOM_model(filename: str) -> FOM_model:
    """
    Load a linear FOM model into the FOM_model class

    Args:
        - filename: filename for saved model
    """
    model = load_linear_model(filename)
    return FOM_model(model.predict, model.columns, model)


def save_xgbranker_model(
    ranker: xgb.XGBRanker, filename: str, scale: Optional[List[float]] = None
) -> None:
    """
    Save a XGBoost Ranker model

    Args:
        - ranker: trained XGBoost Ranker model
        - filename: filename for saved model [`.json` or `.ubj` (binary)]
        - scale: linear scaling factors for features (not required for ranking
          models, but may enhance performance)
    """
    d = json.loads(ranker.get_booster().save_raw("json"))  # Model as dict
    if scale is not None:
        d["scale_"] = list(scale)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(d, f)


def load_xgbranker_model(filename: str) -> xgb.XGBRanker:
    """
    Load an XGBoost Ranker model

    Args:
        - filename: filename for saved model [`.json` or `.ubj` (binary)]
    """
    ranker = xgb.XGBRanker()
    with open(filename, "r", encoding="utf-8") as f:
        d = json.load(f)
    scale = d.pop("scale_", None)
    ranker.load_model(bytearray(json.dumps(d), encoding="utf-8"))
    if scale is not None:
        return ranker, scale
    # ranker.load_model(filename)
    return ranker


def load_xgbranker_FOM_model(filename: str) -> FOM_model:
    """
    Load an XGBoost Ranker model into the FOM_model class

    Args:
        - filename: filename for saved model [`.json` or `.ubj` (binary)]
    """
    ranker = load_xgbranker_model(filename)
    if isinstance(ranker, tuple):
        ranker, scale = ranker
        return FOM_model(
            lambda x: ranker.predict(x / scale),
            ranker.get_booster().feature_names,
            ranker,
        )
    return FOM_model(ranker.predict, ranker.get_booster().feature_names, ranker)


def load_FOM_model(filename: str) -> FOM_model:
    """
    Load a FOM_model using data inside the saved model to determine the model
    type

    Args:
        - filename: filename for saved model [`.json`]
    """
    with open(filename, "r", encoding="utf-8") as f:
        d = json.load(f)
    if d.get("model") == "linear":
        print(f"Model {filename} is linear")
        return load_linear_FOM_model(filename)
    else:
        print(f"Model {filename} is assumed to be XGB Ranker")
        return load_xgbranker_FOM_model(filename)


class sns_model:
    """
    Singles and Non-singles model.

    Maintains two separate models, one for singles data, one for non-singles.
    Fits and predicts separately using these models and then combines them for a
    final combined model.
    """

    def __init__(
        self,
        model=LinearRegression,
        scaler=preprocessing.StandardScaler,
        use_combiner: bool = False,
        lower_bound: float = 0.0,
        upper_bound: float = 1e6,
        **kwargs,
    ):
        """
        Initialize the models and scalers

        Args:
            - model: the model type for singles/non-singles
            - scaler: the data rescaling method
            - use_combiner: combine the singles and non-single models using a third model
        """
        self.model = model

        self.scaler_ns = scaler()
        self.model_ns = model(**kwargs)
        self.pca_ns = PCA(n_components=0.95)

        self.scaler_s = scaler()
        self.model_s = model(**kwargs)
        self.pca_s = PCA(n_components=0.95)

        self.use_combiner = use_combiner
        if self.use_combiner:
            self.model_combiner = model(**kwargs)

        self.lb = lower_bound
        self.ub = upper_bound

    def split(self, X: np.ndarray, train_frac: float = 0.75, random_state: int = 42):
        """
        Split the data into testing and training

        Args:
            - X: training data
            - train_frac: fraction of data for training (1-train_frac for
              validation)
            - random_state: state for pseudo random number generation

        Returns:
            - training indices, validation indices
        """
        num_train = int(X.shape[0] * train_frac)
        rng = np.random.RandomState(random_state)
        indices = np.arange(X.shape[0], dtype=int)
        rng.shuffle(indices)
        return indices[:num_train], indices[num_train:]

    def fit(self, X: np.ndarray, y: np.ndarray, singles: np.ndarray) -> None:
        """
        Fit the classifiers to the training data

        Args:
            - X: n feature vectors for each of m gamma-rays [m by n]
            - y: 0 if gamma-ray should be in the spectrum, 1 if it should be excluded [m]
            - singles: indicator if the given data is a single interaction [m]
        """
        # Scale data using the provided scaler
        self.scaler_ns.fit(np.clip(X[~singles], self.lb, self.ub))

        # Fit a PCA model to the scaled data
        self.pca_ns.fit(
            self.scaler_ns.transform(np.clip(X[~singles], self.lb, self.ub))
        )

        # Fit the scaled and PCA transformed data for scattered gamma-rays with multiple interactions
        self.model_ns.fit(
            self.pca_ns.transform(
                self.scaler_ns.transform(np.clip(X[~singles], self.lb, self.ub))
            ),
            y[~singles].astype(float),
        )

        # Scale singles data
        self.scaler_s.fit(np.clip(X[singles], self.lb, self.ub))
        # Fit a separate PCA model to the scaled singles data
        self.pca_s.fit(self.scaler_s.transform(np.clip(X[singles], self.lb, self.ub)))
        # Fit the scaled and PCA transformed data for single interactions
        self.model_s.fit(
            self.pca_s.transform(
                self.scaler_s.transform(np.clip(X[singles], self.lb, self.ub))
            ),
            y[singles].astype(float),
        )

        if self.use_combiner:
            # If using a model to combine outputs of the singles and non-singles
            # models, generate the predictions from the models
            if self.model is LogisticRegression:
                pred_ns = self.model_ns.predict_proba(
                    self.scaler_ns.transform(np.clip(X[~singles], self.lb, self.ub))
                )[:, 1]
                pred_s = self.model_s.predict_proba(
                    self.scaler_s.transform(np.clip(X[singles], self.lb, self.ub))
                )[:, 1]
            else:
                pred_ns = self.model_ns.predict(
                    self.scaler_ns.transform(np.clip(X[~singles], self.lb, self.ub))
                )
                pred_s = self.model_s.predict(
                    self.scaler_s.transform(np.clip(X[singles], self.lb, self.ub))
                )

            # Join the predictions from both models
            joined_pred = self.join_predictions(pred_ns, pred_s, singles)

            # Fit the model that combines the predictions
            self.model_combiner.fit(joined_pred, y.astype(float))
            print("fit combiner")

    def join_predictions(
        self,
        pred_ns: np.ndarray,
        pred_s: np.ndarray,
        singles: np.ndarray,
        pad_value: float = 0.0,
    ) -> np.ndarray:
        """
        Join together predictions from a non-singles and a singles model into a
        two dimensional feature vector with zeros

        Args:
            - pred_ns: predictions from the non-singles model
            - pred_s: predictions from the singles model
            - singles: 0 if the gamma-ray is a non-single, 1 if a single

        Returns:
            The joined predictions where [:,0] are all predictions from the
            non-singles model, [:,1] are all predictions from the singles model
        """
        joined_pred = pad_value * np.ones((singles.shape[0], 2))
        joined_pred[~singles, 0] = pred_ns
        joined_pred[singles, 1] = pred_s
        return joined_pred

    def fit_train_data_dict(self, data: dict, train_frac=0.75, random_state=42):
        """
        Fit the models to the training data given a dictionary representation of
        the data

        Args:
            - data: a dictionary of the data with keywords: "features",
              "completeness", "length". These are converted into X, y and
              singles for the fit
        """
        train_inds, _test_inds = self.split(data["features"], train_frac, random_state)
        self.fit(
            data["features"][train_inds],
            data["completeness"][train_inds],
            data["length"][train_inds] == 1,
        )

    def fit_data_dict(self, data: dict):
        """
        Fit the models to all data given a dictionary representation of the data

        Args:
            - data: a dictionary of the data with keywords: "features",
              "completeness", "length". These are converted into X, y and
              singles for the fit
        """
        self.fit(data["features"], data["completeness"], data["length"] == 1)

    def predict(
        self,
        X: np.ndarray,
        singles: np.ndarray,
        weights: list = None,
        bias: list = None,
    ) -> np.ndarray:
        """
        Use the two models to get predictions

        Args:
            - X: features of the gamma-rays
            - singles: indicator of which gamma-rays are singles
            - weights: a vector with index [0] for weighting non-singles, [1] for singles
            - bias: a vector with index [0] for biasing non-singles, [1] for singles
        """
        if self.use_combiner:
            pred = np.zeros((singles.shape[0], 2))

            # Get predictions to pass to the combiner model
            if self.model is LogisticRegression:
                if np.sum(~singles) > 0:
                    # predict_proba for continuous LogisticRegression output
                    pred[~singles, 0] = self.model_ns.predict_proba(
                        self.pca_ns.transform(
                            self.scaler_ns.transform(
                                np.clip(X[~singles], self.lb, self.ub)
                            )
                        )
                    )[:, 1]
                if np.sum(singles) > 0:
                    pred[singles, 1] = self.model_s.predict_proba(
                        self.pca_s.transform(
                            self.scaler_s.transform(
                                np.clip(X[singles], self.lb, self.ub)
                            )
                        )
                    )[:, 1]
                # Get predictions from the combiner model
                pred = self.model_combiner.predict_proba(pred)[:, 1]
            else:
                if np.sum(~singles) > 0:
                    pred[~singles, 0] = self.model_ns.predict(
                        self.pca_ns.transform(
                            self.scaler_ns.transform(
                                np.clip(X[~singles], self.lb, self.ub)
                            )
                        )
                    )
                if np.sum(singles) > 0:
                    pred[singles, 1] = self.model_s.predict(
                        self.pca_s.transform(
                            self.scaler_s.transform(
                                np.clip(X[singles], self.lb, self.ub)
                            )
                        )
                    )
                # Get predictions from the combiner model
                pred = self.model_combiner.predict(pred)

            # Apply weights and biases
            if weights is not None:
                if len(weights) == 2:
                    pred[~singles] *= weights[0]
                    pred[singles] *= weights[1]
            if bias is not None:
                if len(bias) == 2:
                    pred[~singles] += bias[0]
                    pred[singles] += bias[1]
        else:  # Not using a third combiner model to join the prediction values
            pred = np.zeros((singles.shape[0],))

            if self.model is LogisticRegression:
                if np.sum(~singles) > 0:
                    pred[~singles] = self.model_ns.predict_proba(
                        self.pca_ns.transform(
                            self.scaler_ns.transform(
                                np.clip(X[~singles], self.lb, self.ub)
                            )
                        )
                    )[:, 1]
                if np.sum(singles) > 0:
                    pred[singles] = self.model_s.predict_proba(
                        self.pca_s.transform(
                            self.scaler_s.transform(
                                np.clip(X[singles], self.lb, self.ub)
                            )
                        )
                    )[:, 1]
            else:
                if np.sum(~singles) > 0:
                    pred[~singles] = self.model_ns.predict(
                        self.pca_ns.transform(
                            self.scaler_ns.transform(
                                np.clip(X[~singles], self.lb, self.ub)
                            )
                        )
                    )
                if np.sum(singles) > 0:
                    pred[singles] = self.model_s.predict(
                        self.pca_s.transform(
                            self.scaler_s.transform(
                                np.clip(X[singles], self.lb, self.ub)
                            )
                        )
                    )
        return pred

    def predict_separate(
        self,
        X: np.ndarray,
        singles: np.ndarray,
        weights: list = None,
        bias: list = None,
    ) -> np.ndarray:
        """
        It isn't clear how to combine the two predictions in the best way, so we
        may want their predictions separately

        Args:
            - X: features of the gamma-rays
            - singles: indicator of which gamma-rays are singles
            - weights: a vector with index [0] for weighting non-singles, [1]
              for singles
            - bias: a vector with index [0] for biasing non-singles, [1] for
              singles
        """
        pred = self.predict(X, singles, weights, bias)
        pred_ns = pred[~singles]
        pred_s = pred[singles]
        return pred_ns, pred_s

    def predict_data_dict(
        self, data: dict, weights: Optional[list] = None, bias: Optional[list] = None
    ):
        """
        Get predictions using data stored in a dictionary

        Args:
            - data: dictionary with keys "features" and "length"
            - weights: a vector with index [0] for weighting non-singles, [1]
              for singles
            - bias: a vector with index [0] for biasing non-singles, [1] for
              singles
        """
        return self.predict(
            data["features"], data["length"] == 1, weights=weights, bias=bias
        )

    def predict_test_data_dict(
        self,
        data: dict,
        train_frac: float = 0.75,
        random_state: int = 42,
        weights: Optional[list] = None,
        bias: Optional[list] = None,
    ) -> np.ndarray:
        """
        Given data provided in a dictionary, get predictions for testing data

        Args:
            - data: dictionary with keys "features", "lengths"
            - train_frac: fraction of data reserved for training (1-train_frac
              is testing reserved data)
            - random_state: pseudo random number generator state (for splitting
              training and testing)
            - weights: a vector with index [0] for weighting non-singles, [1]
              for singles
            - bias: a vector with index [0] for biasing non-singles, [1] for
              singles
        """
        _train_inds, test_inds = self.split(data["features"], train_frac, random_state)
        return (
            self.predict(
                data["features"][test_inds],
                data["length"][test_inds] == 1,
                weights=weights,
                bias=bias,
            ),
            data["completeness"][test_inds],
        )

    def predict_separate_data_dict(
        self, data: dict, weights: list = None, bias: list = None
    ) -> np.ndarray:
        """
        Given data provided in a dictionary, get separate predictions (singles
        and non-singles) for testing data

        Args:
            - data: dictionary with keys "features", "lengths"
            - train_frac: fraction of data reserved for training (1-train_frac
              is testing reserved data)
            - random_state: pseudo random number generator state (for splitting
              training and testing)
            - weights: a vector with index [0] for weighting non-singles, [1]
              for singles
            - bias: a vector with index [0] for biasing non-singles, [1] for
              singles
        """
        return self.predict_separate(
            data["features"], data["length"] == 1, weights=weights, bias=bias
        )
