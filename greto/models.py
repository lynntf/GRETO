"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Handling for models, e.g., linear models, XGBoost models

TODO: Scikit-learn integration?
"""

import json
import pickle as pkl
import warnings
from typing import Dict, Iterable, Optional

import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression

import greto.fast_features as ff
from greto.fom_tools import FOM_model

# , column_names_to_bool, permute_column_names


class LinearModel:
    """
    Simple model that applies linear weights
    """

    def __init__(
        self,
        weights: Iterable[float],
        bias: float = 0.0,
        scale: Iterable[float] = None,
        columns: Optional[Iterable[str]] = None,
        weight_threshold: float = 1e-8,
    ) -> None:
        """
        Create a linear model object:
        weights^T * data[columns] + bias = prediction

        Args:
            - weights: feature weight vector
            - bias: bias term (default is no bias; ranking does not need a bias)
            - scale: scale factor for features; computed by division: features / scale
            - columns: feature names
        """
        # Feature names may not be provided in the order that they are created by other code
        # This maps weights (etc.) to an order that matches features
        # self.permutation = permute_column_names(columns)
        self.permutation = ff.permute_column_names(columns)
        self.weights = np.array(weights)[self.permutation]

        # Get only the weights that exceed the threshold
        self.weight_threshold = weight_threshold
        self.weight_indicator = np.abs(self.weights) > self.weight_threshold

        if bias is None:
            bias = 0.0
        self.bias = bias
        if scale is None:
            scale = np.ones(self.weights.shape)
        self.scale = scale[self.permutation]
        self.columns = np.array(columns)[self.permutation]

        # select down weights, scale, and columns to only those that exceed the threshold
        self.weights_thresholded = self.weights[self.weight_indicator]
        self.scale_thresholded = self.scale[self.weight_indicator]
        self.columns_thresholded = self.columns[self.weight_indicator]

        # self.columns_bool = column_names_to_bool(self.columns_thresholded)
        self.boolean_vectors = ff.convert_feature_names_to_boolean_vectors(
            self.columns_thresholded
        )

        self.effective_weights = self.weights_thresholded / self.scale_thresholded

    def change_weight_threshold(self, weight_threshold: float = 1e-8) -> None:
        """
        Change the weight thresholding value
        """
        # Get only the weights that exceed the threshold
        self.weight_threshold = weight_threshold
        self.weight_indicator = np.abs(self.weights) > self.weight_threshold

        # select down weights, scale, and columns to only those that exceed the threshold
        self.weights_thresholded = self.weights[self.weight_indicator]
        self.scale_thresholded = self.scale[self.weight_indicator]
        self.columns_thresholded = self.columns[self.weight_indicator]

        # self.columns_bool = column_names_to_bool(self.columns_thresholded)
        self.boolean_vectors = ff.convert_feature_names_to_boolean_vectors(
            self.columns_thresholded
        )

        self.effective_weights = self.weights_thresholded / self.scale_thresholded

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values by applying weights to input data X

        Args:
            - X: input features

        Returns:
            - dot product of features with model weights
        """
        return np.dot(X, self.effective_weights) + self.bias


def save_linear_model(
    w: Iterable[float],
    scale: Optional[Iterable[float]] = None,
    bias: Optional[float] = None,
    columns: Optional[Iterable[str]] = None,
    filename: Optional[str] = None,
) -> Dict:
    """
    Save a linear model to a json

    Args:
        - w: model feature weights
        - scale: weight scaling (if provided, weights require scaling)
        - bias: model bias term (not needed for ranking problems)
        - columns: feature names
        - filename: filename to write the model to
    """
    model_dict = {
        "model": "linear",
        "weights": list(w),
        "scale": list(scale),
        "bias": bias,
        "columns": list(columns) if columns is not None else None,
    }

    if filename is not None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(model_dict, f)

    return model_dict


def load_linear_model(
    filename: str, weight_threshold: float = 1e-8
) -> Dict | LinearModel:
    """
    Read a json linear model
    """
    with open(filename, "r", encoding="utf-8") as f:
        d = json.load(f)
    if d.get("model") != "linear":
        warnings.warn(f"{__name__}: Loaded model ({filename}) may not be linear")
    return LinearModel(
        weights=d.get("weights"),
        scale=d.get("scale"),
        bias=d.get("bias"),
        columns=d.get("columns"),
        weight_threshold=weight_threshold,
    )


def load_linear_FOM_model(filename: str, weight_threshold: float = 1e-8) -> FOM_model:
    """
    Load a linear FOM model into the FOM_model class

    Args:
        - filename: filename for saved model
    """
    model = load_linear_model(filename, weight_threshold=weight_threshold)
    return FOM_model(
        model_evaluation=model.predict,
        boolean_vectors=model.boolean_vectors,
        model=model,
    )


class xgbranker_FOM_model:
    """
    Ranking model using XGBoost

    Feature scaling is not important for boosted tree models so we don't use it.
    """

    def __init__(self, ranker) -> None:
        self.model = ranker
        self.columns = ranker.get_booster().feature_names

        # self.columns_bool = column_names_to_bool(self.columns)
        self.boolean_vectors = ff.convert_feature_names_to_boolean_vectors(self.columns)

        # Here we need to map the data onto the structure of the model so we
        # need the inverse permutation
        self.permutation = np.argsort(ff.permute_column_names(self.columns))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values by applying weights to input data X

        Args:
            - X: input features, shape: (sample index, feature index)

        Returns:
            - ranker inferred scores
        """
        # perm_X = np.copy(X[:, self.permutation])
        return self.model.get_booster().inplace_predict(X[:, self.permutation])
        # return self.model.predict(perm_X)


def save_xgb_model(
    model: xgb.XGBRanker | xgb.XGBClassifier,
    filename: str,
    columns:list[str] = None,
) -> None:
    """
    Save a XGBoost Ranker model (feature scaling is not used)

    Args:
        - ranker: trained XGBoost Ranker model
        - filename: filename for saved model [`.json` or `.ubj` (binary)]
    """
    # d = json.loads(ranker.get_booster().save_raw("json"))  # Model as dict
    # with open(filename, "w", encoding="utf-8") as f:
    #     json.dump(d, f)
    if columns is not None:
        model.get_booster().feature_names = columns
    model.save_model(filename)


def load_xgbranker_model(filename: str) -> xgb.XGBRanker:
    """
    Load an XGBoost Ranker model

    Args:
        - filename: filename for saved model [`.json` or `.ubj` (binary)]
    """
    ranker = xgb.XGBRanker()
    ranker.load_model(filename)
    ranker.get_booster().set_param({"device": "cpu", "nthread": 1})
    return ranker


def load_xgbranker_FOM_model(filename: str) -> FOM_model:
    """
    Load an XGBoost Ranker model into the FOM_model class

    Args:
        - filename: filename for saved model [`.json` or `.ubj` (binary)]
    """
    ranker = load_xgbranker_model(filename)
    return xgbranker_FOM_model(ranker)


def load_order_FOM_model(filename: str) -> FOM_model:
    """
    Load a FOM_model using data inside the saved model to determine the model
    type

    Args:
        - filename: filename for saved model [`.ubj` or `.json`]
    """
    if filename.endswith(".ubj"):
        print(f"Model {filename} is assumed to be XGB Ranker")
        return load_xgbranker_FOM_model(filename)
    with open(filename, "r", encoding="utf-8") as f:
        d = json.load(f)
    if d.get("model") == "linear":
        print(f"Model {filename} is linear")
        return load_linear_FOM_model(filename)
    else:
        print(f"Model {filename} is assumed to be XGB Ranker")
        return load_xgbranker_FOM_model(filename)


def load_xgbclassifier_FOM_model(filename):
    """Load an XGBoost classifier"""
    lb = 0
    ub = 1e16
    classifier = xgb.XGBClassifier()
    classifier.load_model(filename)
    columns = classifier.get_booster().feature_names
    if columns is None:
        columns = ff.all_feature_names
    return FOM_model(
        lambda X: classifier.predict_proba(np.clip(X, lb, ub))[:, 0],
        columns,
        None,
        classifier,
    )


def load_suppression_FOM_model(filename: str) -> FOM_model:
    """
    Load a FOM_model using data inside the saved model to determine the model
    type

    Args:
        - filename: filename for saved model [`.pkl` or `.ubj`]
    """
    if filename.endswith(".ubj"):
        print(f"Model {filename} is assumed to be XGB Ranker")
        return load_xgbclassifier_FOM_model(filename)
    if filename.endswith(".pkl"):
        with open(filename, "rb") as f:
            model = pkl.load(f)
            return FOM_model(model.predict, ff.all_feature_names, None, model)


class sns_model:
    """
    Singles and Non-singles model.

    Maintains two separate models, one for singles data, one for non-singles.
    Fits and predicts separately using these models and then combines them for a
    final combined model.

    Q: Why might we want separate models for singles and non-singles?
    A: Data imbalance. Different behaviors. Inability to capture either
    independently in a single model. Additional model flexibility. Independent
    feature scaling (keeping zero features changes scaling).

    In general, we should not expect that a model need to be trained on singles
    and non-singles separately. Having zero value features for the irrelevant
    values should be sufficient. However, it seems that performance is better
    when separating the two. Given a value from a singles model, how should that
    be compared to a non-singles output model value? The two are separate
    objectives, so there is some Pareto front where we can balance the two
    objectives.

    We add in more flexibility in this model because we are less constrained by
    computation than in the ordering case. We add in a PCA fit, a scaling function.
    """

    def __init__(
        self,
        model_class=LinearRegression,
        scaler_class=preprocessing.StandardScaler,
        use_combiner: bool = False,
        lower_bound: float = 0.0,
        upper_bound: float = 1e6,
        pca_n_components: float = 0.95,
        columns: list[str] = None,
        **kwargs,
    ):
        """
        Initialize the models and scalers

        Args:
            - model: the model type for singles/non-singles
            - scaler: the data rescaling method
            - use_combiner: combine the singles and non-single models using a third model
        """
        self.model = model_class

        self.columns = columns
        if self.columns is None:
            self.columns = ff.all_feature_names

        self.scaler_ns = scaler_class()
        self.model_ns = model_class(**kwargs)
        self.pca_ns = PCA(n_components=pca_n_components)

        self.scaler_s = scaler_class()
        self.model_s = model_class(**kwargs)
        self.pca_s = PCA(n_components=pca_n_components)

        self.use_combiner = use_combiner
        if self.use_combiner:
            self.model_combiner = model_class(**kwargs)

        self.lb = lower_bound
        self.ub = upper_bound

    @classmethod
    def split(self, X: np.ndarray, train_frac: float = 0.75, random_state: int = 42):
        """
        Get indices that split the data into testing and training

        Args:
            - X: training data
            - train_frac: fraction of data for training (1-train_frac for
              validation)
            - random_state: state for pseudo random number generation

        Returns:
            - training indices, validation indices
        """
        num_train = int(X.shape[0] * train_frac)
        rng = np.random.RandomState(random_state)  # pylint: disable=no-member
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

        # Fit the scaled and PCA transformed data for scattered gamma-rays with
        # multiple interactions
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
                    self.pca_ns.transform(
                        self.scaler_ns.transform(np.clip(X[~singles], self.lb, self.ub))
                    )
                )[:, 1]
                pred_s = self.model_s.predict_proba(
                    self.pca_s.transform(
                        self.scaler_s.transform(np.clip(X[singles], self.lb, self.ub))
                    )
                )[:, 1]
            else:
                pred_ns = self.model_ns.predict(
                    self.pca_ns.transform(
                        self.scaler_ns.transform(np.clip(X[~singles], self.lb, self.ub))
                    )
                )
                pred_s = self.model_s.predict(
                    self.pca_s.transform(
                        self.scaler_s.transform(np.clip(X[singles], self.lb, self.ub))
                    )
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
              "completeness", "length". These are converted into X, y, and
              singles boolean for the fit
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
