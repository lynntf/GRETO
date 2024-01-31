"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Methods for finding an optimally separating hyperplane between clusters of
points. Primarily used in the case where the clusters are all the same class and
we wish to find the hyperplane through the origin that places as many of those
clusters on the positive side as possible

Methods include index selection for column generation, initial estimates for
weights, weighted and unweighted SVM methods using scikit-learn, weighted
unweighted logistic regression methods using scikit-learn, SVM methods using
cvxpy, and linear relaxations of the binary variable problem using cvxpy.
"""
from __future__ import annotations

import datetime
import warnings
from typing import Callable, Dict, List, Literal, Tuple

import cvxpy as cp
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# %% w0 estimate for linear model


def mean_of_means(
    X: np.ndarray[float],
    qid: np.ndarray[int],
    non_neg: bool = True,
    norm_ord: int = 1,
) -> np.ndarray:
    """
    Compute an estimate for the hyperplane normal `w` by getting average
    direction from `X` for each query-id in `qid` and then averaging all of
    these averages.

    The first averaging re-weights the data in `X` such that each `qid` is
    assigned a single average direction. The second averaging finds the average
    direction of all `qid` averages.

    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param non_neg:  only return non negative weights (negative weights are zero)
    :param norm_ord:  order of norm used to normalize hyperplane weights `w`

    :return:  initial weight vector `w` estimate `(N,)`
    """
    means = []
    for cluster_id in np.unique(qid):
        means.append(np.mean(X[qid == cluster_id, :], axis=0))
    means = np.array(means)
    w = np.mean(means, axis=0)
    if np.sum(w > 0) > 0 and non_neg:
        w[w < 0] = 0.0
    w = w / np.linalg.norm(w, ord=norm_ord, keepdims=True)  # Normalize
    return w


# %% Data selection methods for linear model


def all_indices(
    w: np.ndarray,  # pylint: disable=unused-argument
    X: np.ndarray,  # pylint: disable=unused-argument
    qid: np.ndarray,
    w_prev: np.ndarray = None,  # pylint: disable=unused-argument
) -> List[int]:
    """
    Given w, data X, and qid, return all indices

    Function arguments match other selection methods, but `w` and `X` are unused

    :param w:  weight vector (N,)
    :param X:  input data matrix (M, N)
    :param qid:  problem labels/query-ids `(M,)`
    :return:  indices of data that define the problem
    """
    selected_indices = list(range(len(qid)))
    return selected_indices


def select_indices(
    w: np.ndarray,
    X: np.ndarray,
    qid: np.ndarray,
    w_prev: np.ndarray = None,  # pylint: disable=unused-argument
) -> List[int]:
    """
    Given `w`, data `X`, and `qid`, find the indices of data `X` that are the worst

    Computes `wTx` for each label in `qid` and selects the one that's the worst (most negative)

    :param w:  weight vector `(N,)`
    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param w_prev:  previously used `w` vector `(N,)`
    :return:  indices of data that define the problem
    """
    selected_indices = []
    indices = np.arange(X.shape[0])
    wtX = np.dot(X, w)
    for cluster_id in np.unique(qid):
        mask = qid == cluster_id  # Only examine the data in the current cluster
        # Select the index from that cluster with the smallest value
        selected_indices.append(indices[mask][np.argmin(wtX[mask])])
    return selected_indices


def select_indices_pos_neg(
    w: np.ndarray,
    X: np.ndarray,
    qid: np.ndarray,
    w_prev: np.ndarray = None,  # pylint: disable=unused-argument
) -> List[int]:
    """
    Given `w`, data `X`, and `qid`, find the indices of data `X` that are the worst if
    negative and best if positive

    Computes `wTx` for each label in `qid` and selects the one that's the worst (most
    negative) if the worst is negative and best (most positive) if the worst is
    positive

    :param w:  weight vector `(N,)`
    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param w_prev:  previously used `w` vector `(N,)`
    :return:  indices of data that might lead to a better `w` estimate
    """
    selected_indices = []
    indices = np.arange(X.shape[0])
    wtX = np.dot(X, w)
    for cluster_id in np.unique(qid):
        mask = qid == cluster_id  # Only examine the data in the current cluster
        # Select the index from that cluster with the smallest value
        index = indices[mask][np.argmin(wtX[mask])]
        # If that value is positive, then choose the largest value instead
        if wtX[index] >= 0:
            index = indices[mask][np.argmax(wtX[mask])]
        selected_indices.append(index)
    return selected_indices


def select_indices_neg(
    w: np.ndarray,
    X: np.ndarray,
    qid: np.ndarray,
    w_prev: np.ndarray = None,  # pylint: disable=unused-argument
) -> List[int]:
    """
    Given `w`, data `X`, and `qid`, find the indices of data `X` that are the worst if negative

    Computes `wTx` for each label in qid and selects the one that's the worst (most
    negative) if negative, otherwise doesn't select anything

    :param w:  weight vector `(N,)`
    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param w_prev:  previously used `w` vector `(N,)`
    :return:  indices of data to add to the problem
    """
    selected_indices = []
    indices = np.arange(X.shape[0])
    wtX = np.dot(X, w)
    for cluster_id in np.unique(qid):
        mask = qid == cluster_id
        index = indices[mask][np.argmin(wtX[mask])]
        if wtX[index] <= 0:
            selected_indices.append(index)
    return selected_indices


def select_indices_change_sign(
    w: np.ndarray, X: np.ndarray, qid: np.ndarray, w_prev: np.ndarray
) -> List[int]:
    """
    Given `w`, data `X`, and `qid`, find the indices of data `X` that are the worst if the
    sign changed from the last time (need to find a new representative for the
    cluster for system to be correct)

    Computes `wTx` for each label in `qid` and selects the one that's the worst (most
    negative) if newly negative, otherwise doesn't select anything. Values that
    continue to be negative or positive are left alone, they continue to
    represent the cluster.

    :param w:  weight vector `(N,)`
    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param w_prev:  previously used `w` vector `(N,)`
    :return:  indices of data to add to the problem
    """
    if w_prev is None:
        return select_indices_neg(w, X, qid)
    selected_indices = []
    indices = np.arange(X.shape[0])
    wTx = np.dot(X, w)
    w_prevTx = np.dot(X, w_prev)
    # Only need to check the ones where the sign changed from positive to negative
    changed_sign = np.logical_and(np.sign(wTx) < 0, np.sign(w_prevTx) > 0)
    for cluster_id in np.unique(qid):
        mask = qid == cluster_id
        index = indices[mask][np.argmin(wTx[mask])]
        if wTx[index] <= 0 and changed_sign[index]:
            selected_indices.append(index)
    return selected_indices


def select_indices_n_neg(
    w: np.ndarray,
    X: np.ndarray,
    qid: np.ndarray,
    n: int = 2,
    w_prev: np.ndarray = None,  # pylint: disable=unused-argument
) -> List[int]:
    """
    Given `w`, data `X`, and `qid`, find the indices of data `X` that are the worst if negative

    Computes `wTx` for each label in `qid` and selects the `n` that are the worst (most
    negative) if negative, otherwise doesn't select anything

    :param w:  weight vector `(N,)`
    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param n:  number of worst indices to select
    :param w_prev:  previously used `w` vector `(N,)`
    :return:  indices of data to add to the problem
    """
    selected_indices = []
    indices = np.arange(X.shape[0])
    wtX = np.dot(X, w)
    for cluster_id in np.unique(qid):
        mask = qid == cluster_id
        if n is not None:
            possible_indices = indices[mask][np.argsort(wtX[mask])[:n]]
        else:
            possible_indices = indices[mask][np.argsort(wtX[mask])]
        for index in possible_indices:
            if wtX[index] <= 0:
                selected_indices.append(index)
    return selected_indices


def select_indices_all_neg(
    w: np.ndarray,
    X: np.ndarray,
    qid: np.ndarray,  # pylint: disable=unused-argument
    w_prev: np.ndarray = None,  # pylint: disable=unused-argument
) -> List[int]:
    """
    Given `w`, data `X`, and `qid`, find the all of the indices of data `X` that are negative

    Computes `wTx` and selects all of them that are negative (could be many).

    :param w:  weight vector `(N,)`
    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param w_prev:  previously used `w` vector `(N,)`
    :return:  indices of data to add to the problem
    """
    selected_indices = []
    indices = np.arange(X.shape[0])
    wtX = np.dot(X, w)
    selected_indices = indices[wtX <= 0]
    return selected_indices


def select_indices_best(
    w: np.ndarray,
    X: np.ndarray,
    qid: np.ndarray,
    w_prev: np.ndarray = None,  # pylint: disable=unused-argument
) -> List[int]:
    """
    Given `w`, data `X`, and `qid`, find the indices of data `X` that are closest to the decision boundary

    Computes `wTx` for each label in qid and selects the one closest to the decision
    boundary if all are positive and closest to the decision boundary on the
    negative side if any are negative.

    :param w:  weight vector `(N,)`
    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param w_prev:  previously used `w` vector `(N,)`
    :return:  indices of data that define the problem
    """
    selected_indices = []
    indices = np.arange(X.shape[0])
    wtX = np.dot(X, w)
    for cluster_id in np.unique(qid):
        mask = qid == cluster_id
        if np.min(wtX[mask]) < 0:
            mask = np.logical_and(mask, wtX < 0)
            selected_indices.append(indices[mask][np.argmax(wtX[mask])])
        else:
            selected_indices.append(indices[mask][np.argmin(wtX[mask])])
    return selected_indices


def select_indices_neg_best(
    w: np.ndarray,
    X: np.ndarray,
    qid: np.ndarray,
    w_prev: np.ndarray = None,  # pylint: disable=unused-argument
) -> List[int]:
    """
    Given `w`, data `X`, and `qid`, find the indices of data `X` that are the best
    performing wrong values

    Computes `wTx` for each label in qid and selects the one that's the best (least
    negative) if any are negative, otherwise doesn't select anything

    :param w:  weight vector `(N,)`
    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param w_prev:  previously used `w` vector `(N,)`
    :return:  indices of data to add to the problem
    """
    selected_indices = []
    indices = np.arange(X.shape[0])
    wtX = np.dot(X, w)
    for cluster_id in np.unique(qid):
        mask = np.logical_and(qid == cluster_id, wtX < 0)
        if np.sum(mask) > 0:
            index = indices[mask][np.argmax(wtX[mask])]
            selected_indices.append(index)
    return selected_indices


# %% Initial guess method with limited data


# estimation method:
def solve_estimate(
    X: np.ndarray,
    qid: np.ndarray,
    initial_selection_method: Callable = select_indices,
    subsequent_selection_method: Callable = select_indices_neg,
    max_iter: int = 10,
    allow_duplicates=False,
    debug: bool = False,
    non_neg: bool = True,
    return_indices: bool = False,
) -> np.ndarray:
    """
    Iteratively apply the mean_of_means method to get an estimate solution.
    Start with all of the data, then select the worst from it, and recompute the mean_of_means.
    Progressively add data as needed (solution should only depend on worst performers)

    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param initial_selection_method:  Initial method for selecting data indices
    :param subsequent_selection_method:  Method for selecting data indices after the initial
    :param max_iter:  maximum number of iterations
    :param allow_duplicates:  expand the set of selected indices (allowing duplicate
        indices; shifts mean_of_means towards frequent indices)
    :return:  estimate for weight vector `w`
    """
    if debug:
        print("Computing initial estimate for w")
    w0 = mean_of_means(X, qid, non_neg=non_neg)
    total_problems = len(np.unique(qid))
    selected_indices = initial_selection_method(w0, X, qid)
    num_unsolved = len(select_indices_neg(w0, X, qid))
    if debug:
        print(f"Data contains {total_problems} problems")
        print(
            f"  w0 solves {total_problems - num_unsolved} problems"
            + f" ({(total_problems - num_unsolved)/total_problems*100:5.2f}%)"
        )
        print(f"Initial data usage: {len(selected_indices)} / {X.shape[0]}")
        print("Selecting specific data to attempt to improve the solution")
    old_num_indices = 0
    num_iters = 0
    w_prev = w0
    while len(selected_indices) > old_num_indices and num_iters < max_iter:
        num_iters += 1
        # Compute a new estimate using the selected indices
        w0 = mean_of_means(X[selected_indices], qid[selected_indices], non_neg=non_neg)
        new_indices = subsequent_selection_method(w0, X, qid, w_prev)
        w_prev = w0
        num_unsolved = len(select_indices_neg(w0, X, qid))
        old_num_indices = len(selected_indices)
        if not allow_duplicates:
            selected_indices = list(set(new_indices) | set(selected_indices))
        else:
            selected_indices.extend(new_indices)
        if debug:
            print(
                f"{num_iters:4d}:  {total_problems - num_unsolved:6d} solved; "
                + f"{len(set(selected_indices)):6d} / {X.shape[0]} data used",
                end="",
            )
            if allow_duplicates:
                print(
                    f" ({len(selected_indices)-len(set(selected_indices)):5d} "
                    + "duplicates)",
                    end="",
                )
            print(
                f";  {np.sum(w0 == 0)}/{X.shape[1]} "
                + f"({np.sum(w0 == 0)/X.shape[1]*100:5.2f}%) zero features"
            )
    if debug:
        if num_iters >= max_iter:
            print("Reached maximum iteration")
        if len(selected_indices) <= old_num_indices:
            print("No new indices added, estimate is stable")
    if return_indices:
        return w0, list(set(selected_indices))
    return w0


# %% Evaluation method
def check_solutions(
    w: np.ndarray, X: np.ndarray, qid: np.ndarray, debug: bool = False
) -> Tuple[int, int]:
    """
    Check how many of the problems are solved by hyperplane weights `w`

    Counts the number of problems `wTX` that have no negative values, one or more
    negative values, and a percentage of the number solved

    :param w:  weight vector `(N,)`
    :param X:  input data matrix (relative data)
    :param qid:  problem labels/query-ids `(M,)`
    :return:  number of solved problems (no negatives), number of unsolved
        problems (at least one negative)
    """
    wTX = np.dot(X, w)
    num_solved = 0
    num_unsolved = 0
    for cluster_id in np.unique(qid):
        if np.sum(wTX[cluster_id == qid] <= 0) == 0:
            num_solved += 1
        else:
            num_unsolved += 1
    if debug:
        print(
            f"Solved {num_solved}/{num_solved + num_unsolved} "
            + f"({num_solved/(num_solved + num_unsolved)*100:5.2f}%)"
        )
    return num_solved, num_unsolved  # , num_unsolved/(num_unsolved + num_solved) * 100


def check_solutions_baseline(X: np.ndarray, qid: np.ndarray) -> List:
    """
    Check how many of the problems are solved by the features of `X` to get a
    baseline performance

    Counts the number of problems `wTX` that have no negative values, one or more
    negative values, and a percentage of the number solved

    :param X:  input data matrix (relative data)
    :param qid:  problem labels/query-ids `(M,)`
    :return:  number of solved problems (no negatives), number of unsolved
        problems (at least one negative)
    """
    output = np.array([[0, 0]] * X.shape[1])
    for cluster_id in np.unique(qid):
        s = np.sum(X[cluster_id == qid, :] <= 0, axis=0)
        for i in range(X.shape[1]):
            if s[i] == 0:
                output[i][0] += 1
            else:
                output[i][1] += 1
    return output


def check_solutions_general(
    pred: np.ndarray,
    qid: np.ndarray,
    y: np.ndarray[bool],
    debug: bool = False,
    return_report: bool = False,
) -> Tuple[int, int] | Dict:
    """
    Given predictions `pred`, how many of the problems are solved

    :param pred:  the predictions of the ranking model
    :param qid:  the problem indices (also query ids)
    :param y:  indicator if the data index is not a solution (1 if acceptable and
        or correct, 0 otherwise)
    :param debug:  print additional messages fro debugging
    :param return_report:  return a dictionary with the solved status of the `qid`s
    :return:  (num correct, num incorrect) or Dict[qid] = 1 or 0
    """
    indices = np.arange(y.shape[0], dtype=int)
    num_solved = 0
    num_unsolved = 0
    solved = {}
    for i in np.unique(qid):
        local_predictions = pred[qid == i]
        local_indices = indices[qid == i]
        local_indices_matching_min = local_indices[
            local_predictions == np.min(local_predictions)
        ]
        acceptability_of_indices_matching_min = y[local_indices_matching_min]
        # if all(y[indices[qid == i][pred[qid == i] == np.min(pred[qid == i])]]):
        if all(acceptability_of_indices_matching_min > 0):
            num_solved += 1
            solved[i] = True
        else:
            num_unsolved += 1
            solved[i] = False
    if debug:
        print(
            f"Solved {num_solved}/{num_solved + num_unsolved} "
            + f"({num_solved/(num_solved + num_unsolved)*100:5.2f}%)"
        )
    if return_report:
        return solved
    return num_solved, num_unsolved


def get_data_weight(qid: np.ndarray, indices: List[int] = None) -> np.ndarray:
    """
    Count the number of data points for each problem in the provided indices and
    provides weights for each indexed data point

    Returns a weighting such that each problem is given the total same weight
    for a typical SVM solve (problems with more data are not overly favored).
    Indices are not necessary for computing the weights,
    `get_data_weight(qid[indices])` is equivalent to `get_data_weight(qid,
    indices)`.

    :param qid:  problem labels/query-ids `(M,)`
    :param indices:  the indices of the currently selected data
    :return:  1/(the number of data points included for each problem)
    """
    if indices is None:
        indices = np.arange(qid.shape[0], dtype=int)
    weights = np.zeros(np.array(indices).shape)
    for i, _ in enumerate(weights):
        weights[i] = 1 / np.sum(qid[indices] == qid[indices[i]])
    return weights


# %% SVM methods


def fix_problem_labels(qid: np.ndarray):
    """
    The problem labels `qid` may not necessarily be integers from 0 to the number of
    problems present

    In the case that some problems are excluded from the training, the problem
    labels need to be adjusted to work with the LP formulations (indexes on
    cluster data should be `0..`number of clusters). This involves mapping the
    integer problem labels to the integers `0..`number of problems present in the
    data.

    :param qid:  problem integer labels
    :return:  a mapping of the problem integer labels to `0..`(number of problems)
    """
    _, adjusted_qid = np.unique(qid, return_inverse=True)
    return adjusted_qid.astype(int)


def weighted_svc(
    X: np.ndarray,
    qid: np.ndarray,
    indices: List[int] = None,
    non_neg: bool = True,
    weighted: bool = True,
    penalty: Literal["l1", "l2"] = "l1",
    loss: Literal["squared_hinge", "hinge"] = "squared_hinge",
    dual: bool = False,
    tol: float = 0.0001,
    C: float = 1e2,
    fit_intercept: bool = False,
    debug: bool = False,  # pylint: disable=unused-argument
    relaxation: bool = False,  # pylint: disable=unused-argument
    mirror: bool = True,
    **SVCkwargs,
) -> None:
    """
    Weighted Linear Support Vector Classifier (SVC)

    If multiple indices from the same problem are provided, the sample weight is
    adjusted to compensate. This SVC method requires two classes centered around
    the origin, so we introduce a mirrored second class (this duplicates the
    data and requires `C` to be halved to correct for the now doubled loss).
    Non-negative support is not present, so any negative weights are set to zero
    if non-negative weights are desired. An l1 penalty and a hinge loss is also
    not supported by this method.

    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param indices:  selected data indices
    :param non_neg:  set negative weights in `w` to zero if `True`; rudimentary clipping
    :param weighted:  weight selected data based on cluster participation
    :param penalty:  regularization penalty parameter
    :param loss:  slack error term (hinge is max(0, 1 - y_i(wT x_i + b)); squared_hinge is (hinge)^2)
    :param dual:  solve the dual problem or not; default is `False`
    :param tol:  numerical tolerance for solution
    :param C:  regularization parameter (large `C` means regularization is not as important)
    :param fit_intercept:  the default SVC will try to find an intercept
        parameter; this is zero by design for relative data, and will be
        approximately zero due to data mirroring if fitting the intercept is
        allowed
    :param mirror:  mirror the data (introduce negative copies of data points)
        or introduce a negative class for some of the data
    :param **SVCkwargs:  extra keyword args for the SVC (e.g., `max_iter`, `verbose`)
    :return:  weight vector `w` for the centered SVC
    """
    if indices is None:
        indices = np.arange(X.shape[0], dtype=int)
    if weighted:
        weights = get_data_weight(qid, indices=indices)

    num_clusters = np.sum(weights)

    clf = LinearSVC(
        penalty=penalty,
        loss=loss,
        dual=dual,
        tol=tol,
        C=C / 2 / num_clusters,
        # C=C/2,
        fit_intercept=fit_intercept,
        **SVCkwargs,
    )
    selected_X = X[indices]
    with warnings.catch_warnings(record=True) as w:  # Catch convergence warnings
        if mirror:  # Duplicate data into a second negative class
            X_mirror = np.concatenate((selected_X, -selected_X), axis=0)
            y = np.ones((selected_X.shape[0],))
            y_mirror = np.concatenate((y, -y), axis=0)

            if weighted:
                weights_mirror = np.concatenate((weights, weights), axis=0)
                clf.fit(X_mirror, y_mirror, sample_weight=weights_mirror)
            else:
                clf.fit(X_mirror, y_mirror)
        else:  # Introduce a second negative class by swapping some data to the other class
            # y = np.random.choice([-1,1], size = (selected_X.shape[0],))
            y = np.ones((selected_X.shape[0],))
            y[: len(y) // 2] = -1
            if weighted:
                clf.fit(y[:, np.newaxis] * selected_X, y, sample_weight=weights)
            else:
                clf.fit(y[:, np.newaxis] * selected_X, y)
        # Check if there is a ConvergenceWarning
        if any(isinstance(warn.message, ConvergenceWarning) for warn in w):
            # Handle the situation when ConvergenceWarning occurs
            print("SVM did not converge!")
            convergence_warning = True
        else:
            # No ConvergenceWarning occurred
            convergence_warning = False
    if non_neg:
        return np.clip(clf.coef_[0], 0, np.inf), convergence_warning
    return clf.coef_[0], convergence_warning


def weighted_lr(
    X: np.ndarray,
    qid: np.ndarray,
    indices: List[int] = None,
    non_neg: bool = True,
    weighted: bool = True,
    penalty: Literal["l1", "l2", "elasticnet"] | None = "l1",
    loss: str = None,  # pylint: disable=unused-argument
    dual: bool = False,
    tol: float = 0.0001,
    C: float = 1e2,
    fit_intercept: bool = False,
    debug: bool = False,  # pylint: disable=unused-argument
    relaxation: bool = False,  # pylint: disable=unused-argument
    mirror: bool = True,
    **LRkwargs,
) -> None:
    """
    Weighted Logistic Regression (LR) (similar to SVC, but different options available)

    If multiple indices from the same problem are provided, the sample weight is
    adjusted to compensate. This LR method requires two classes centered around
    the origin, so we introduce a mirrored second class (this duplicates the
    data and requires C to be halved to correct for the now doubled loss).
    Non-negative support is not present, so any negative weights are set to zero
    if non-negative weights are desired.

    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param indices:  selected data indices
    :param non_neg:  set negative weights in `w` to zero if `True`; rudimentary clipping
    :param weighted:  weight selected data based on cluster participation
    :param penalty:  regularization penalty parameter
    :param dual:  solve the dual problem or not; default is to solve the primal problem
    :param tol:  numerical tolerance for solution
    :param C:  regularization parameter (large `C` means regularization is not as important)
    :param fit_intercept:  the default LR will try to find an intercept
        parameter; this is zero by design for relative data, and will be
        approximately zero due to data mirroring if fitting the intercept is
        allowed
    :param mirror:  mirror the data (introduce negative copies of data points)
        or introduce a negative class for some of the data
    :param **LRkwargs:  extra keyword args for the LR (e.g., `max_iter`, `verbose`)
    :return:  weight vector for the centered LR
    """
    if indices is None:
        indices = np.arange(X.shape[0], dtype=int)
        # if debug:
        #     print('    No indices provided. Including all data.')
    if weighted:
        weights = get_data_weight(qid, indices=indices)
        # if debug:
        #     print('    Computed weights for weighted approach.')

    num_clusters = len(np.unique(qid))

    solver = "liblinear" if penalty == "l1" else "lbfgs"

    clf = LogisticRegression(
        penalty=penalty,
        dual=dual,
        tol=tol,
        C=C / 2 / num_clusters,
        # C = C/2,
        fit_intercept=fit_intercept,
        solver=solver,
        **LRkwargs,
    )
    selected_X = X[indices]
    with warnings.catch_warnings(record=True) as w:  # Catch convergence warnings
        if mirror:  # Duplicate data into a second negative class
            X_mirror = np.concatenate((selected_X, -selected_X), axis=0)
            y = np.ones((selected_X.shape[0],))
            y_mirror = np.concatenate((y, -y), axis=0)
            # if debug:
            #     print(f'    Mirrored data with shape {X_mirror.shape}. '+\
            #           f'False classes with shape {y_mirror.shape}.')
            if weighted:
                weights_mirror = np.concatenate((weights, weights), axis=0)
                clf.fit(X_mirror, y_mirror, sample_weight=weights_mirror)
            else:
                clf.fit(X_mirror, y_mirror)
        else:  # Introduce a second negative class by swapping some data to the other class
            # y = np.random.choice([-1,1], size = (selected_X.shape[0],))
            y = np.ones((selected_X.shape[0],))
            y[: len(y) // 2] = -1
            if weighted:
                clf.fit(y[:, np.newaxis] * selected_X, y, sample_weight=weights)
            else:
                clf.fit(y[:, np.newaxis] * selected_X, y)
        # Check if there is a ConvergenceWarning
        if any(isinstance(warn.message, ConvergenceWarning) for warn in w):
            # Handle the situation when ConvergenceWarning occurs
            print("SVM did not converge!")
            convergence_warning = True
        else:
            # No ConvergenceWarning occurred
            convergence_warning = False
    if non_neg:
        # if debug:
        #     print('    Clipping negative weights.')
        return np.clip(clf.coef_[0], 0, np.inf), convergence_warning
    return clf.coef_[0], convergence_warning


def LP_method(
    X: np.ndarray,
    qid: np.ndarray,
    indices: List[int] = None,
    non_neg: bool = True,
    penalty: Literal["l1", "l2"] = "l1",
    loss: Literal["squared_hinge", "hinge"] = "squared_hinge",
    C: float = 1e2,
    clustered: bool = True,
    verbose: bool = False,
    debug: bool = False,  # pylint: disable=unused-argument
    relaxation: bool = True,  # pylint: disable=unused-argument
    **kwargs,  # pylint: disable=unused-argument
) -> np.ndarray:
    """
    Clustered single-class SVC for relative (centered) data

    .. math::
        \\min_w            \\|\\mathbf{w}\\|_1 + C\\sum_{k=1}^{K} \\zeta_k\\
        \\text{subject to} 0 \\le \\xi_i \\le \\zeta_{k_i}\\
                           1 - \\mathbf{w}^\\top \\mathbf{x}_i <= \\xi_i
                           0 <= w

    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param indices:  selected data indices
    :param non_neg:  set negative weights in w to zero if `True`
    :param penalty:  regularization penalty parameter
    :param loss:  slack error term (hinge is :math:`\\max(0, 1 - y_i(w^\\top x_i + b))`;
        `squared_hinge` is `(hinge)**2`)
    :param C:  regularization parameter (large `C` means regularization is not as important)
    :param clustered:  should the loss be applied to the cluster or the individual data points
    :param verbose:  print more information to standard out
    :param debug: not used
    :param relaxation: not used
    :param **kwargs: not used, catches potentially unwanted kwargs
    :return:  weight vector for the centered SVC

    TODO - warn on kwargs?
    """
    if indices is None:
        indices = np.arange(X.shape[0], dtype=int)
    selected_X = X[indices]
    selected_I = qid[indices]
    num_features = selected_X.shape[1]
    num_data = selected_X.shape[0]
    cluster_ids, updated_I = np.unique(selected_I, return_inverse=True)
    num_clusters = len(cluster_ids)
    w = cp.Variable(num_features)
    xi_i = cp.Variable(num_data)
    zeta_k = cp.Variable(num_clusters)

    # Define regularization
    if penalty == "l1":
        reg = cp.norm(w, 1)
    elif penalty == "l2":
        reg = cp.norm(w, 2)
    else:
        reg = cp.norm(w, penalty)

    # Define the constraints
    constraints = [1 - (selected_X @ w) <= xi_i, 0 <= xi_i, 0 <= zeta_k]
    if non_neg:
        constraints.append(0 <= w)

    # Add constraints on zeta values for each cluster
    for i, k in zip(range(num_data), updated_I[list(range(num_data))]):
        constraints.append(xi_i[i] <= zeta_k[k])
        # If there is only one data point per cluster, zeta_{k_i} = xi_i

    # Define the loss
    if clustered:
        if loss == "squared_hinge":
            loss_fun = cp.sum(cp.square(zeta_k))
        elif loss == "hinge":
            loss_fun = cp.sum(zeta_k)
        else:
            loss_fun = cp.sum(zeta_k)
    else:
        if loss == "squared_hinge":
            loss_fun = cp.sum(cp.square(xi_i))
        elif loss == "hinge":
            loss_fun = cp.sum(xi_i)
        else:
            loss_fun = cp.sum(xi_i)

    lamb = 1 / C  # lambda parameter

    # equivalent C in SVC is C/num_clusters
    prob = cp.Problem(
        cp.Minimize(loss_fun / num_clusters + lamb * reg), constraints=constraints
    )

    prob.solve(verbose=verbose, solver="CLARABEL")
    return w.value


def MILP_method(
    X: np.ndarray,
    qid: np.ndarray,
    indices: List[int] = None,
    non_neg: bool = True,
    relaxation: bool = True,
    debug: bool = False,
    verbose: bool = False,
    eps_min: float = 1e-5,
    eps_max: float = 0.99,
    **kwargs,  # pylint: disable=unused-argument
) -> np.ndarray:
    """
    Clustered single-class MILP for relative (centered) data

    .. math::
        \\min_w            -\\varepsilon + \\sum_{k=1}^{K} z_k\\
        \\text{subject to} 0 \\le y_i \\le z_{k_i}\\
                           \\varepsilon - \\mathbf{w}^\\top \\mathbf{x}_i <= y_i (1 + \\varepsilon)
                           0 <= w
                           \\|w\\|_1 = 1

    Here, the value of epsilon is the margin of the SVC (when written in a standard form, epsilon).
    The epsilon parameter is a margin width and is :math:`0 < \\varepsilon < 1` such that the margin
    will never be larger than another solved problem.

    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param indices:  selected data indices
    :param non_neg:  set negative weights in w to zero if `True`
    :param relaxation:  use the linear relaxation (`True`) or binary variables (`False`);
        cvxpy cannot solve with binary variables so currently always solves as if this is `True`
    :param debug:  print more information to standard output
    :param verbose:  print additional solving information
    :param eps_min:  minimum value of epsilon (should be greater than 0)
    :param eps_max:  maximum value of epsilon (should be less than 1)
    :return:  weight vector for the centered SVC
    """
    if indices is None:
        indices = np.arange(X.shape[0], dtype=int)
    selected_X = X[indices]
    selected_qid = qid[indices]
    num_features = selected_X.shape[1]
    num_data = selected_X.shape[0]
    cluster_ids, updated_I = np.unique(selected_qid, return_inverse=True)
    num_clusters = len(cluster_ids)
    w = cp.Variable(num_features)
    y_i = cp.Variable(num_data, boolean=not relaxation)
    z_k = cp.Variable(num_clusters, boolean=not relaxation)
    epsilon = cp.Variable()
    eta_i = cp.Variable()  # Define eta_i as epsilon*y_i

    # # Define the constraints
    # constraints = [
    #     # epsilon - (selected_X @ w) <= y_i * (1 + epsilon),
    #     # epsilon - (selected_X @ w) <= y_i,
    #     (selected_X @ w) >= epsilon - y_i,
    #     0 <= y_i,
    #     0 <= z_k,
    #     eps_min <= epsilon,
    #     epsilon <= eps_max,
    # ]
    # Define constraints using eta_i and convex hull reformulation
    constraints = [
        selected_X @ w >= epsilon - y_i - eta_i,
        0 <= y_i,
        0 <= z_k,
        -1 + y_i <= eta_i - epsilon,
        eta_i - epsilon <= 1 - y_i,
        0 <= eta_i,
        eta_i <= y_i,
        eps_min <= epsilon,
        epsilon <= eps_max,
    ]

    if non_neg:  # enforce the 1-norm
        constraints.append(0 <= w)
        constraints.append(cp.sum(w) == 1)
    else:  # enforce the 1 norm if w is allowed to be non-negative
        w_plus = cp.Variable(num_features)
        w_minus = cp.Variable(num_features)
        constraints.append(w_plus - w_minus == w)
        constraints.append(0 <= w_plus)
        constraints.append(0 <= w_minus)
        constraints.append(cp.sum(w_plus + w_minus) == 1)

    # Add constraints on zeta values for each cluster
    for i, k in zip(range(num_data), updated_I[list(range(num_data))]):
        constraints.append(y_i[i] <= z_k[k])
        # If there is only one data point per cluster, z_{k_i} = y_i

    # Define the loss
    loss = cp.sum(z_k)

    # equivalent C in SVC is C/num_clusters
    prob = cp.Problem(cp.Minimize(loss - epsilon), constraints=constraints)
    if relaxation:
        solver_name = "CLARABEL"
    else:
        solver_name = "SCIPY"
    prob.solve(verbose=verbose, solver=solver_name)
    if debug:
        print(f"  Found solution with epsilon (margin half-width) {epsilon.value}")
    return w.value


class csvm(LinearSVC):
    """Class for the clustered SVC"""

    def __init__(
        self,
        penalty: Literal["l1", "l2"] = "l2",
        loss: Literal["squared_hinge", "hinge"] = "squared_hinge",
        dual: bool = True,
        tol: float = 0.0001,
        C: float = 1e3,
        max_iter: int = 1000,
        sol_method: Callable | str = weighted_lr,
    ):
        super().__init__(penalty=penalty, loss=loss, tol=tol, C=C, max_iter=max_iter)
        self.C = C
        self.trained = False
        self.w = None
        self.sol_method = sol_method

    def fit(
        self, R: np.ndarray, qid: np.ndarray, **kwargs
    ):  # pylint: disable=arguments-differ
        """Fit the parameters of the clustered classifier

        :param R:  residual data
        :type R:  np.ndarray
        :param qid:  query ids
        :type qid:  np.ndarray
        """
        self.w = column_generation(
            R,
            qid,
            sol_method=self.sol_method,
            loss=self.loss,
            penalty=self.penalty,
            **kwargs,
        )
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predicted objective values using the fit weights

        :param X:  feature data (not necessarily relative)
        :type X:  np.ndarray
        :raises ValueError:  if not fitted
        :return:  predicted objective values
        :rtype:  np.ndarray
        """
        if not self.trained:
            raise ValueError("Not trained.")
        return np.dot(X, self.w)


def column_generation(
    X: np.ndarray,
    qid: np.ndarray,
    w0: np.ndarray = None,
    indices: List[int] = None,
    C: float = 1e2,
    sol_method: Callable | str = weighted_lr,
    initial_selection_method: Callable = select_indices,
    subsequent_selection_method: Callable = select_indices_change_sign,
    allow_duplicates: bool = False,
    non_neg: bool = True,
    debug: bool = False,
    return_indices: bool = False,
    normalize_w: bool = False,
    return_report: bool = False,
    validation_X: np.ndarray = None,
    validation_qid: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    """
    Iteratively apply the selected solution method, selecting more data as the
    algorithm proceeds

    The structure of the problem allows solutions to be found with a fraction of
    the data, as little as a single data point per cluster

    ```
    # Initial data:
    x =[[x1], [x2], [x3]], x_i, i in (1, 2, 3)
    qid =[[ 1], [ 2], [ 3]], I_i = k, k in (1, 2, 3)
    # Master solve.
    # Subproblem:
    # Add new data to x and qid
    x =[[x1], [x2], [x3], [x4]], x_i, i in (1, 2, 3, 4)  # updated i
    qid =[[ 1], [ 2], [ 3], [ 2]], I_i = k, k in (1, 2, 3)  # k remains the same
    # Rebuild master with new constraints (MIP) or new weighted data (LR/SVC):
    y_i <= z_[qid[i]]
    # qid is an "index set"
    ```

    :param X:  input data matrix `(M, N)`
    :param qid:  problem labels/query-ids `(M,)`
    :param w0:  initial weight guess
    :param indices:  initial data usage indices
    :param sol_method:  solution method (e.g., "lp" or `LP_method`, "svc" or
        `weighted_svc`, "milp" or `MILP_method`, "lr" or `weighted_lr`)
    :param initial_selection_method:  method for selecting initial data indices
    :param subsequent_selection_method:  method for selecting data indices to add
        to the problem after a solve
    :param allow_duplicates:  allow indices to be selected multiple times (eventually
        biases data indices by true importance, but grows the size of the problem quickly)
    :param non_neg:  should the weights be non negative (or allow negative weights if `False`)
    :param debug:  print more information
    :param return_indices:  return the indices of the data used
    :param normalize_w:  apply a l1 norm to the weight vector `w`
    :param return_report:  return a dictionary with solution information, performance
        history, weights, and indices
    :param validation_X:  validation data
    :param validation_qid:  validation data qid's
    :param **kwargs:  keyword arguments for the solution method(s)
    :return:  weight vector for the hyperplane, or weight vector and data indices,
        or a dictionary report of the solution
    """
    start_time = datetime.datetime.now()
    assert X.shape[0] == qid.shape[0]

    validation = False
    if (validation_qid is not None) and (validation_X is not None):
        validation = True

    if isinstance(sol_method, str):
        sol_method = sol_method.lower()
        if sol_method.lower() == "lp":
            sol_method = LP_method
        elif sol_method.lower() in ["milp", "mip"]:
            sol_method = MILP_method
        elif sol_method.lower() in ["svc", "svm"]:
            sol_method = weighted_svc
        elif sol_method.lower() == "lr":
            sol_method = weighted_lr
        else:
            raise ValueError(f"Unknown solution method: {sol_method}")

    if sol_method is MILP_method:
        subsequent_selection_method = select_indices_neg

    if w0 is None:
        if debug:
            print("Creating initial estimate for w")
        w0 = solve_estimate(
            X,
            qid,
            initial_selection_method=initial_selection_method,
            subsequent_selection_method=subsequent_selection_method,
            allow_duplicates=allow_duplicates,
            non_neg=non_neg,
            debug=debug,
        )

    if debug or return_report:
        num_solved, num_unsolved = check_solutions(w0, X, qid)

    solved_history = []
    unsolved_history = []
    validation_solved_history = []
    validation_unsolved_history = []
    validation_num_solved, validation_num_unsolved = 0, 0

    if return_report:
        solved_history.append(num_solved)
        unsolved_history.append(num_unsolved)
        if validation:
            validation_num_solved, validation_num_unsolved = check_solutions(
                w0, validation_X, validation_qid
            )
            validation_solved_history.append(validation_num_solved)
            validation_unsolved_history.append(validation_num_unsolved)

    if debug:
        print(
            f"Initial estimate for w solves {num_solved} of "
            + f"{num_solved + num_unsolved} problems "
            + f"({num_solved/(num_solved + num_unsolved)*100:5.2f}%)"
        )

    if indices is None:
        indices = initial_selection_method(w0, X, qid)
    num_data = 0
    num_iters = 0
    active_history = [len(indices)]
    w_prev = w0
    convergence_warning = False

    while len(indices) > num_data:  # repeat until the indices don't change (solved)
        num_iters += 1
        num_data = len(indices)
        # solve master using current indices
        w = sol_method(
            X, qid, indices=indices, debug=debug, non_neg=non_neg, C=C, **kwargs
        )
        if isinstance(w, tuple):
            w, convergence_warning = w[0], (w[1] or convergence_warning)
        if non_neg:
            w = np.clip(w, 0, np.inf)
        if normalize_w:
            norm_w = np.linalg.norm(w, 1, keepdims=True)
            if norm_w > 0:
                w /= norm_w
        new_indices = subsequent_selection_method(
            w, X, qid, w_prev=w_prev
        )  # subproblem
        w_prev = w

        # combine the old indices and the new_indices using a set union
        indices = list(set(indices) | set(new_indices))
        active_history.append(len(indices))
        if debug or return_report:
            num_solved, num_unsolved = check_solutions(w, X, qid)
            solved_history.append(num_solved)
            unsolved_history.append(num_unsolved)
            if validation:
                validation_num_solved, validation_num_unsolved = check_solutions(
                    w, validation_X, validation_qid
                )
                validation_solved_history.append(validation_num_solved)
                validation_unsolved_history.append(validation_num_unsolved)
        if debug:
            print(
                f"  Master: {num_data:5d} of {X.shape[0]:5d} data "
                + f"({num_data/X.shape[0]*100:5.2f}%); "
                + f"solves {num_solved:5d} of {num_solved + num_unsolved:5d} problems "
                + f"({num_solved/(num_solved + num_unsolved)*100:5.2f}%); "
                + f"found {len(indices) - num_data:5d} data"
            )
    if debug:
        print("  --No new data to add. Solution is stable.")
    if return_report:
        return {
            "w": w,
            "indices": indices,
            "num_problems": len(np.unique(qid)),
            "active": len(indices),
            "num_data": X.shape[0],
            "active_history": active_history,
            "solved": solved_history[-1],
            "solved_history": solved_history,
            "validation_solved": validation_num_solved,
            "validation_solved_history": validation_solved_history,
            "unsolved": unsolved_history[-1],
            "unsolved_history": unsolved_history,
            "validation_unsolved": validation_num_unsolved,
            "validation_unsolved_history": validation_unsolved_history,
            "num_iters": len(active_history),
            "sol_method": sol_method.__name__,
            "C": C,
            "non_neg": non_neg,
            "initial_selection_method": initial_selection_method.__name__,
            "subsequent_selection_method": subsequent_selection_method.__name__,
            "start_time": start_time,
            "end_time": datetime.datetime.now(),
            "convergence_warning": convergence_warning,
        }
    if return_indices:
        return w, indices
    return w
