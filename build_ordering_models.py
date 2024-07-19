"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Script to build and save models
"""

# import sys

import numpy as np
import xgboost as xgb
from sklearn import preprocessing

import greto.file_io as gr_file_io
import greto.models as models
from greto import cluster_svm as csvm
from greto import fom_optimization_data as fo

TRAIN_XGB = False

print("Loading events")
events, clusters = gr_file_io.load_m30()

N = 2000  # number of events to include in training
width = 5
seed = 42  # RNG seed for reproducibility
C = 1000  # Sparsity parameter for linear models

print("Creating training data")

training_events, training_clusters = events[:N], clusters[:N]
X, Y = fo.create_data(
    training_events, training_clusters, semi_greedy_width=width, seed=seed
)

for column_set_name, columns in fo.column_sets.items():
    print(f"Beginning column set: {column_set_name}")
    print("Scaling data")
    scaler = preprocessing.StandardScaler(with_mean=False)  # z = (x - mean_) / scale_
    scaler.fit(X[columns])

    print("Making residuals")
    r, qid = fo.make_residuals(
        scaler.transform(X[columns]),
        # X[columns].to_numpy(),
        np.array(Y["cluster_ids"].astype(int)),
        np.array(Y["opt_index"].astype(int)),
        np.array(Y["other_index"].astype(int)),
    )

    # # We can scale X or r; in general, this doesn't really matter as we separate
    # # the scaling and the weights at the end. However, it is incorrect to shift values by
    # # their mean for r scaling (no impact for X scaling)
    # scaler = preprocessing.StandardScaler(with_mean=False)  # z = (x - mean_) / scale_
    # scaler.fit(r)
    # r = scaler.transform(r)

    # Force weights to be non-negative or not (allow negative weights)
    for non_neg in [True, False]:
        w0 = csvm.solve_estimate(
            r,
            qid,
            initial_selection_method=csvm.select_indices,
            subsequent_selection_method=csvm.select_indices_neg,
            allow_duplicates=False,
            non_neg=non_neg,
            debug=True,
        )

        # Loop over solution methods
        for sol_method in ["lr", "svm", "lp", "milp"]:
            # for sol_method in []:

            report = csvm.column_generation(
                r,
                qid,
                w0=w0,
                sol_method=sol_method,
                C=C,
                initial_selection_method=csvm.select_indices,
                subsequent_selection_method=csvm.select_indices_neg,
                non_neg=non_neg,
                return_report=True,
                debug=True,
                mirror=False,
            )

            scaled_w = np.array(report["w"] / scaler.scale_)  # move scale from X to w
            w = np.array(report["w"])  # weights on scaled (dimensionless) features
            weights = list(w[np.abs(w) > 0.0])  # eliminate zero weight features
            scaled_weights = list(scaled_w[np.abs(scaled_w) > 0.0])
            column_names = list(np.array(columns)[np.abs(w) > 0.0])
            scales = list(scaler.scale_[np.abs(w) > 0.0])

            fname = (
                f"models/ordering/N{N}_{sol_method}_nonneg{non_neg}_C{C}"
                + f"_cols-{column_set_name}_width{width}.json"
            )

            models.save_linear_model(
                weights,
                scale=scales,
                columns=column_names,
                scaled_weights=scaled_weights,
                filename=fname,
            )

            solved, unsolved = csvm.check_solutions_general(
                np.dot(scaler.transform(X[columns]), w),
                Y["cluster_ids"],
                Y["ordered"],
            )

            print(f"training acc = {solved / (solved + unsolved)}")

            report.pop("indices")
            print(report)

    if TRAIN_XGB:
        # Train a boosted tree ranker using XGBoost
        sol_method = "XGBRanker_pairwise"
        print("Training XGB ranker")

        max_depth = 5
        C_xgb = 10

        ranker = xgb.XGBRanker(
            tree_method="hist",
            n_estimators=100,
            objective="rank:pairwise",
            reg_alpha=C_xgb,
            max_depth=max_depth,
        )

        ranker.fit(
            X[columns],
            1 - Y["ordered"],
            qid=Y["cluster_ids"],
        )

        ranker.get_booster().feature_names = columns
        solved, unsolved = csvm.check_solutions_general(
            ranker.predict(X[columns]),
            Y["cluster_ids"],
            Y["ordered"],
        )

        print(f"training acc = {solved / (solved + unsolved)}")

        ranker_fname = f"models/ordering/N{N}_{sol_method}_C{C_xgb}_cols-{column_set_name}_width{width}.ubj"

        models.save_xgb_model(ranker, ranker_fname)
