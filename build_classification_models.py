"""Build classification models"""

import pickle as pkl

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression

from greto import fast_features as ff
from greto import file_io
from greto import fom_optimization_data as fo
from greto import models
from greto.models import sns_model

# Load multiplicity 30 data
print("Loading data")
events, clusters = file_io.load_m30()  # pylint: disable=unbalanced-tuple-unpacking

print("Gathering features of true-ordered events")
X, Y = fo.create_classification_data(events[:], clusters[:], use_true=True)

print("Getting test/train indices for data")
train_indices, test_indices = sns_model.split(X)


order_model_filename = "models/ordering/N2000_lp_nonnegFalse_C10000_cols-oft_fast_tango_width5.json"
print(f"Load ordering model: {order_model_filename}")
model = models.load_order_FOM_model(
    order_model_filename
)
print("Ordering true clusters and gathering features of model-ordered events")
X_model, Y_model = fo.create_classification_data(
    events[:], clusters[:], model=model, width=5, stride=3
)

alpha_degrees = 13.0
print(f"Re-clustering (alpha = {alpha_degrees} degrees) and model-ordering events to gather features")
X_re, Y_re = fo.create_classification_data_with_clustering(
    events[:],
    clusters[:],
    model=model,
    width=5,
    stride=3,
    alpha_degrees=alpha_degrees,
)

print("Getting test/train indices for reclustered data")
train_indices_re, test_indices_re = sns_model.split(X_re)

# Logistic Regression
## True data
### 7_500 samples trained
print("Training LogisticRegression based model using true order (training only)")
classification_model = sns_model(
    use_combiner=True,
    model_class=LogisticRegression,
    pca_n_components=0.95,
    columns=ff.all_feature_names,
)
classification_model.fit(
    X.to_numpy()[train_indices],
    1 - Y["complete"].to_numpy()[train_indices],
    Y["lengths"].to_numpy()[train_indices] <= 1,
)

with open(
    "models/suppression/N7500_sns-logistic_pca-0.95_order-true.pkl", "wb"
) as file:
    pkl.dump(classification_model, file)

### 10_000 sampled trained
print("Training LogisticRegression based model using true order (testing and training)")
classification_model = sns_model(
    use_combiner=True,
    model_class=LogisticRegression,
    pca_n_components=0.95,
    columns=ff.all_feature_names,
)
classification_model.fit(
    X.to_numpy(), 1 - Y["complete"].to_numpy(), Y["lengths"].to_numpy() <= 1
)

with open(
    "models/suppression/N10000_sns-logistic_pca-0.95_order-true.pkl", "wb"
) as file:
    pkl.dump(classification_model, file)

## Model ordered data
### 7_500 samples trained
print("Training LogisticRegression based model using model order (training only)")
classification_model = sns_model(
    use_combiner=True,
    model_class=LogisticRegression,
    pca_n_components=0.95,
    columns=ff.all_feature_names,
)
classification_model.fit(
    X_model.to_numpy()[train_indices],
    1 - Y_model["complete"].to_numpy()[train_indices],
    Y_model["lengths"].to_numpy()[train_indices] <= 1,
)

with open(
    "models/suppression/N7500_sns-logistic_pca-0.95_order-model.pkl", "wb"
) as file:
    pkl.dump(classification_model, file)

### 10_000 sampled trained
print("Training LogisticRegression based model using true order (testing and training)")
classification_model = sns_model(
    use_combiner=True,
    model_class=LogisticRegression,
    pca_n_components=0.95,
    columns=ff.all_feature_names,
)
classification_model.fit(
    X_model.to_numpy(),
    1 - Y_model["complete"].to_numpy(),
    Y_model["lengths"].to_numpy() <= 1,
)

with open(
    "models/suppression/N10000_sns-logistic_pca-0.95_order-model.pkl", "wb"
) as file:
    pkl.dump(classification_model, file)

## Model ordered reclustered data
### 7_500 samples trained
print("Training LogisticRegression based model using reclustered model order (training only)")
classification_model = sns_model(
    use_combiner=True,
    model_class=LogisticRegression,
    pca_n_components=0.95,
    columns=ff.all_feature_names,
)
classification_model.fit(
    X_re.to_numpy()[train_indices_re],
    1 - Y_re["complete"].to_numpy()[train_indices_re],
    Y_re["lengths"].to_numpy()[train_indices_re] <= 1,
)

with open(
    "models/suppression/N7500_sns-logistic_pca-0.95_order-model_reclustered.pkl", "wb"
) as file:
    pkl.dump(classification_model, file)

### 10_000 samples trained
print("Training LogisticRegression based model using reclustered model order (testing and training)")
classification_model = sns_model(
    use_combiner=True,
    model_class=LogisticRegression,
    pca_n_components=0.95,
    columns=ff.all_feature_names,
)
classification_model.fit(
    X_re.to_numpy(), 1 - Y_re["complete"].to_numpy(), Y_re["lengths"].to_numpy() <= 1
)

with open(
    "models/suppression/N10000_sns-logistic_pca-0.95_order-model_reclustered.pkl", "wb"
) as file:
    pkl.dump(classification_model, file)


# Linear Regression
## True data
### 7_500 samples trained
print("Training LinearRegression based model using true order (training only)")
classification_model = sns_model(
    use_combiner=True,
    model_class=LinearRegression,
    pca_n_components=0.95,
    columns=ff.all_feature_names,
)
classification_model.fit(
    X.to_numpy()[train_indices],
    1 - Y["complete"].to_numpy()[train_indices],
    Y["lengths"].to_numpy()[train_indices] <= 1,
)

with open("models/suppression/N7500_sns-linear_pca-0.95_order-true.pkl", "wb") as file:
    pkl.dump(classification_model, file)

### 10_000 sampled trained
print("Training LinearRegression based model using true order (testing and training)")
classification_model = sns_model(
    use_combiner=True,
    model_class=LinearRegression,
    pca_n_components=0.95,
    columns=ff.all_feature_names,
)
classification_model.fit(
    X.to_numpy(), 1 - Y["complete"].to_numpy(), Y["lengths"].to_numpy() <= 1
)

with open("models/suppression/N10000_sns-linear_pca-0.95_order-true.pkl", "wb") as file:
    pkl.dump(classification_model, file)

## Model ordered data
### 7_500 samples trained
print("Training LinearRegression based model using model order (training only)")
classification_model = sns_model(
    use_combiner=True,
    model_class=LinearRegression,
    pca_n_components=0.95,
    columns=ff.all_feature_names,
)
classification_model.fit(
    X_model.to_numpy()[train_indices],
    1 - Y_model["complete"].to_numpy()[train_indices],
    Y_model["lengths"].to_numpy()[train_indices] <= 1,
)

with open("models/suppression/N7500_sns-linear_pca-0.95_order-model.pkl", "wb") as file:
    pkl.dump(classification_model, file)

### 10_000 sampled trained
print("Training LinearRegression based model using model order (testing and training)")
classification_model = sns_model(
    use_combiner=True,
    model_class=LinearRegression,
    pca_n_components=0.95,
    columns=ff.all_feature_names,
)
classification_model.fit(
    X_model.to_numpy(),
    1 - Y_model["complete"].to_numpy(),
    Y_model["lengths"].to_numpy() <= 1,
)

with open(
    "models/suppression/N10000_sns-linear_pca-0.95_order-model.pkl", "wb"
) as file:
    pkl.dump(classification_model, file)

## Model ordered reclustered data
### 7_500 samples trained
print("Training LinearRegression based model using reclustered model order (training only)")
classification_model = sns_model(
    use_combiner=True,
    model_class=LinearRegression,
    pca_n_components=0.95,
    columns=ff.all_feature_names,
)
classification_model.fit(
    X_re.to_numpy()[train_indices_re],
    1 - Y_re["complete"].to_numpy()[train_indices_re],
    Y_re["lengths"].to_numpy()[train_indices_re] <= 1,
)

with open(
    "models/suppression/N7500_sns-linear_pca-0.95_order-model_reclustered.pkl", "wb"
) as file:
    pkl.dump(classification_model, file)

### 10_000 samples trained
print("Training LinearRegression based model using reclustered model order (testing and training)")
classification_model = sns_model(
    use_combiner=True,
    model_class=LinearRegression,
    pca_n_components=0.95,
    columns=ff.all_feature_names,
)
classification_model.fit(
    X_re.to_numpy(), 1 - Y_re["complete"].to_numpy(), Y_re["lengths"].to_numpy() <= 1
)

with open(
    "models/suppression/N10000_sns-linear_pca-0.95_order-model_reclustered.pkl", "wb"
) as file:
    pkl.dump(classification_model, file)

# XGB Classifiers
## Drop columns that allow energy "cheating". Model will simply select the peak
## energies or any column that scales (nearly) monotonically with energy
print("Training XGBoost models")
drop_columns = [
    "linear_attenuation_cm-1",
    "energy",
    "-log_p_abs_max",
    "cross_abs_min",
    "p_compt_max",
    "cross_compt_min",
    "cross_compt_min_nonfinal",
    "-log_p_compt_min_nonfinal",
    "-log_p_compt_min",
    "-log_p_abs_max_tango",
    "p_compt_min_nonfinal",
    "p_compt_max_tango",
    "p_abs_min",
    "cross_abs_min_tango",
    "-log_p_compt_min_nonfinal_tango",
    "-log_p_compt_min_tango",
    "p_compt_mean_nonfinal",
    "-log_p_compt_sum_nonfinal",
    "cross_total_min",
    "cross_compt_min_tango",
    "-log_p_compt_mean_nonfinal",
    "cross_compt_min_nonfinal_tango",
    "cross_compt_mean_nonfinal",
    "cross_compt_sum_nonfinal",
    "p_compt_sum_nonfinal",
    "cross_total_min_tango",
    "p_compt_min_nonfinal_tango",
    "p_compt_mean_nonfinal_tango",
    "-log_p_compt_sum_nonfinal_tango",
    "p_abs_min_tango",
    "-log_p_compt_mean_nonfinal_tango",
    "cross_abs_dist_max",
    "cross_compt_mean_nonfinal_tango",
    "p_compt_sum_nonfinal_tango",
    "cross_compt_sum_nonfinal_tango",
    "cross_abs_dist_max_tango",
    "cross_abs_dist_min",
    "cross_abs_ge_dist_min",
]

print(f"Dropping columns too similar to energy:\n{drop_columns}")
X_dropped = X.drop(columns=drop_columns, inplace=False)
X_model_dropped = X_model.drop(columns=drop_columns, inplace=False)
X_re_dropped = X_re.drop(columns=drop_columns, inplace=False)

classifier = xgb.XGBClassifier()
classifier.fit(
    np.clip(X_dropped.to_numpy()[train_indices], 0, 1e16),
    1 - Y["complete"].to_numpy()[train_indices] < 0.5,
)
classifier.get_booster().feature_names = list(X_dropped.columns)
classifier.save_model("models/suppression/N7500_XGBClassifier_order-true.ubj")

classifier = xgb.XGBClassifier()
classifier.fit(
    np.clip(X_dropped.to_numpy(), 0, 1e16), 1 - Y["complete"].to_numpy() < 0.5
)
classifier.get_booster().feature_names = list(X_dropped.columns)
classifier.save_model("models/suppression/N10000_XGBClassifier_order-true.ubj")

classifier = xgb.XGBClassifier()
classifier.fit(
    np.clip(X_model_dropped.to_numpy()[train_indices], 0, 1e16),
    1 - Y_model["complete"].to_numpy()[train_indices] < 0.5,
)
classifier.get_booster().feature_names = list(X_model_dropped.columns)
classifier.save_model("models/suppression/N7500_XGBClassifier_order-model.ubj")

classifier = xgb.XGBClassifier()
classifier.fit(
    np.clip(X_model_dropped.to_numpy(), 0, 1e16),
    1 - Y_model["complete"].to_numpy() < 0.5,
)
classifier.get_booster().feature_names = list(X_model_dropped.columns)
classifier.save_model("models/suppression/N10000_XGBClassifier_order-model.ubj")

classifier = xgb.XGBClassifier()
classifier.fit(
    np.clip(X_re_dropped.to_numpy()[train_indices_re], 0, 1e16),
    1 - Y_re["complete"].to_numpy()[train_indices_re] < 0.5,
)
classifier.get_booster().feature_names = list(X_re_dropped.columns)
classifier.save_model(
    "models/suppression/N7500_XGBClassifier_order-model_reclustered.ubj"
)

classifier = xgb.XGBClassifier()
classifier.fit(
    np.clip(X_re_dropped.to_numpy(), 0, 1e16), 1 - Y_re["complete"].to_numpy() < 0.5
)
classifier.get_booster().feature_names = list(X_re_dropped.columns)
classifier.save_model(
    "models/suppression/N10000_XGBClassifier_order-model_reclustered.ubj"
)

print("Complete")
