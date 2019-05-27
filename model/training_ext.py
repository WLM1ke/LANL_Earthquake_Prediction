"""Обучение модели - ExtraTreesRegressor."""
import logging
import time

from sklearn import ensemble
from sklearn import metrics
import pandas as pd
import numpy as np

from model import conf
from model import processing
from model.conf import K_FOLDS
from model.conf import SEED

LOGGER = logging.getLogger(__name__)

ITERATIONS = 100

CLF_PARAMS = dict(
    n_estimators=ITERATIONS,
    criterion="mae",
    n_jobs=-1,
    random_state=SEED,
    verbose=10
)

DROP = [
    "num_peaks_10", "percentile_roll_std_5", "afc_50"
]


def train_ext():
    """Обучение LightGBM."""
    x_train, y_train = processing.train_set()
    x_test = processing.test_set()

    x_train.drop(DROP, axis=1, inplace=True)
    x_test.drop(DROP, axis=1, inplace=True)

    y_oof = pd.Series(0, index=x_train.index, name="oof_ext")
    y_pred = pd.Series(0, index=x_test.index, name="time_to_failure")
    feat_importance = 0
    scores = []

    for index_train, index_valid in K_FOLDS.split(x_train):

        clf = ensemble.ExtraTreesRegressor(**CLF_PARAMS)
        clf.fit(x_train.iloc[index_train], y_train.iloc[index_train])

        y_oof.iloc[index_valid] = clf.predict(x_train.iloc[index_valid])

        scores.append(metrics.mean_absolute_error(y_train.iloc[index_valid], y_oof.iloc[index_valid]))

        y_pred += clf.predict(x_test) / K_FOLDS.get_n_splits()

        feat_importance += clf.feature_importances_ / K_FOLDS.get_n_splits()
        print("\n")

    LOGGER.info(f"MAE на кроссвалидации: " + str(np.round(sorted(scores), 5)))
    LOGGER.info(f"MAE среднее: {np.mean(scores):0.3f} +/- {np.std(scores):0.3f}")

    stamp = (
        f"{time.strftime('%Y-%m-%d_%H-%M')}_"
        f"{np.mean(scores):0.3f}_"
        f"{np.mean(scores) + np.std(scores) * 2 / len(scores) ** 0.5:0.3f}_ext")
    y_oof.to_csv(conf.DATA_PROCESSED + f"oof_{stamp}.csv", header=True)
    y_pred.to_csv(conf.DATA_PROCESSED + f"sub_{stamp}.csv", header=True)
    print(pd.DataFrame(feat_importance, index=x_train.columns, columns=["value"]).sort_values("value", ascending=False))


if __name__ == '__main__':
    train_ext()
