"""Обучение модели - catboost."""
import logging
import time

import catboost
import pandas as pd
import numpy as np

from model import conf
from model import processing
from model.conf import K_FOLDS
from model.conf import SEED

LOGGER = logging.getLogger(__name__)

ITERATIONS = 10000
LEARNING_RATE = 0.04

CLF_PARAMS = dict(
    loss_function="MAE",
    eval_metric=None,
    random_state=SEED,
    depth=6,
    od_type="Iter",
    od_wait=ITERATIONS // 10,
    verbose=ITERATIONS // 100,
    learning_rate=LEARNING_RATE,
    iterations=ITERATIONS,
    allow_writing_files=False,
)

DROP = [
    "num_peaks_10", "percentile_roll_std_5", "afc_50"
]


def train_catboost():
    """Обучение catboost."""
    x_train, y_train = processing.train_set()
    x_test = processing.test_set()

    x_train.drop(DROP, axis=1, inplace=True)
    x_test.drop(DROP, axis=1, inplace=True)

    pool_test = catboost.Pool(
            data=x_test,
            label=None,
            cat_features=None,
            weight=None
        )
    y_oof = pd.Series(0, index=x_train.index, name="oof_y")
    y_pred = pd.Series(0, index=x_test.index, name="time_to_failure")
    trees = []
    scores = []
    feat_importance = 0

    for index_train, index_valid in K_FOLDS.split(x_train):
        pool_train = catboost.Pool(
            data=x_train.iloc[index_train],
            label=y_train.iloc[index_train],
            cat_features=None,
            weight=None
        )
        pool_valid = catboost.Pool(
            data=x_train.iloc[index_valid],
            label=y_train.iloc[index_valid],
            cat_features=None,
            weight=None
        )
        clf = catboost.CatBoostRegressor(**CLF_PARAMS)
        clf.fit(
            X=pool_train,
            eval_set=[pool_valid],
        )
        trees.append(clf.tree_count_)
        scores.append(clf.best_score_['validation_0']['MAE'])
        y_oof.iloc[index_valid] = clf.predict(pool_valid)
        y_pred += clf.predict(pool_test) / K_FOLDS.get_n_splits()
        feat_importance += pd.DataFrame(
            clf.get_feature_importance(prettified=True),
            columns=["name", "value"]
        ).set_index("name") / K_FOLDS.get_n_splits()

    LOGGER.info(f"Количество деревьев: {sorted(trees)}")
    LOGGER.info(f"Среднее количество деревьев: {np.mean(trees):.0f} +/- {np.std(trees):.0f}")
    LOGGER.info(f"MAE на кроссвалидации: " + str(np.round(sorted(scores), 5)))
    LOGGER.info(f"MAE среднее: {np.mean(scores):0.3f} +/- {np.std(scores):0.3f}")

    stamp = (
        f"{time.strftime('%Y-%m-%d_%H-%M')}_"
        f"{np.mean(scores):0.3f}_"
        f"{np.mean(scores) + np.std(scores) * 2 / len(scores) ** 0.5:0.3f}_cat")
    y_oof.to_csv(conf.DATA_PROCESSED + f"oof_{stamp}.csv", header=True)
    y_pred.to_csv(conf.DATA_PROCESSED + f"sub_{stamp}.csv", header=True)
    print(feat_importance.sort_values("value", ascending=False))


if __name__ == '__main__':
    train_catboost()
