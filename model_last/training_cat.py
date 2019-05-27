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
    depth=10,
    od_type="Iter",
    od_wait=ITERATIONS // 10,
    verbose=ITERATIONS // 100,
    learning_rate=LEARNING_RATE,
    iterations=ITERATIONS,
    allow_writing_files=False,
)

DROP = [
    "welch_clipped_4", "welch_clipped_22",  #
    "welch_clipped_2", "welch_clipped_18", "welch_clipped_15", "welch_clipped_3",  #
    "welch_clipped_20",  #
    "welch_clipped_6", "welch_clipped_19", "welch_clipped_13", "welch_clipped_16",
    "welch_clipped_9", "welch_clipped_26", "welch_clipped_0", "welch_clipped_12", "welch_clipped_23",  #
    "welch_clipped_10",  #
    "welch_clipped_31", "welch_clipped_1",
    "welch_clipped_5",  #
    "welch_clipped_30", "welch_clipped_21",
    "welch_clipped_14",  #
    "welch_28", "welch_clipped_17",
    "welch_clipped_32",
    "welch_clipped_7",
    "welch_clipped_29",
    "q05_roll_std_1000"  #
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
    y_oof = pd.Series(0, index=x_train.index, name="oof_cat")
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
