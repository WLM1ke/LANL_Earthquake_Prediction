"""Обучение модели - LightGBM RF-like."""
import logging
import time

import lightgbm as lgb
import pandas as pd
import numpy as np

from model import conf
from model import processing
from model.conf import K_FOLDS
from model.conf import SEED

LOGGER = logging.getLogger(__name__)

ITERATIONS = 10000

CLF_PARAMS = dict(
    learning_rate=None,
    metric="mae",
    objective="mae",
    num_leaves=31,
    min_data_in_leaf=20,
    max_depth=-1,
    bagging_freq=1,
    bagging_fraction=0.632,
    feature_fraction=0.632,
    boost="rf",
    seed=SEED,
    verbosity=-1,
)

DROP = [
    "std_roll_half1", "percentile_roll_std_5", "q05_roll_std_375", "std_roll_half2"
]


def train_light_gbm_rf():
    """Обучение LightGBM."""
    x_train, y_train = processing.train_set()
    x_test = processing.test_set()

    x_train.drop(DROP, axis=1, inplace=True)
    x_test.drop(DROP, axis=1, inplace=True)

    y_oof = pd.Series(0, index=x_train.index, name="oof_lgbm_rf")
    y_pred = pd.Series(0, index=x_test.index, name="time_to_failure")
    trees = []
    scores = []
    feat_importance = 0

    for index_train, index_valid in K_FOLDS.split(x_train):
        pool_train = lgb.Dataset(
            x_train.iloc[index_train],
            label=y_train.iloc[index_train],
        )
        pool_valid = lgb.Dataset(
            x_train.iloc[index_valid],
            label=y_train.iloc[index_valid],
        )
        clf = lgb.train(
            CLF_PARAMS,
            pool_train,
            ITERATIONS,
            valid_sets=[pool_train, pool_valid],
            verbose_eval=ITERATIONS // 100,
            early_stopping_rounds=ITERATIONS // 10
        )

        trees.append(clf.best_iteration)
        scores.append(clf.best_score["valid_1"]["l1"])

        y_oof.iloc[index_valid] = clf.predict(x_train.iloc[index_valid], num_iteration=clf.best_iteration)
        y_pred += clf.predict(x_test, num_iteration=clf.best_iteration) / K_FOLDS.get_n_splits()

        feat_importance += clf.feature_importance("gain") / K_FOLDS.get_n_splits()
        print("\n")

    LOGGER.info(f"Количество деревьев: {sorted(trees)}")
    LOGGER.info(f"Среднее количество деревьев: {np.mean(trees):.0f} +/- {np.std(trees):.0f}")
    LOGGER.info(f"MAE на кроссвалидации: " + str(np.round(sorted(scores), 5)))
    LOGGER.info(f"MAE среднее: {np.mean(scores):0.3f} +/- {np.std(scores):0.3f}")

    stamp = (
        f"{time.strftime('%Y-%m-%d_%H-%M')}_"
        f"{np.mean(scores):0.3f}_"
        f"{np.mean(scores) + np.std(scores) * 2 / len(scores) ** 0.5:0.3f}_lgbm_rf")
    y_oof.to_csv(conf.DATA_PROCESSED + f"oof_{stamp}.csv", header=True)
    y_pred.to_csv(conf.DATA_PROCESSED + f"sub_{stamp}.csv", header=True)
    print(pd.DataFrame(feat_importance, index=x_train.columns, columns=["value"]).sort_values("value", ascending=False))


if __name__ == '__main__':
    train_light_gbm_rf()
