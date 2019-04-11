"""Обучение модели."""
import logging
import time

import catboost
import pandas as pd
import numpy as np

from model import conf
from model import processing

LOGGER = logging.getLogger(__name__)

# Настройки валидации
SEED = 284702
FOLDS = 13
ITERATIONS = 8000
LEARNING_RATE = 0.03

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
    allow_writing_files=False
)


def train_catboost():
    """Обучение catboost."""
    x_train, y_train = processing.train_set()
    x_test = processing.test_set()
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

    for fold in range(FOLDS):
        index_valid = x_train.index[fold::FOLDS]
        index_train = x_train.index.difference(index_valid)
        pool_train = catboost.Pool(
            data=x_train.loc[index_train],
            label=y_train.loc[index_train],
            cat_features=None,
            weight=None
        )
        pool_valid = catboost.Pool(
            data=x_train.loc[index_valid],
            label=y_train.loc[index_valid],
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
        y_oof.loc[index_valid] = clf.predict(pool_valid)
        y_pred += clf.predict(pool_test) / FOLDS
        feat_importance += pd.DataFrame(
            clf.get_feature_importance(prettified=True),
            columns=["name", "value"]
        ).set_index("name") / FOLDS

    LOGGER.info(f"Количество деревьев: {sorted(trees)}")
    LOGGER.info(f"Среднее количество деревьев: {np.mean(trees):.0f} +/- {np.std(trees):.0f}")
    LOGGER.info(f"MAE на кроссвалидации: " + str(np.round(sorted(scores), 5)))
    LOGGER.info(f"MAE среднее: {np.mean(scores):0.3f} +/- {np.std(scores):0.3f}")

    y_oof.to_csv(conf.DATA_PROCESSED + f"oof_{time.strftime('%Y-%m-%d_%H-%M')}_mae-{np.mean(scores):0.3f}.csv", header=True)
    y_pred.to_csv(conf.DATA_PROCESSED + f"sub_{time.strftime('%Y-%m-%d_%H-%M')}_mae-{np.mean(scores):0.3f}.csv", header=True)
    print(feat_importance.sort_values("value", ascending=False))


if __name__ == '__main__':
    train_catboost()
