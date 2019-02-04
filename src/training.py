"""Обучение моделей."""
import logging

from sklearn import metrics
from sklearn import model_selection
import catboost
import pandas as pd
import numpy as np

from src import processing

# Настройки валидации
SEED = 284702
FOLDS = model_selection.KFold(n_splits=5, shuffle=True, random_state=SEED)


def train_catboost(passes, folds=FOLDS):
    """Обучение catboost."""
    df = processing.make_train_set(passes=passes)
    y = df.y
    group = df.group
    x = df.drop(["y", "group"], axis=1)
    oof_y = pd.Series(0, index=df.index)
    trees = []
    scores = []
    for train_ind, valid_ind in folds.split(x, y, groups=group):
        train_x = x.iloc[train_ind]
        train_y = y.iloc[train_ind]

        valid_x = x.iloc[valid_ind]
        valid_y = y.iloc[valid_ind]

        clf_params = dict(
            loss_function="MAE",
            random_state=SEED,
            depth=6,
            od_type="Iter",
            od_wait=20,
            verbose=100,
            learning_rate=0.1,
            iterations=10000,
            allow_writing_files=False
        )
        clf = catboost.CatBoostRegressor(**clf_params)

        fit_params = dict(
            X=train_x,
            y=train_y,
            eval_set=(valid_x, valid_y),
            cat_features=[]
        )

        clf.fit(**fit_params)
        trees.append(clf.tree_count_)
        scores.append(clf.best_score_['validation_0']['MAE'])
        oof_y.iloc[valid_ind] = clf.predict(valid_x)

    logging.info(f"Количество деревьев: {trees}")
    logging.info(f"Среднее количество деревьев: {np.mean(trees):.0f} +/- {np.std(trees):.0f}")
    logging.info(f"MAE на кроссвалидации: " + str(np.round(scores, 5)))
    logging.info(f"MAE среднее: {np.mean(scores):0.5f} +/- {np.std(scores):0.5f}")
    logging.info(f"OOF MAE: {metrics.mean_absolute_error(y, oof_y):0.5f}")


if __name__ == '__main__':
    train_catboost(1)

"""
INFO:root:Количество деревьев: [401, 410, 481, 543, 474]
INFO:root:Среднее количество деревьев: 462 +/- 52
INFO:root:MAE на кроссвалидации: [2.11137 2.15002 2.18329 2.13952 2.21558]
INFO:root:MAE среднее: 2.15995 +/- 0.03610
INFO:root:OOF MAE: 2.15995
"""
