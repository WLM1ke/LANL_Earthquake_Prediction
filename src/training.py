"""Обучение моделей."""
import logging
import time

from sklearn import metrics
from sklearn import model_selection
import catboost
import pandas as pd
import numpy as np

from src import conf
from src import processing

# Настройки валидации
SEED = 284702
N_SPLITS = 5
FOLDS = model_selection.KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def train_catboost(passes, folds=FOLDS):
    """Обучение catboost."""
    df = processing.make_train_set(passes=passes)
    y = df.y
    group = df.group
    x = df.drop(["y", "group"], axis=1)
    oof_y = pd.Series(0, index=df.index, name="oof_y")
    trees = []
    scores = []
    clf_params = dict(
        loss_function="MAE",
        random_state=SEED,
        depth=conf.DEPTH,
        od_type="Iter",
        od_wait=100,
        verbose=100,
        learning_rate=conf.LEARNING_RATE,
        iterations=10000,
        allow_writing_files=False
    )

    for train_ind, valid_ind in folds.split(x, y, groups=group):
        train_x = x.iloc[train_ind]
        train_y = y.iloc[train_ind]

        valid_x = x.iloc[valid_ind]
        valid_y = y.iloc[valid_ind]

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
    logging.info(f"MAE среднее: {np.mean(scores):0.3f} +/- {np.std(scores):0.3f}")
    oof_mae = metrics.mean_absolute_error(y, oof_y)
    logging.info(f"OOF MAE: {oof_mae:0.3f}")

    pd.concat([y, oof_y], axis=1).to_pickle(
        conf.DATA_PROCESSED + "oof.pickle"
    )

    test_x = processing.make_test_set()
    clf_params["iterations"] = sorted(trees)[N_SPLITS // 2 + 1]
    clf = catboost.CatBoostRegressor(**clf_params)
    fit_params = dict(
        X=x,
        y=y,
        cat_features=[]
    )
    clf.fit(**fit_params)
    sub = pd.DataFrame(clf.predict(test_x), index=test_x.index, columns=["time_to_failure"])
    sub.to_csv(conf.DATA_PROCESSED + f"sub_{time.strftime('%Y-%m-%d_%H-%M')}_passes-{passes}_mae-{oof_mae:0.3f}.csv")

    logging.info("Важность признаков:")
    for i, v in clf.get_feature_importance(prettified=True):
        logging.info(i.ljust(20) + str(v))

    pd.DataFrame(clf.get_feature_importance(prettified=True)).set_index(0).to_pickle(
        conf.DATA_PROCESSED + "importance.pickle"
    )

    logging.info("Попарная важность признаков:")
    for i, j, value in clf.get_feature_importance(fstr_type="Interaction", prettified=True)[:20]:
        logging.info(x.columns[i].ljust(20) + x.columns[j].ljust(20) + str(value))


if __name__ == '__main__':
    train_catboost(30)
