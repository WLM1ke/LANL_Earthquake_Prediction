"""Обучение моделей."""
import logging
import time

from sklearn import metrics
from sklearn import model_selection
import catboost
import pandas as pd
import numpy as np
import boruta
import lightgbm

from src import conf
from src import processing

# Настройки валидации
SEED = 284702
N_SPLITS = 7
FOLDS = model_selection.KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
DROP = [
    "norm_welch_22", "norm_welch_10", "norm_welch_14", "norm_welch_29",
    "norm_welch_12", "norm_welch_15", "norm_welch_13", "norm_welch_20", "norm_welch_31",
    "norm_welch_max", "norm_welch_21",
    "norm_welch_30", "norm_welch_11",  #
    "norm_welch_25", "norm_welch_23",
    "norm_welch_9",
    "norm_welch_5",
    "norm_welch_32",
    "norm_welch_22"
]

CLF_PARAMS = dict(
    loss_function="MAE",
    random_state=SEED,
    depth=conf.DEPTH,
    od_type="Iter",
    od_wait=400,
    verbose=100,
    learning_rate=conf.LEARNING_RATE,
    iterations=10000,
    allow_writing_files=False
)


def catboost_cv():
    """Кросс-валидация и выбор оптимального количества деревьев."""
    x, y = processing.train_set()
    x = x.drop(DROP, axis=1)
    oof_y = pd.Series(0, index=x.index, name="oof_y")
    trees = []
    scores = []
    for train_index, valid_index in FOLDS.split(x):
        x_train = x.iloc[train_index]
        y_train = y.iloc[train_index]

        x_valid = x.iloc[valid_index]
        y_valid = y.iloc[valid_index]

        train = catboost.Pool(
            data=x_train,
            label=y_train,
            cat_features=None,

        )

        valid = catboost.Pool(
            data=x_valid,
            label=y_valid,
            cat_features=None,
        )

        clf = catboost.CatBoostRegressor(**CLF_PARAMS)

        fit_params = dict(
            X=train,
            eval_set=[valid],
        )

        clf.fit(**fit_params)
        trees.append(clf.tree_count_)
        scores.append(clf.best_score_['validation_0']['MAE'])
        oof_y.iloc[valid_index] = clf.predict(valid)

    logging.info(f"Количество деревьев: {trees}")
    logging.info(f"Среднее количество деревьев: {np.mean(trees):.0f} +/- {np.std(trees):.0f}")
    logging.info(f"MAE на кроссвалидации: " + str(np.round(scores, 5)))
    logging.info(f"MAE среднее: {np.mean(scores):0.3f} +/- {np.std(scores):0.3f}")
    oof_mae = metrics.mean_absolute_error(y, oof_y)
    logging.info(f"OOF MAE: {oof_mae:0.3f}")
    pd.concat([y, oof_y], axis=1).to_pickle(
        conf.DATA_PROCESSED + "oof.pickle"
    )
    best_trees = sorted(trees)[N_SPLITS // 2 + 1]
    return best_trees, oof_mae


def catboost_predict():
    """Прогноз для catboost."""
    best_trees, oof_mae = catboost_cv()
    train_x, train_y = processing.train_set()

    test_x = processing.test_set()[train_x.columns]
    CLF_PARAMS["iterations"] = best_trees
    clf = catboost.CatBoostRegressor(**CLF_PARAMS)
    fit_params = dict(
        X=train_x,
        y=train_y,
        cat_features=[]
    )
    clf.fit(**fit_params)
    sub = pd.DataFrame(
        clf.predict(test_x),
        index=test_x.index,
        columns=["time_to_failure"])
    sub.to_csv(conf.DATA_PROCESSED + f"sub_{time.strftime('%Y-%m-%d_%H-%M')}_mae-{oof_mae:0.3f}.csv")

    logging.info("Важность признаков:")
    for i, v in clf.get_feature_importance(prettified=True):
        logging.info(i.ljust(30) + str(v))

    pd.DataFrame(clf.get_feature_importance(prettified=True)).set_index(0).to_pickle(
        conf.DATA_PROCESSED + "importance.pickle"
    )

    logging.info("Попарная важность признаков:")
    for i, j, value in clf.get_feature_importance(fstr_type="Interaction", prettified=True)[:20]:
        logging.info(train_x.columns[i].ljust(20) + train_x.columns[j].ljust(20) + str(value))


def feat_sel():
    """Выбор признаков."""
    x, y = processing.train_set()
    x = x.drop(DROP, axis=1)
    clf = lightgbm.LGBMRegressor(boosting_type="rf",
                                 bagging_freq=1,
                                 bagging_fraction=0.632,
                                 feature_fraction=0.632)
    feat_selector = boruta.BorutaPy(clf, n_estimators=500, verbose=2)
    feat_selector.fit(x.values, y.values)
    print(x.columns[feat_selector.support_weak_])
    print(x.columns[feat_selector.support_])
    print(pd.Series(feat_selector.ranking_, index=x.columns).sort_values())


if __name__ == '__main__':
    catboost_predict()
    feat_sel()
