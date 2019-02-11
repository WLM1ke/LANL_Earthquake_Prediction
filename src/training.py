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


def train_catboost(rebuild=conf.REBUILD, passes=conf.PASSES):
    """Обучение catboost."""
    x, y, group1, group2 = processing.train_set(rebuild=rebuild, passes=passes)
    x = x.drop(DROP, axis=1)
    mask_high = y < conf.CROP_Y
    x, y, group1, group2 = x[mask_high], y[mask_high], group1[mask_high], group2[mask_high]

    oof_y = pd.Series(0, index=x.index, name="oof_y")
    trees = []
    scores = []

    groups = group1.unique()

    for _, valid_group_index in FOLDS.split(groups):
        valid_groups = groups[valid_group_index]
        valid_mask = group1.isin(valid_groups) | group2.isin(valid_groups)
        train_mask = ~valid_mask

        train_x = x.loc[train_mask]
        train_y = y.loc[train_mask]

        valid_x = x.loc[valid_mask]
        valid_y = y.loc[valid_mask]

        clf = catboost.CatBoostRegressor(**CLF_PARAMS)

        fit_params = dict(
            X=train_x,
            y=train_y,
            eval_set=(valid_x, valid_y),
            cat_features=[]
        )

        clf.fit(**fit_params)
        trees.append(clf.tree_count_)
        scores.append(clf.best_score_['validation_0']['MAE'])
        oof_y.loc[valid_mask] = clf.predict(valid_x)

    logging.info(f"Количество деревьев: {trees}")
    logging.info(f"Среднее количество деревьев: {np.mean(trees):.0f} +/- {np.std(trees):.0f}")
    logging.info(f"MAE на кроссвалидации: " + str(np.round(scores, 5)))
    logging.info(f"MAE среднее: {np.mean(scores):0.3f} +/- {np.std(scores):0.3f}")
    oof_mae = metrics.mean_absolute_error(y, oof_y)
    logging.info(f"OOF MAE: {oof_mae:0.3f}")

    pd.concat([y, oof_y], axis=1).to_pickle(
        conf.DATA_PROCESSED + "oof.pickle"
    )

    test_x = processing.test_set(rebuild=rebuild)[x.columns]
    CLF_PARAMS["iterations"] = sorted(trees)[N_SPLITS // 2 + 1]
    clf = catboost.CatBoostRegressor(**CLF_PARAMS)
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


def feat_sel(rebuild=conf.REBUILD, passes=conf.PASSES):
    """Выбор признаков."""
    x, y, _, _ = processing.train_set(rebuild=rebuild, passes=passes)
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
    train_catboost()
    feat_sel()
