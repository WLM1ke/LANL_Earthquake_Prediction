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
    "mean", "abs_q01", "med", "count_big", "abs_min",
    "abs_q01", "abs_min", "q05", "max_roll_mean_100", "MA_1000MA_std_mean",
    "abs_q05", "q01", "sum", "std", "max_roll_std_100",
    "max_roll_std_100", "Rstd_last_15000", "Rmin_last_15000", "Istd", "max_roll_std_1000",
    "ave_roll_mean_1000", "max_roll_std_10", "q99", "ave_roll_mean_10", "Moving_average_700_mean",
    "q99", "ave_roll_mean_100", "q999", "std_roll_std_100", "abs_q99",
    "abs_q95", "min_roll_std_10", "MA_700MA_std_mean", "std_roll_std_1000", "abs_std",
    "ave_roll_std_100", "mad", "std_roll_std_1000", "abs_std", "iqr",
    "ave_roll_std_1000", "abs_max_roll_std_10", "q05_roll_mean_10", "MA_400MA_std_mean", "min",
    "ave_roll_std_1000", "abs_max_roll_std_100", "av_change_rate_roll_mean_100", "q05_roll_mean_10", "av_change_rate_roll_mean_10",
    "max_roll_mean_1000", "q001", "std_roll_std_10", "exp_Moving_average_300_mean", "Moving_average_3000_mean",
    "q001", "min_roll_mean_10", "Imin", "mean_change_abs", "q95",
    "MA_400MA_BB_low_mean", "av_change_rate_roll_std_1000", "abs_max_roll_std_1000", "max_last_10000", "q99_roll_std_10",
    "MA_700MA_BB_low_mean", "Rmin", "abs_max_roll_mean_10", "kurt", "ave_roll_std_10",  #
    "max_first_10000", "min_last_50000", "Moving_average_1500_mean", "max_last_50000", "std_roll_mean_10",
    "max", "MA_700MA_BB_high_mean", "max_last_50000", "min_roll_mean_100", "q95_roll_std_100", #
    "min_first_10000", "Imax", "abs_trend", "MA_400MA_BB_high_mean", "q99_roll_mean_10",
    "Rmax", "av_change_rate_roll_std_100", "classic_sta_lta3_mean", "Hilbert_mean", "av_change_rate_roll_mean_1000", #
    "Moving_average_6000_mean", "av_change_rate_roll_std_10", "exp_Moving_average_3000_mean", "avg_last_50000", "Rstd",
    "max_roll_mean_10", "abs_max", "q01_roll_mean_10", "std_last_50000", "std_first_50000", #
    "Rmin_last_5000", "std_roll_mean_1000", "min_last_10000", "Hann_window_mean", "std_roll_mean_100",
    "av_change_abs_roll_std_10", "Rmean_last_5000", "mean_change_rate_last_10000", "av_change_abs_roll_std_1000", "min_last_10000",
    "std_first_10000", "av_change_abs_roll_mean_1000", "Rmax_last_15000", "Rstd__last_5000", "trend",
    "q95_roll_mean_10", "q99_roll_std_1000", "Rmax_last_5000", "Rmax_last_15000", "Rmean", #
    "q99_roll_std_100", "q05_roll_mean_1000", "std_last_10000", "q01_roll_mean_100", "mean_change_rate",
    "classic_sta_lta4_mean", "classic_sta_lta2_mean", "mean_change_rate_first_50000", "max_to_min", "q95_roll_std_10", #
    "abs_max_roll_mean_100", "min_first_50000", "max_to_min_diff", "av_change_abs_roll_mean_10", "classic_sta_lta1_mean",
    "min_first_50000", "q95_roll_mean_100", "skew", "min_roll_mean_1000", "classic_sta_lta1_mean", #
    "count_std5_1", "Imean", "q95_roll_std_1000", "max_first_50000", "av_change_abs_roll_mean_100",
    "q01_roll_std_10", "abs_mean", "exp_Moving_average_30000_mean", "avg_last_10000", "Imean",
    "mean_change_rate_last_50000", "av_change_abs_roll_std_100", "q95_roll_mean_1000", "mean_change_rate_first_10000", "abs_max_roll_mean_1000",
    "Rmean_last_15000", "mean_change_rate_last_50000", "av_change_abs_roll_std_100", "min_roll_std_1000", "min_roll_std_100", #
    "q99_roll_mean_100", "std_roll_half2",
    "q01_roll_std_100", "welch_14", #
    "q05_roll_mean_100", "avg_first_10000", #
    "avg_first_50000", "q01_roll_std_1000", #

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

    oof_y = pd.Series(0, index=x.index, name="oof_y")
    trees = []
    scores = []

    groups = group1.unique()

    for _, valid_group_index in FOLDS.split(groups):
        valid_groups = groups[valid_group_index]
        valid_mask = group1.isin(valid_groups) | group2.isin(valid_groups)
        train_mask = ~valid_mask

        x_train = x.loc[train_mask]
        y_train = y.loc[train_mask]

        x_valid = x.loc[valid_mask]
        y_valid = y.loc[valid_mask]

        weight_train = None
        weight_valid = None

        if conf.WEIGHTED:
            max_group = len(conf.GROUP_WEIGHTS) - 1

            group_id_train = y_train.astype('int')
            group_id_train[group_id_train > max_group] = max_group
            weight_train = group_id_train.map(pd.Series(conf.GROUP_WEIGHTS) / group_id_train.value_counts())

            group_id_valid = y_valid.astype('int')
            group_id_valid[group_id_valid > max_group] = max_group
            weight_valid = group_id_valid.map(pd.Series(conf.GROUP_WEIGHTS) / group_id_valid.value_counts())

        train = catboost.Pool(
            data=x_train,
            label=y_train,
            cat_features=None,
            weight=weight_train

        )

        valid = catboost.Pool(
            data=x_valid,
            label=y_valid,
            cat_features=None,
            weight=weight_valid
        )

        clf = catboost.CatBoostRegressor(**CLF_PARAMS)

        fit_params = dict(
            X=train,
            eval_set=[valid],
        )

        clf.fit(**fit_params)
        trees.append(clf.tree_count_)
        scores.append(clf.best_score_['validation_0']['MAE'])
        oof_y.loc[valid_mask] = clf.predict(valid)

    logging.info(f"Количество деревьев: {trees}")
    logging.info(f"Среднее количество деревьев: {np.mean(trees):.0f} +/- {np.std(trees):.0f}")
    logging.info(f"MAE на кроссвалидации: " + str(np.round(scores, 5)))
    logging.info(f"MAE среднее: {np.mean(scores):0.3f} +/- {np.std(scores):0.3f}")

    weight = None
    if conf.WEIGHTED:
        max_group = len(conf.GROUP_WEIGHTS) - 1
        group_id = y.astype('int')
        group_id[group_id > max_group] = max_group
        weight = group_id.map(pd.Series(conf.GROUP_WEIGHTS) / group_id.value_counts())

    oof_mae = metrics.mean_absolute_error(y, oof_y, weight)
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
