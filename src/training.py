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
    "welch_6", "welch_15",
    "welch_86", "welch_16",
    "welch_75", "welch_0", "welch_97", "welch_73",
    "welch_13", "welch_62", "welch_108", "welch_94", "welch_107", "welch_54",
    "welch_63", "welch_81", "welch_91", "welch_96", "welch_48", "welch_38", "welch_51", "welch_43",
    "welch_122", "welch_116", "welch_17", "welch_12", "welch_112", "welch_23", "welch_114", "welch_84", "welch_8", "welch_93",
    "welch_66", "welch_11", "welch_77", "welch_10", "welch_40", "welch_34", "welch_92", "welch_71", "welch_80", "welch_99", "welch_106",
    "welch_55", "welch_87", "welch_113", "welch_44", "welch_59", "welch_18", "welch_88",
    "welch_5", "welch_44", "welch_22", "welch_57", "welch_59", "welch_20", "welch_70",
    "welch_128", "welch_56", "welch_72", "welch_125", "welch_76", "welch_69", "welch_53", "welch_49",
    "welch_105", "welch_53", "welch_52",  "welch_35", "welch_9",  "welch_85", "welch_41", "welch_119",
    "welch_21", "welch_61", "welch_127", "welch_67", "welch_102", "welch_37", "welch_78", "welch_90", "welch_124",
    "welch_82", "welch_36", "welch_115", "welch_111", "welch_7", "welch_21", "welch_50", "welch_110", "welch_117",
    "welch_47", "welch_121", "welch_65", "welch_45", "welch_46", "welch_123", "welch_32", "welch_126", "welch_39", "welch_1",
    "welch_33", "welch_45", "welch_25", "welch_31", "welch_100", "welch_24", "welch_1", "welch_121", "welch_26", "welch_46",
    "welch_74", "welch_79", "welch_60", "welch_58",
    "welch_64", "welch_74", "welch_83", "welch_58", "welch_118", "welch_120", "welch_27", "welch_104",
    "welch_68",
    "welch_19",
    "welch_89", "welch_42",
    "welch_95", "welch_101",
    "welch_98", "welch_109",
    "welch_103",
    "welch_4",
    "hurst",
    "welch_29",
    "count_std_5",
    "mean_abs_min",
    "std_roll_min_375",

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
