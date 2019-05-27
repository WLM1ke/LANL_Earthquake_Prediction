"""Блендинг результатов нескольких моделей."""
import logging
import time

import catboost
import pandas as pd
import numpy as np

from model import conf
from model import processing

SOURCE = [
    "sub_2019-04-29_10-00_1.938_1.974_cat.csv",
    "sub_2019-04-29_10-57_1.940_1.974_lgbm.csv",
]


def blend():
    """Загрузка тестовых предсказаний и взятие среднего."""
    df = 0
    for name in SOURCE:
        df += pd.read_csv(conf.DATA_PROCESSED + name, header=0, index_col=0) / len(SOURCE)
    stamp = f"sub_{time.strftime('%Y-%m-%d_%H-%M')}_blend.csv"
    df.to_csv(conf.DATA_PROCESSED + stamp, header=True)


if __name__ == '__main__':
    blend()
