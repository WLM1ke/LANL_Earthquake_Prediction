"""Загрузка и обработка данных."""
import logging
import pathlib

import pandas as pd
import numpy as np
import tqdm
from scipy import signal
from scipy import stats

from model import conf

LOGGER = logging.getLogger(__name__)

# Размер тестовой серии
TEST_SIZE = 150000


def load_data():
    """Загрузка и преобразование типов данных."""
    LOGGER.info("Начало загрузки данных")
    df = pd.read_csv(
            conf.DATA_RAW,
            names=["x", "y"],
            skiprows=1,
            dtype={"x": "float32", "y": "float32"}
        )
    LOGGER.info("Done")
    return df


def make_features(df_x):
    """Данные разбиваются на блоки и создают признаки для них."""
    feat = dict()
    mean_abs = (df_x - df_x.mean()).abs()
    feat["mean_abs_med"] = mean_abs.median()

    roll_std = df_x.rolling(375).std().dropna()
    feat["std_roll_med_375"] = roll_std.median()

    half = len(roll_std) // 2
    feat["std_roll_half1"] = roll_std.iloc[:half].median()
    feat["std_roll_half2"] = roll_std.iloc[-half:].median()

    welch = signal.welch(df_x)[1]
    for num in [2, 3, 28, 30]:
        feat[f"welch_{num}"] = welch[num]

    feat["ave10"] = stats.trim_mean(df_x, 0.1)

    feat["q05_roll_std_25"] = df_x.rolling(25).std().dropna().quantile(0.05)
    feat["q05_roll_std_375"] = df_x.rolling(375).std().dropna().quantile(0.05)
    feat["q05_roll_std_1500"] = df_x.rolling(1500).std().dropna().quantile(0.05)
    feat["q05_roll_std_1000"] = df_x.rolling(1000).std().dropna().quantile(0.05)
    feat["q01_roll_mean_1500"] = df_x.rolling(1500).mean().dropna().quantile(0.01)
    feat["q99_roll_mean_1500"] = df_x.rolling(1500).mean().dropna().quantile(0.99)

    return feat


def make_train_set():
    """Создает и сохраняет на диск признаки для обучающих примеров."""
    df = load_data()
    data = []
    df_x = df.x
    df_y = df.y
    for loc_end in tqdm.tqdm(df.index[TEST_SIZE::TEST_SIZE]):
        first_time = df_y.iloc[loc_end - TEST_SIZE]
        last_time = df_y.iloc[loc_end]
        # В середине блока произошло землятресение и отсчет времени начался заново
        if first_time < last_time:
            continue
        feat = make_features(df_x.iloc[loc_end - TEST_SIZE:loc_end])
        feat["y"] = last_time
        data.append(feat)
    data = pd.DataFrame(data).sort_index(axis=1)
    return data.sort_values("y").reset_index(drop=True)


def train_set():
    """Загружает или создает данные для обучения."""
    path = pathlib.Path(conf.DATA_PROCESSED + f"train.pickle")
    if path.exists():
        data = pd.read_pickle(path)
    else:
        data = make_train_set()
        data.to_pickle(path)
    return data.drop(["y"], axis=1), data.y


def test_set():
    """Формирование признаков по аналогии с тренировочным набором."""
    path = pathlib.Path(conf.DATA_PROCESSED + f"test.pickle")
    if path.exists():
        return pd.read_pickle(path)
    data = []
    seg_id = pd.read_csv(conf.DATA_SUB).seg_id
    for name in tqdm.tqdm(seg_id):
        df = pd.read_csv(
            conf.DATA_TEST.format(name),
            names=["x"],
            skiprows=1
        )
        feat = make_features(df.x)
        data.append(feat)
    data = pd.DataFrame(data).sort_index(axis=1)
    data.index = seg_id
    path = pathlib.Path(conf.DATA_PROCESSED + f"test.pickle")
    data.to_pickle(path)
    return data


if __name__ == '__main__':
    train_set()
    test_set()
