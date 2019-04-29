"""Загрузка и обработка данных."""
import logging
import pathlib
from itertools import product

import pandas as pd
import tqdm
import numpy as np
from scipy import signal
from scipy import stats
from scipy.signal import convolve
from scipy.signal import hilbert
from scipy.signal.windows import hann
from sklearn.linear_model import LinearRegression
from tsfresh.feature_extraction import feature_calculators

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

    # Спектральная плотность (диапазоны выбраны в ручную) - нечто похожее используется при анализе голоса в NN
    welch = signal.welch(df_x)[1]
    for num in [2, 3, 28, 30]:
        feat[f"welch_{num}"] = welch[num]

    # Фичи на скользящих медианах - идейно похоже на Pooling только не max и average, а MedianPolling
    mean_abs = (df_x - df_x.mean()).abs()
    feat["mean_abs_med"] = mean_abs.median()

    roll_std = df_x.rolling(375).std().dropna()
    feat["std_roll_med_375"] = roll_std.median()

    half = len(roll_std) // 2
    feat["std_roll_half1"] = roll_std.iloc[:half].median()
    feat["std_roll_half2"] = roll_std.iloc[-half:].median()

    # Фичи на скользящих глубоких квантилях - тоже нейкий QuantilePolling
    feat["q05_roll_std_25"] = df_x.rolling(25).std().dropna().quantile(0.05)
    feat["q05_roll_std_375"] = df_x.rolling(375).std().dropna().quantile(0.05)
    feat["q05_roll_std_1500"] = df_x.rolling(1500).std().dropna().quantile(0.05)
    feat["q05_roll_std_1000"] = df_x.rolling(1000).std().dropna().quantile(0.05)
    feat["q01_roll_mean_1500"] = df_x.rolling(1500).mean().dropna().quantile(0.01)
    feat["q99_roll_mean_1500"] = df_x.rolling(1500).mean().dropna().quantile(0.99)

    feat["ave10"] = stats.trim_mean(df_x, 0.1)

    # Pre Main
    feat["num_peaks_10"] = feature_calculators.number_peaks(df_x, 10)
    feat["percentile_roll_std_5"] = np.percentile(df_x.rolling(10000).std().dropna().values, 5)
    feat["afc_50"] = feature_calculators.autocorrelation(df_x, 50)

    return feat


def make_train_set():
    """Создает и сохраняет на диск признаки для обучающих примеров."""
    df = load_data()
    data = []
    df_x = df.x
    df_y = df.y
    for loc_end in tqdm.tqdm(range(TEST_SIZE - 1, len(df), TEST_SIZE)):
        first_time = df_y.iloc[loc_end - TEST_SIZE + 1]
        last_time = df_y.iloc[loc_end]
        # В середине блока произошло землятресение и отсчет времени начался заново
        if first_time < last_time:
            continue
        feat = make_features(df_x.iloc[loc_end - TEST_SIZE + 1:loc_end + 1])
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
