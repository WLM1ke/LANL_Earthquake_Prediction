"""Процессинг исходных данных."""
import pathlib

import pandas as pd
import tqdm
import nolds
from scipy import signal
from scipy import stats
from scipy.signal import hilbert
from scipy.signal import convolve
from scipy.signal import hann
import numpy as np
from sklearn.linear_model import LinearRegression

from src import conf

# Размер тестовой серии
TEST_SIZE = 150000


def yield_train_blocks(passes):
    """Генерирует последовательные куски данных.

    Массив данных может не помещаться в память, поэтому гениратор выдает данные блоками размером с тестовые выборки.
    К собственно акустическим данным добавляется последнее значение времени до землятресения и номер группы (номер
    номера кусков данных использовавшихся для создания блока).
    """
    for pass_ in range(passes):
        dfs_gen = pd.read_csv(
            conf.DATA_RAW,
            names=["x", "y"],
            skiprows=1 + TEST_SIZE // passes * pass_,
            chunksize=TEST_SIZE
        )
        group1 = -1
        if pass_:
            group2 = 0
        else:
            group2 = -1
        for df in dfs_gen:
            group1 += 1
            group2 += 1
            time = df.y
            first_time = time.iloc[0]
            last_time = time.iloc[-1]
            # В середине блока произошло землятресение и отсчет времени начался заново
            if first_time < last_time:
                continue
            # Последний неполный блок в тренировочной серии отбрасывается
            if len(df) == TEST_SIZE:
                df.reset_index(drop=True, inplace=True)
                yield df.x, last_time, group1, group2




def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]


def classic_sta_lta(x, length_sta, length_lta):
    sta = np.cumsum(x ** 2)
    # Convert to float
    sta = np.require(sta, dtype=np.float)
    # Copy for LTA
    lta = sta.copy()
    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta
    # Pad zeros
    sta[:length_lta - 1] = 0
    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    return sta / lta



def make_features(df_x):
    """Данные разбиваются на блоки и создают признаки для них."""
    feat = dict()
    # feat["mean"] = df_x.mean()
    # feat["std"] = df_x.std()
    # feat["skew"] = df_x.skew()
    # feat["kurt"] = df_x.kurt()

    # feat[f"count_std5_1"] = (((df_x - 4.5) / 5).abs() > 1).sum()

    mean_abs = (df_x - df_x.mean()).abs()
    feat["mean_abs_med"] = mean_abs.median()

    roll_std = df_x.rolling(375).std().dropna()
    feat["std_roll_med_375"] = roll_std.median()

    half = len(roll_std) // 2
    feat["std_roll_half1"] = roll_std.iloc[:half].median()
    # feat["std_roll_half2"] = roll_std.iloc[-half:].median()

    # feat["hurst"] = nolds.hurst_rs(df_x.values)

    # Не нравится мне это, но дает очень похожий инкрементальный результат на паблике и кросс-валидации
    # Возможный плюс, что welch с дефолтными настройками - попыки их подрихтовать не дают улучшений
    welch = signal.welch(df_x)[1]
    for num in [2, 3, 28, 30]:  # 14
        feat[f"welch_{num}"] = welch[num]

    # New
    feat["ave10"] = stats.trim_mean(df_x, 0.1)

    feat["q05_roll_std_10"] = df_x.rolling(10).std().dropna().quantile(0.05)
    feat["q05_roll_std_100"] = df_x.rolling(100).std().dropna().quantile(0.05)
    feat["q05_roll_std_1000"] = df_x.rolling(1000).std().dropna().quantile(0.05)

    feat["q01_roll_mean_1000"] = df_x.rolling(1000).mean().dropna().quantile(0.01)
    feat["q99_roll_mean_1000"] = df_x.rolling(1000).mean().dropna().quantile(0.99)

    return feat


def make_train_set(passes):
    """Создает и сохраняет на диск признаки для обучающих примеров."""
    data = []
    for df_x, y, group1, group2 in tqdm.tqdm(yield_train_blocks(passes)):
        feat = make_features(df_x)
        feat["y"] = y
        feat["group1"] = group1
        feat["group2"] = group2
        data.append(feat)
    data = pd.DataFrame(data).sort_index(axis=1)
    path = pathlib.Path(conf.DATA_PROCESSED + f"train_passes_{passes}.pickle")
    data.to_pickle(path)
    return data


def train_set(rebuild, passes):
    """Данные для обучения."""
    path = pathlib.Path(conf.DATA_PROCESSED + f"train_passes_{passes}.pickle")
    if not rebuild and path.exists():
        data = pd.read_pickle(path)
    else:
        data = make_train_set(passes)
    return data.drop(["y", "group1", "group2"], axis=1), data.y, data.group1, data.group2


def test_set(rebuild):
    """Формирование признаков по аналогии с тренировочным набором."""
    path = pathlib.Path(conf.DATA_PROCESSED + f"test.pickle")
    if not rebuild and path.exists():
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
    print(make_train_set(1))
