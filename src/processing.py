"""Процессинг исходных данных."""
import pathlib

import pandas as pd
import tqdm
from scipy import signal
from scipy import stats

from src import conf

# Размер тестовой серии
TEST_SIZE = 150000


def yield_train_blocks():
    """Генерирует последовательные куски данных.

    Массив данных может не помещаться в память, поэтому гениратор выдает данные блоками размером с тестовые выборки.
    К собственно акустическим данным добавляется последнее значение времени до землятресения.
    """
    dfs_gen = pd.read_csv(
        conf.DATA_RAW,
        names=["x", "y"],
        skiprows=1,
        chunksize=TEST_SIZE
    )
    for df in dfs_gen:
        time = df.y
        first_time = time.iloc[0]
        last_time = time.iloc[-1]
        # В середине блока произошло землятресение и отсчет времени начался заново
        if first_time < last_time:
            continue
        # Последний неполный блок в тренировочной серии отбрасывается
        if len(df) == TEST_SIZE:
            df.reset_index(drop=True, inplace=True)
            yield df.x, last_time


def make_features(df_x: pd.Series):
    """Создание признаков для обучения и тестов."""
    feat = dict()

    mean_abs = (df_x - df_x.mean()).abs()
    feat["mean_abs_med"] = mean_abs.median()

    feat["trim_mean_10"] = stats.trim_mean(df_x, 0.1)

    welch = signal.welch(df_x)[1]
    for num in [2, 3, 14, 28, 30]:   # Еще 14
        feat[f"welch_{num}"] = welch[num]

    feat["roll_25_std_q05"] = df_x.rolling(25).std().dropna().quantile(0.05)

    feat["roll_125_std_q05"] = df_x.rolling(125).std().dropna().quantile(0.05)  #

    roll_375_std = df_x.rolling(375).std().dropna()
    feat["roll_375_std_q05"] = roll_375_std.quantile(0.05)
    feat["roll_375_std_med"] = roll_375_std.median()

    half = len(roll_375_std) // 2
    feat["roll_375_std_half1_med"] = roll_375_std.iloc[:half].median()
    feat["roll_375_std_half2_med"] = roll_375_std.iloc[-half:].median()

    feat["roll_1000_std_q05"] = df_x.rolling(1000).std().dropna().quantile(0.05)

    roll_1500 = df_x.rolling(1500)
    feat["roll_1500_std_q05"] = roll_1500.std().dropna().quantile(0.05)
    roll_1500_mean = roll_1500.mean().dropna()
    feat["roll_1500_mean_q01"] = roll_1500_mean.quantile(0.01)
    feat["roll_1500_mean_q99"] = roll_1500_mean.quantile(0.99)

    # New
    norm_welch = signal.welch((df_x / df_x.rolling(375).std()).dropna())[1]

    feat["norm_welch_max"] = max(norm_welch)
    for num in range(1, 34):
        feat[f"norm_welch_{num}"] = norm_welch[num]

    return feat


def get_train_path():
    """Путь к обучающим данным."""
    return pathlib.Path(conf.DATA_PROCESSED + f"train.pickle")


def make_train_set():
    """Создает и сохраняет на диск признаки для обучающих примеров."""
    data = []
    for df_x, y in tqdm.tqdm(yield_train_blocks()):
        feat = make_features(df_x)
        feat["y"] = y
        data.append(feat)
    data = pd.DataFrame(data).sort_index(axis=1)
    path = get_train_path()
    data.to_pickle(path)
    return data


def train_set():
    """Данные для обучения."""
    path = get_train_path()
    if path.exists():
        data = pd.read_pickle(path)
    else:
        data = make_train_set()
    return data.drop("y", axis=1), data.y


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
    data.to_pickle(path)
    return data


if __name__ == '__main__':
    print(make_train_set())
