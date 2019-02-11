"""Процессинг исходных данных."""
import pathlib

import pandas as pd
import tqdm
import nolds
from scipy import signal

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


def make_features(df_x):
    """Разбивает данные на блоки и создает описательную статистику для них."""
    feat = dict()
    feat["mean"] = df_x.mean()
    # feat["std"] = df_x.std()
    # feat["skew"] = df_x.skew()
    # feat["kurt"] = df_x.kurt()

    std = df_x.std()
    mean = feat["mean"]
    feat[f"count_std_5"] = (((df_x - mean) / std).abs() > 5).sum()
    feat[f"count_std5_1"] = (((df_x - 4.5) / 5).abs() > 1).sum()

    mean_abs = (df_x - feat["mean"]).abs()
    feat["mean_abs_min"] = mean_abs.min()
    feat["mean_abs_med"] = mean_abs.median()

    roll_std = df_x.rolling(375).std().dropna()
    feat["std_roll_min_375"] = roll_std.min()
    feat["std_roll_med_375"] = roll_std.median()

    half = len(roll_std) // 2
    feat["std_roll_half1"] = roll_std.iloc[:half].median()
    feat["std_roll_half2"] = roll_std.iloc[-half:].median()

    # feat["hurst"] = nolds.hurst_rs(df_x.values)

    # Какая-то полная хрень - хотелось бы более осмысленный выбор
    welch = signal.welch(df_x)[1]

    for num in [2, 3, 19, 25, 28, 29, 30, 32]:
        feat[f"welch_{num}"] = welch[num]

    welch = welch / welch.sum()
    for num in [3, 6, 8, 15, 18, 24]:
        feat[f"welch_share_{num}"] = welch[num]

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
