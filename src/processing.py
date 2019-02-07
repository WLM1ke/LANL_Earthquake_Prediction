"""Процессинг исходных данных."""
import pathlib

import pandas as pd
import tqdm

from src import conf

# Размер тестовой серии
TEST_SIZE = 150000


def yield_train_blocks(passes, chunk_size=TEST_SIZE):
    """Генерирует последовательные куски данных.

    Массив данных может не помещаться в память, поэтому гениратор выдает данные блоками размером с тестовые выборки.
    К собственно акустическим данным добавляется последнее значение времени до землятресения и номер группы (номер
    землятресения в обучающей выборке).
    """
    for pass_ in range(passes):
        dfs_gen = pd.read_csv(
            conf.DATA_RAW,
            names=["x", "y"],
            dtype={"x": "int32", "y": "float32"},
            skiprows=1 + chunk_size // passes * pass_,
            chunksize=chunk_size
        )
        group = 0
        for df in dfs_gen:
            time = df.y
            first_time = time.iloc[0]
            last_time = time.iloc[-1]
            # В середине блока произошло землятресение и отсчет времени начался заново
            if first_time < last_time:
                group += 1
                continue
            # Последний неполный блок в тренировочной серии отбрасывается
            if len(df) == chunk_size:
                df.reset_index(drop=True, inplace=True)
                yield df.x, last_time, group


def make_features(df_x):
    """Разбивает данные на блоки и создает описательную статистику для них."""
    rez = pd.Series()
    rez["mean"] = df_x.mean()
    rez["std"] = df_x.std()
    # rez["skew"] = df_x.skew()
    # rez["max"] = df_x.max()
    # rez["min"] = df_x.min()
    rez["kurt"] = df_x.kurt()
    mean_abs = (df_x - rez["mean"]).abs()
    rez["mean_abs_min"] = mean_abs.min()
    rez["mean_abs_med"] = mean_abs.median()
    # rez["mean_abs_max"] = mean_abs.max()
    roll_std = df_x.rolling(375).std().dropna()
    rez["std_roll_min_375"] = roll_std.min()
    rez["std_roll_med_375"] = roll_std.median()
    # rez["std_roll_max_375"] = roll_std.max()
    # rez["std_roll_cov"] = roll_std.clip(roll_std.quantile(0.1), roll_std.quantile(0.9)).reset_index().cov().iloc[0, 1] / 100000
    half = len(roll_std) // 2
    rez["std_roll_half_pct"] = roll_std.iloc[-half:].median() / roll_std.iloc[:half].median()
    # rez["std_roll_half_delta"] = roll_std.iloc[-half:].median() - roll_std.iloc[:half].median()
    return rez


def make_train_set(rebuild=conf.REBUILD, passes=conf.PASSES):
    """Данные для обучения."""
    path = pathlib.Path(conf.DATA_PROCESSED + f"train_passes_{passes}.pickle")
    if not rebuild and path.exists():
        return pd.read_pickle(path)
    data = []
    for df_x, y, group in tqdm.tqdm(yield_train_blocks(passes)):
        feat = make_features(df_x)
        feat["y"] = y
        feat["group"] = group
        data.append(feat)
    data = pd.concat(data, axis=1, ignore_index=True).T
    data.to_pickle(path)
    return data


def make_test_set(rebuild=conf.REBUILD):
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
    data = pd.concat(data, axis=1, ignore_index=True).T
    data.index = seg_id
    path = pathlib.Path(conf.DATA_PROCESSED + f"test.pickle")
    data.to_pickle(path)
    return data


if __name__ == '__main__':
    print(make_features(next(yield_train_blocks(1))[0]))
