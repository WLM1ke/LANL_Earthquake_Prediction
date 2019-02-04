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
            # Последний не полный блок в тренировочной серии отбрасывается
            if len(df) == chunk_size:
                df.reset_index(drop=True, inplace=True)
                yield df.x, last_time, group


def make_features(df_x, blocks=15):
    """Разбивает данные на блоки и создает описательную статистику для них."""
    size, res = divmod(len(df_x), blocks)
    assert df_x.shape == (TEST_SIZE,), "Неверный размер даных"
    assert not res, "Неверное количество блоков"
    features = ["std"]
    df_x = df_x.groupby(lambda x: x // size).agg(features).stack()
    df_x.index = df_x.index.map(lambda x: f"{x[1]}_{x[0]}")
    return df_x


def make_train_set(rebuild=False, passes=30):
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


if __name__ == '__main__':
    print(make_train_set(passes=1))
    make_train_set().info()
