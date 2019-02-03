"""Процессинг исходных данных."""
import pathlib

import pandas as pd
import tqdm

from src import conf


def yield_train_blocks(chunk_size=conf.TEST_SIZE):
    """Генерирует последовательные куски данных."""
    dfs_gen = pd.read_csv(
        conf.DATA_RAW,
        names=["x", "y"],
        dtype={"x": "int32", "y": "float32"},
        skiprows=1,
        chunksize=chunk_size
    )
    for df in dfs_gen:
        time = df.y
        if len(df) == chunk_size and time.iloc[-1] < time.iloc[0]:
            df.reset_index(drop=True, inplace=True)
            yield df


def make_features(df_x, blocks=15):
    """Разбивает данные на блоки и создает описательную статистику для них."""
    size, res = divmod(len(df_x), blocks)
    assert df_x.shape == (conf.TEST_SIZE,), "Неверный размер даных"
    assert not res, "Неверное количество блоков"
    features = ["std"]
    df_x = df_x.groupby(lambda x: x // size).agg(features).stack()
    df_x.index = df_x.index.map(lambda x: f"{x[1]}_{x[0]}")
    return df_x.astype("float32")


def make_train_set():
    """Данные для обучения."""
    path = pathlib.Path(conf.DATA_PROCESSED + "train.pickle")
    if path.exists():
        return pd.read_pickle(path)
    data = []
    for df in tqdm.tqdm(yield_train_blocks()):
        y = df.y.iloc[-1]
        feat = make_features(df.x)
        feat['y'] = y
        data.append(feat)
    data = pd.concat(data, axis=1, ignore_index=True).T
    data.to_pickle(path)
    return data


if __name__ == '__main__':
    make_train_set()
