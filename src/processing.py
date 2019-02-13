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
    feat["mean"] = df_x.mean()
    # feat["std"] = df_x.std()
    # feat["skew"] = df_x.skew()
    # feat["kurt"] = df_x.kurt()

    feat[f"count_std5_1"] = (((df_x - 4.5) / 5).abs() > 1).sum()

    mean_abs = (df_x - feat["mean"]).abs()
    feat["mean_abs_med"] = mean_abs.median()

    roll_std = df_x.rolling(375).std().dropna()
    feat["std_roll_med_375"] = roll_std.median()

    half = len(roll_std) // 2
    feat["std_roll_half1"] = roll_std.iloc[:half].median()
    feat["std_roll_half2"] = roll_std.iloc[-half:].median()

    # feat["hurst"] = nolds.hurst_rs(df_x.values)

    # Не нравится мне это, но дает очень похожий инкрементальный результат на паблике и кросс-валидации
    # Возможный плюс, что welch с дефолтными настройками - попыки их подрихтовать не дают улучшений
    welch = signal.welch(df_x)[1]
    for num in [2, 3, 14, 28, 30]:
        feat[f"welch_{num}"] = welch[num]

    # New
    xc = pd.Series(df_x.values)
    zc = np.fft.fft(xc)

    feat['mean'] = xc.mean()
    feat['std'] = xc.std()
    feat['max'] = xc.max()
    feat['min'] = xc.min()

    # FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    feat['Rmean'] = realFFT.mean()
    feat['Rstd'] = realFFT.std()
    feat['Rmax'] = realFFT.max()
    feat['Rmin'] = realFFT.min()
    feat['Imean'] = imagFFT.mean()
    feat['Istd'] = imagFFT.std()
    feat['Imax'] = imagFFT.max()
    feat['Imin'] = imagFFT.min()
    feat['Rmean_last_5000'] = realFFT[-5000:].mean()
    feat['Rstd__last_5000'] = realFFT[-5000:].std()
    feat['Rmax_last_5000'] = realFFT[-5000:].max()
    feat['Rmin_last_5000'] = realFFT[-5000:].min()
    feat['Rmean_last_15000'] = realFFT[-15000:].mean()
    feat['Rstd_last_15000'] = realFFT[-15000:].std()
    feat['Rmax_last_15000'] = realFFT[-15000:].max()
    feat['Rmin_last_15000'] = realFFT[-15000:].min()

    feat['mean_change_abs'] = np.mean(np.diff(xc))
    feat['mean_change_rate'] = np.mean(np.nonzero((np.diff(xc) / xc[:-1]))[0])
    feat['abs_max'] = np.abs(xc).max()
    feat['abs_min'] = np.abs(xc).min()

    feat['std_first_50000'] = xc[:50000].std()
    feat['std_last_50000'] = xc[-50000:].std()
    feat['std_first_10000'] = xc[:10000].std()
    feat['std_last_10000'] = xc[-10000:].std()

    feat['avg_first_50000'] = xc[:50000].mean()
    feat['avg_last_50000'] = xc[-50000:].mean()
    feat['avg_first_10000'] = xc[:10000].mean()
    feat['avg_last_10000'] = xc[-10000:].mean()

    feat['min_first_50000'] = xc[:50000].min()
    feat['min_last_50000'] = xc[-50000:].min()
    feat['min_first_10000'] = xc[:10000].min()
    feat['min_last_10000'] = xc[-10000:].min()

    feat['max_first_50000'] = xc[:50000].max()
    feat['max_last_50000'] = xc[-50000:].max()
    feat['max_first_10000'] = xc[:10000].max()
    feat['max_last_10000'] = xc[-10000:].max()

    feat['max_to_min'] = xc.max() / np.abs(xc.min())
    feat['max_to_min_diff'] = xc.max() - np.abs(xc.min())
    feat['count_big'] = len(xc[np.abs(xc) > 500])
    feat['sum'] = xc.sum()

    feat['mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(xc[:50000]) / xc[:50000][:-1]))[0])
    feat['mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(xc[-50000:]) / xc[-50000:][:-1]))[0])
    feat['mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(xc[:10000]) / xc[:10000][:-1]))[0])
    feat['mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(xc[-10000:]) / xc[-10000:][:-1]))[0])

    feat['q95'] = np.quantile(xc, 0.95)
    feat['q99'] = np.quantile(xc, 0.99)
    feat['q05'] = np.quantile(xc, 0.05)
    feat['q01'] = np.quantile(xc, 0.01)

    feat['abs_q95'] = np.quantile(np.abs(xc), 0.95)
    feat['abs_q99'] = np.quantile(np.abs(xc), 0.99)
    feat['abs_q05'] = np.quantile(np.abs(xc), 0.05)
    feat['abs_q01'] = np.quantile(np.abs(xc), 0.01)

    feat['trend'] = add_trend_feature(xc)
    feat['abs_trend'] = add_trend_feature(xc, abs_values=True)
    feat['abs_mean'] = np.abs(xc).mean()
    feat['abs_std'] = np.abs(xc).std()

    feat['mad'] = xc.mad()
    feat['kurt'] = xc.kurtosis()
    feat['skew'] = xc.skew()
    feat['med'] = xc.median()

    feat['Hilbert_mean'] = np.abs(hilbert(xc)).mean()
    feat['Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()
    feat['classic_sta_lta1_mean'] = classic_sta_lta(xc, 500, 10000).mean()
    feat['classic_sta_lta2_mean'] = classic_sta_lta(xc, 5000, 100000).mean()
    feat['classic_sta_lta3_mean'] = classic_sta_lta(xc, 3333, 6666).mean()
    feat['classic_sta_lta4_mean'] = classic_sta_lta(xc, 10000, 25000).mean()
    feat['Moving_average_700_mean'] = xc.rolling(window=700).mean().mean(skipna=True)
    feat['Moving_average_1500_mean'] = xc.rolling(window=1500).mean().mean(skipna=True)
    feat['Moving_average_3000_mean'] = xc.rolling(window=3000).mean().mean(skipna=True)
    feat['Moving_average_6000_mean'] = xc.rolling(window=6000).mean().mean(skipna=True)
    ewma = pd.Series.ewm
    feat['exp_Moving_average_300_mean'] = (ewma(xc, span=300).mean()).mean(skipna=True)
    feat['exp_Moving_average_3000_mean'] = ewma(xc, span=3000).mean().mean(skipna=True)
    feat['exp_Moving_average_30000_mean'] = ewma(xc, span=6000).mean().mean(skipna=True)
    no_of_std = 2
    feat['MA_700MA_std_mean'] = xc.rolling(window=700).std().mean()
    feat['MA_700MA_BB_high_mean'] = (
            feat['Moving_average_700_mean'] + no_of_std * feat['MA_700MA_std_mean']).mean()
    feat['MA_700MA_BB_low_mean'] = (
            feat['Moving_average_700_mean'] - no_of_std * feat['MA_700MA_std_mean']).mean()
    feat['MA_400MA_std_mean'] = xc.rolling(window=400).std().mean()
    feat['MA_400MA_BB_high_mean'] = (
            feat['Moving_average_700_mean'] + no_of_std * feat['MA_400MA_std_mean']).mean()
    feat['MA_400MA_BB_low_mean'] = (
            feat['Moving_average_700_mean'] - no_of_std * feat['MA_400MA_std_mean']).mean()
    feat['MA_1000MA_std_mean'] = xc.rolling(window=1000).std().mean()

    feat['iqr'] = np.subtract(*np.percentile(xc, [75, 25]))
    feat['q999'] = np.quantile(xc, 0.999)
    feat['q001'] = np.quantile(xc, 0.001)
    feat['ave10'] = stats.trim_mean(xc, 0.1)

    for windows in [10, 100, 1000]:
        x_roll_std = xc.rolling(windows).std().dropna().values
        x_roll_mean = xc.rolling(windows).mean().dropna().values

        feat['ave_roll_std_' + str(windows)] = x_roll_std.mean()
        feat['std_roll_std_' + str(windows)] = x_roll_std.std()
        feat['max_roll_std_' + str(windows)] = x_roll_std.max()
        feat['min_roll_std_' + str(windows)] = x_roll_std.min()
        feat['q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        feat['q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        feat['q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        feat['q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        feat['av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        feat['av_change_rate_roll_std_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        feat['abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        feat['ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        feat['std_roll_mean_' + str(windows)] = x_roll_mean.std()
        feat['max_roll_mean_' + str(windows)] = x_roll_mean.max()
        feat['min_roll_mean_' + str(windows)] = x_roll_mean.min()
        feat['q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        feat['q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        feat['q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        feat['q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        feat['av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        feat['av_change_rate_roll_mean_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        feat['abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

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
