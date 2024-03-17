import numpy as np
import pandas as pd
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def nodo(data):
    return data


def MMS(data):
    return MinMaxScaler().fit_transform(data)


def SS(data):
    return StandardScaler().fit_transform(data)


def CT(data):
    for i in range(data.shape[0]):
        MEAN = np.mean(data[i])
        data[i] = data[i] - MEAN
    return data


def SNV(data):
    data = MMS(data)
    m = data.shape[0]
    n = data.shape[1]
    data_std = np.std(data, axis=1)
    data_average = np.mean(data, axis=1)
    data_snv = [[((data[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)]
    return np.array(data_snv)


def MA(data, WSZ=5):
    for i in range(data.shape[0]):
        out0 = np.convolve(data[i], np.ones(WSZ, dtype=int), 'valid') / WSZ
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(data[i, :WSZ - 1])[::2] / r
        stop = (np.cumsum(data[i, :-WSZ:-1])[::2] / r)[::-1]
        data[i] = np.concatenate((start, out0, stop))
    return data


def SG(data, w=11, p=2):
    return signal.savgol_filter(data, w, p)


def D1(data):
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return Di


def D2(data):
    temp2 = (pd.DataFrame(data)).diff(axis=1)
    temp3 = np.delete(temp2.values, 0, axis=1)
    temp4 = (pd.DataFrame(temp3)).diff(axis=1)
    spec_D2 = np.delete(temp4.values, 0, axis=1)
    return spec_D2


def DT(data):
    x = np.asarray(range(0, 256), dtype=np.float32)
    out = np.array(data)
    l = LinearRegression()
    for i in range(out.shape[0]):
        l.fit(x.reshape(-1, 1), out[i].reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        for j in range(out.shape[1]):
            out[i][j] = out[i][j] - (j * k + b)
    return out


def DT2(data):
    x = np.asarray(range(0, data.shape[1]), dtype=np.float32).reshape(1, -1)
    l = LinearRegression()
    l.fit(x.T, data.T)
    trends = l.predict(x.T).T
    return data - trends
