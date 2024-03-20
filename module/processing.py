import os

import numpy as np
import pandas as pd
import skimage.io
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from modelscope.pipelines import pipeline
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Model(object):

    def __init__(self, inputFile):
        self.NoBackgroundImage = None
        self.rgbImage = None
        self.maskImage = None
        self.img = None
        self.inputFile = inputFile
        self.imageFileName = os.path.basename(self.inputFile)
        self.spectral_curve_image = None

    def get_file_extension(self):
        """
        获取文件后缀名
        :return: file_extension
        """
        file_name, file_extension = os.path.splitext(self.inputFile)
        return file_extension

    def get_file_Narray(self):
        if self.get_file_extension() == '.tif':
            self.img = skimage.io.imread(self.inputFile)

    # 根据模型不同获取预处理数据
    def getNdarray(self, modelType, methodType):
        if modelType == 2:
            return process_RGBNdarray(self.inputFile)
        elif modelType == 1:
            self.get_file_Narray()
            return process_Ndarry(methodType, self.img)
        else:
            return None

    # 获取RGB图像
    def getRGBImage(self):
        '''
        ## 还原RGB
        '''
        # 选择波段索引
        red_band = 111
        green_band = 76
        blue_band = 41
        # 提取三原色图像
        red_image = self.img[:, :, red_band]
        green_image = self.img[:, :, green_band]
        blue_image = self.img[:, :, blue_band]
        # 将三个通道组合成RGB图像
        rgb_img = np.stack([red_image, green_image, blue_image], axis=-1)
        # 修改维度
        rgb_img = np.reshape(rgb_img, (red_image.shape[0], red_image.shape[1], 3))
        rgb_img = (rgb_img - np.min(rgb_img)) * (255 / (np.max(rgb_img) - np.min(rgb_img)))
        rgb_img = np.round(rgb_img).astype(np.uint8)
        # 保存合成的RGB图像
        self.rgbImage = f"./result/rgb_image/{self.imageFileName}"
        plt.savefig(self.rgbImage)

    # 获取掩膜图像和去除背景后的数据
    def getMaskImage(self):
        """
        ## 调用分割模型
        """
        p = pipeline('shop-segmentation', 'damo/cv_vitb16_segmentation_shop-seg')
        mask = p(self.rgbImage)
        mask = mask['masks']

        # 掩膜图像
        self.maskImage = f"./result/mask_image/{self.imageFileName}"
        plt.savefig(self.maskImage)
        # 掩膜乘以原图像，得到去掉背景的数据
        Mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        masked_img = Mask * self.img
        self.NoBackgroundImage = f"./result/no_background_image/{self.imageFileName}"
        plt.savefig(self.NoBackgroundImage)

    # 获取所有图像
    def getImage(self):
        self.getRGBImage()
        self.getMaskImage()
        self.spectral_curve_image = spectra_plot(self.img, (100, 100), self.imageFileName)
        return self.rgbImage, self.maskImage, self.NoBackgroundImage, self.spectral_curve_image

    # 删除临时图像
    def deleteImage(self):
        os.remove(self.rgbImage)
        os.remove(self.NoBackgroundImage)
        os.remove(self.maskImage)
        os.remove(self.spectral_curve_image)


# 根据图片文件获取图像数据矩阵
def get_imageNdarray(imageFilePath):
    input_image = Image.open(imageFilePath).convert("RGB")
    return input_image


# 模型预测前必要的图像处理
def process_RGBNdarray(input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    img_chw = preprocess(input_image)
    _, h, w = img_chw.shape
    img_chw = img_chw.reshape(1, 3, h, w)
    print(img_chw.shape)
    return img_chw  # chw:channel height width


# 高光谱数据预处理
def process_Ndarry(method, data):
    if method == "0":
        return nodo(data)
    elif method == "1":
        return MMS(data)
    elif method == "2":
        return SS(data)
    elif method == "3":
        return CT(data)
    elif method == "4":
        return SNV(data)
    elif method == "5":
        return MA(data)
    elif method == "6":
        return SG(data)
    elif method == "7":
        return D1(data)
    elif method == "8":
        return D2(data)
    elif method == "9":
        return DT(data)
    elif method == "10":
        return DT2(data)
    else:
        return None


# 预处理方式
# 什么都不做
def nodo(data):
    return data


# 最大最小值归一化
def MMS(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MinMaxScaler :(n_samples, n_features)
        """
    return MinMaxScaler().fit_transform(data)


# 标准差标准化
def SS(data):
    """
        :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after StandScaler :(n_samples, n_features)
    """
    return StandardScaler().fit_transform(data)


# 均值中心化
def CT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MeanScaler :(n_samples, n_features)
    """
    for i in range(data.shape[0]):
        MEAN = np.mean(data[i])
        data[i] = data[i] - MEAN
    return data


# 标准正态变换
def SNV(data):
    """
        :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after SNV :(n_samples, n_features)
    """
    data = MMS(data)
    m = data.shape[0]
    n = data.shape[1]
    # 求标准差
    data_std = np.std(data, axis=1)  # 每条光谱的标准差
    # 求平均值
    data_average = np.mean(data, axis=1)  # 每条光谱的平均值
    # SNV计算
    data_snv = [[((data[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)]
    return np.array(data_snv)


# 移动平均平滑
def MA(data, WSZ=5):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :param WSZ: int
       :return: data after MA :(n_samples, n_features)
    """
    for i in range(data.shape[0]):
        out0 = np.convolve(data[i], np.ones(WSZ, dtype=int), 'valid') / WSZ  # WSZ是窗口宽度，是奇数
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(data[i, :WSZ - 1])[::2] / r
        stop = (np.cumsum(data[i, :-WSZ:-1])[::2] / r)[::-1]
        data[i] = np.concatenate((start, out0, stop))
    return data


# Savitzky-Golay平滑滤波
def SG(data, w=11, p=2):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :param w: int
       :param p: int
       :return: data after SG :(n_samples, n_features)
    """
    return signal.savgol_filter(data, w, p)


# 一阶导数
def D1(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after First derivative :(n_samples, n_features)
    """
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return Di


# 二阶导数
def D2(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after second derivative :(n_samples, n_features)
    """
    # data = deepcopy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    temp2 = (pd.DataFrame(data)).diff(axis=1)
    temp3 = np.delete(temp2.values, 0, axis=1)
    temp4 = (pd.DataFrame(temp3)).diff(axis=1)
    spec_D2 = np.delete(temp4.values, 0, axis=1)
    return spec_D2


# 趋势校正(DT)
def DT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after DT :(n_samples, n_features)
    """
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


# 改进的趋势校正，时间复杂度更小
def DT2(data):
    x = np.asarray(range(0, data.shape[1]), dtype=np.float32).reshape(1, -1)
    l = LinearRegression()
    l.fit(x.T, data.T)
    trends = l.predict(x.T).T
    return data - trends


# 沿波段方向绘制某一像素波谱曲线
def spectra_plot(img, position, imageFileName):
    x, y = position
    # 提取光谱数据
    spectra = img[x, y, :].reshape(img.shape[2])

    # 创建波段索引
    wavelengths = np.arange(0, img.shape[2])

    # 绘制光谱曲线
    plt.plot(wavelengths, spectra)
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title(f'Spectral Curve of ({x},{y})')
    fileName = f"./result/spectral_curve_image/{imageFileName}"
    plt.savefig(fileName)
    return fileName
