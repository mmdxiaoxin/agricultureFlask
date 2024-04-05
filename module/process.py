import io
import os
import cv2

import numpy as np
import pandas as pd
import requests
import skimage.io
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from modelscope.pipelines import pipeline
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from module.preprocessing import nodo, MMS, SS, CT, SNV, MA, SG, D1, D2, DT, DT2


class Model(object):

    def __init__(self, inputFile, imageFileName):
        """
        类内属性
        :param inputFile:输入文件流
        :param imageFileName:输入文件名
        """
        self.rgb_img = None
        self.NoBackgroundImage = None
        self.rgbImage = None
        self.maskImage = None
        self.img = None
        self.inputFile = inputFile
        self.imageFileName = imageFileName
        self.file_name = ""
        self.imageFilePath = ""
        self.spectral_curve_image = None

    def get_file_extension(self):
        """
        获取文件后缀名
        :return:file_extension:文件拓展名
        """
        self.file_name, file_extension = os.path.splitext(self.imageFileName)
        return file_extension

    def get_file_Narray(self):
        image_stream = io.BytesIO(self.inputFile)
        if image_stream:
            received_dirPath = '../result'
            if not os.path.isdir(received_dirPath):
                os.makedirs(received_dirPath)
            self.imageFilePath = os.path.join(received_dirPath, self.imageFileName)
            with open(self.imageFilePath, "wb") as file:
                file.write(image_stream.getvalue())
            if self.get_file_extension() == ".tif":
                self.img = skimage.io.imread(self.imageFilePath)
                os.remove(self.imageFilePath)

    # 根据模型不同获取预处理数据
    def getNdarray(self, modelType, methodType):
        """
        根据模型不同获取预处理数据
        :param modelType: 模型种类
        :param methodType: 预处理方法种类
        :return: 预处理完成矩阵
        """
        if modelType == 2:
            input_image = get_imageNdarray(self.inputFile)
            return process_RGBNdarray(input_image)
        elif modelType == 1:
            self.get_file_Narray()
            # 转换为灰度图像
            if self.img.ndim == 2:
                self.img = self.img[..., np.newaxis]

            # 转换为彩色图像
            if self.img.shape[2] == 1:
                self.img = np.repeat(self.img, 3, axis=-1)

            # 确保 data 的形状是 (height * width, channels)
            # 这里我们假设 channels 是 3，即彩色图像
            reshaped_data = self.img.reshape(-1, self.img.shape[2])
            return process_Ndarry(methodType, reshaped_data)
        else:
            return None

    # 获取所有图像
    def getImage(self):
        self.getRGBImage()
        self.getMaskImage()
        self.spectral_curve_image = spectra_plot(self.img, (0, 0), self.file_name)
        rgb_data = getImageStream(self.rgbImage)
        mask_data = getImageStream(self.maskImage)
        nbg_data = getImageStream(self.NoBackgroundImage)
        sc_data = getImageStream(self.spectral_curve_image)
        return rgb_data, mask_data, nbg_data, sc_data

    # 删除临时图像
    def deleteImage(self):
        os.remove(self.rgbImage)
        os.remove(self.NoBackgroundImage)
        os.remove(self.maskImage)
        os.remove(self.spectral_curve_image)

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
        self.rgb_img = rgb_img
        # print(rgb_img)
        # 保存合成的RGB图像
        self.rgbImage = f"../result/rgb_image.png"
        plt.imshow(rgb_img)
        plt.imsave(self.rgbImage, rgb_img)

    # 获取掩膜图像和去除背景后的数据
    def getMaskImage(self):
        """
        ## 调用分割模型
        """
        # 使用分割管道处理图像
        p = pipeline('shop-segmentation', 'damo/cv_vitb16_segmentation_shop-seg')
        mask = p(self.rgb_img)
        mask = mask['masks']

        # 掩膜图像
        self.maskImage = "../result/mask_image.png"
        plt.imshow(mask)
        plt.imsave(self.maskImage, mask)
        # 掩膜乘以原图像，得到去掉背景的数据
        Mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        masked_img = Mask * self.rgb_img
        self.NoBackgroundImage = "../result/no_background_image.png"
        plt.imshow(masked_img)
        plt.imsave(self.NoBackgroundImage, masked_img)


# 根据文件路径获取二进制文件流
def getImageStream(filePath):
    """
    根据文件路径获取二进制文件流
    :param filePath: 文件所在路径
    :return: 二进制文件流
    """
    with open(filePath, "rb") as file:
        file_content = file.read()
    if file_content:
        return file_content
    else:
        return None


# 根据图片文件获取图像数据矩阵
def get_imageNdarray(imageFile):
    """

    :param imageFile: 二进制文件流
    :return: 图像矩阵
    """
    imageStream = io.BytesIO(imageFile)
    input_image = Image.open(imageStream).convert("RGB")
    return input_image


# 模型预测前必要的图像处理
def process_RGBNdarray(input_image):
    """

    :param input_image:图像矩阵
    :return: 预处理完成数据
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    img_chw = preprocess(input_image)
    _, h, w = img_chw.shape
    img_chw = img_chw.reshape(1, 3, h, w)
    print(img_chw.shape)
    return img_chw  # chw:channel height width


def process_Ndarry(method, data):
    """

    :param method: 预处理方法
    :param data: 图像数据
    :return: 预处理结果
    """
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


# 沿波段方向绘制某一像素波谱曲线
def spectra_plot(img, position, image_file_name):
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
    file_name = "../result/spectral_curve_image.png"
    plt.savefig(file_name)
    return file_name


if __name__ == '__main__':
    with open("D:/work/B_0.tif", "rb") as tif_image_file:
        tif_image = tif_image_file.read()
    tifModel = Model(tif_image, "B_0.tif")

    # 测试读取文件流进行预处理
    tifModel.get_file_Narray()
    # RGB图像生成
    # tifModel.getRGBImage()
    # print(tifModel.rgbImage)

    # Mask图像及去除背景图像
    # tifModel.getMaskImage()

    # 图像生成测试
    rgb_data, mask_data, nbg_data, sc_data = tifModel.getImage()
    print(rgb_data)
    # 获取概率数组
    print(tifModel.getNdarray(1, "1"))
    tifModel.deleteImage()
