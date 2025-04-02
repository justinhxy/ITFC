import torch
from torchvision import transforms
import numpy as np
import skimage.transform as skitransform

class nonan():  # Python3默认继承object类
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        nan_mask = np.isnan(data)
        data[nan_mask] = 0.0
        data = np.expand_dims(data, axis=0)
        data /= np.max(data)
        return data  # 返回预处理后的图像


class numpy2torch():  # Python3默认继承object类
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        data = torch.tensor(data)
        return data  # 返回预处理后的图像


class resize():  # Python3默认继承object类
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        data = skitransform.resize(data, output_shape=(96, 128, 96), order=1)
        return data  # 返回预处理后的图像


myTransform = {
    'trainTransform': transforms.Compose([
        resize(),
        nonan(),
        numpy2torch(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'testTransform': transforms.Compose([
        resize(),
        nonan(),
        numpy2torch(),
        transforms.Normalize([0.5], [0.5])
    ]),
}
