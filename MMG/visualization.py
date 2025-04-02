import nibabel as nib
import numpy as np
import torch
import cv2 as cv
import skimage.transform as skitransform

# 示例文件路径
file_path = '116_S_0834.nii'
nii_img = nib.load(file_path)
# 获取图像数据
data = nii_img.get_fdata()
print(data.shape)

data = skitransform.resize(data, output_shape=(96, 128, 96), order=1)

nan_mask = np.isnan(data)
data[nan_mask] = 0.0
data /= np.max(data)
vis1 = data[data.shape[0] // 2]
vis1 =np.flip(np.rot90(vis1, k=1, axes=(0, 1)), axis=1)
vis2 = data[:, data.shape[1] // 2]
vis2 =np.rot90(vis2, k=1, axes=(0, 1))
vis3 = data[:, :, data.shape[2] // 2]
vis3 =np.rot90(vis3, k=1, axes=(0, 1))

cv.imshow("1", vis1)
cv.imshow("2", vis2)
cv.imshow("3", vis3)

cv.waitKey(0)


nii_image = nib.Nifti1Image(data, np.eye(4))

nib.save(nii_image, 'output.nii')

file_path = 'output.nii'
nii_img = nib.load(file_path)
# 获取图像数据
data = nii_img.get_fdata()
print(data.shape)
nan_mask = np.isnan(data)
data[nan_mask] = 0.0
data /= np.max(data)
vis1 = data[data.shape[0] // 2]
vis1 =np.flip(np.rot90(vis1, k=1, axes=(0, 1)), axis=1)
vis2 = data[:, data.shape[1] // 2]
vis2 =np.rot90(vis2, k=1, axes=(0, 1))
vis3 = data[:, :, data.shape[2] // 2]
vis3 =np.rot90(vis3, k=1, axes=(0, 1))

cv.imshow("1", vis1)
cv.imshow("2", vis2)
cv.imshow("3", vis3)

cv.waitKey(0)