from torch.utils.data import Dataset
import pandas as pd
import cv2 as cv
import os
import torch
import numpy as np
import nibabel as nib


class myDataset(Dataset):
    def __init__(self, filelist, mri_dir, pet_dir, transform1=None, transform2=None):
        self.mri_dir = mri_dir
        self.pet_dir = pet_dir
        self.transform1 = transform1
        self.transform2 = transform2
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        mri_path = self.mri_dir
        pet_path = self.pet_dir

        file = self.filelist.iloc[idx, 0]  # file name

        nii_mri = nib.load(os.path.join(mri_path, file))
        nii_pet = nib.load(os.path.join(pet_path, file[:-3]))

        mri = nii_mri.get_fdata()
        pet = nii_pet.get_fdata()

        if self.transform1:
            mri = self.transform1(mri)
        if self.transform2:
            pet = self.transform2(pet)

        return mri, pet


class mySingleDataset(Dataset):
    def __init__(self, filelist, data_dir, transform1=None):
        self.data_dir = data_dir
        self.transform1 = transform1
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        data_path = self.data_dir

        file = self.filelist.iloc[idx, 0]  # file name

        if self.data_dir.endswith("PET"):
            file=file[:-3]

        nii_data = nib.load(os.path.join(data_path, file))

        data = nii_data.get_fdata()

        if self.transform1:
            data = self.transform1(data)

        return data
class myMetricsDataset(Dataset):
    def __init__(self, filelist, mri_dir, transform1=None):
        self.mri_dir = mri_dir
        self.transform1 = transform1
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        mri_path = self.mri_dir

        file = self.filelist.iloc[idx, 0]  # file name

        nii_mri = nib.load(os.path.join(mri_path, file))

        mri = nii_mri.get_fdata()

        if self.transform1:
            mri = self.transform1(mri)

        if file.endswith(".gz"):
            return mri, file[:-3]
        else:
            return mri, file

