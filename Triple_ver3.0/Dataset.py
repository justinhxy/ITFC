import nibabel as nib
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from skimage import transform as skt
import numpy as np

def get_clinical(sub_id, clin_df):
    '''Gets clinical features vector by searching dataframe for image id'''
    # 用-1初始化数组，表示缺失值
    clinical = np.full(9, -1.0)

    if sub_id in clin_df["PTID"].values:
        row = clin_df.loc[clin_df["PTID"] == sub_id].iloc[0]

        # GENDER (1表示Male, 2表示Female，缺失默认设置为 -1)
        if pd.isnull(row["PTGENDER"]):
            clinical[0] = -1
        else:
            clinical[0] = 1 if row["PTGENDER"] == 1 else (2 if row["PTGENDER"] == 2 else 0)

        # AGE (用-1标记缺失值)
        clinical[1] = row["AGE"] if not pd.isnull(row["AGE"]) else 0

        # Education (用-1标记缺失值)
        clinical[2] = row["PTEDUCAT"] if not pd.isnull(row["PTEDUCAT"]) else 0

        # FDG_bl (用-1标记缺失值)
        clinical[3] = row["FDG_bl"] if not pd.isnull(row["FDG_bl"]) else 0

        # TAU_bl (用-1标记缺失值)
        clinical[4] = row["TAU_bl"] if not pd.isnull(row["TAU_bl"]) else 0

        # PTAU_bl (用-1标记缺失值)
        clinical[5] = row["PTAU_bl"] if not pd.isnull(row["PTAU_bl"]) else 0

        # APOE4 (保留原有处理方式，缺失则处理为 -1)
        apoe4_allele = row["APOE4"]
        if pd.isnull(apoe4_allele):
            clinical[6], clinical[7], clinical[8] = 0, 0, 0  # 标记缺失值
        elif apoe4_allele == 0:
            clinical[6], clinical[7], clinical[8] = 1, 0, 0
        elif apoe4_allele == 1:
            clinical[6], clinical[7], clinical[8] = 0, 1, 0
        elif apoe4_allele == 2:
            clinical[6], clinical[7], clinical[8] = 0, 0, 1

    return clinical


class NoNan:  # Python3默认继承object类
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        nan_mask = np.isnan(data)
        data[nan_mask] = 0.0
        data = np.expand_dims(data, axis=0)
        data /= np.max(data)
        return data  # 返回预处理后的图像


class Numpy2Torch:  # Python3默认继承object类
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        data = torch.from_numpy(data)
        return data  # 返回预处理后的图像


class Resize:  # Python3默认继承object类
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        data = skt.resize(data, output_shape=(96, 128, 96), order=1)
        return data  # 返回预处理后的图像

# 自定义 Dataset 类来处理 MRI 和 PET 数据
class MriPetDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, cli_dir, csv_file, valid_group=("AD", "CN")):
        """
        Args:
            mri_dir (string or Path): MRI 文件所在的文件夹路径。
            pet_dir (string or Path): PET 文件所在的文件夹路径。
            cli_dir (string or Path): Clinical 文件所在的文件夹路径。
            csv_file (string or Path): CSV 文件路径，其中第一列是文件名，第二列是标签。
            transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.mri_dir = Path(mri_dir)
        if pet_dir  == '':
            self.pet_dir = ''
        else:
            self.pet_dir = Path(pet_dir)
        self.cli_dir = pd.read_csv(cli_dir)
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.groups = {'DM': 1, 'AD': 1, 'CN': 0, 'pMCI': 1, 'sMCI': 0, 'sSCD': 0, 'pSCD': 1,
                       'MCI': 1, 'sSMC': 0, 'pSMC': 1, 'SMC': 0, 'sCN': 0,
                       'pCN': 1, 'ppCN': 1, 'Autism': 1, 'Control': 0}
        self.valid_group = valid_group
        self.transform = transforms.Compose([
            Resize(),
            NoNan(),
            Numpy2Torch(),
            transforms.Normalize([0.5], [0.5])
        ])

        # 过滤只保留 valid_group 中的有效数据
        self.filtered_indices = self.labels_df[self.labels_df.iloc[:, 1].isin(self.valid_group)].index.tolist()

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # 获取过滤后的索引
        filtered_idx = self.filtered_indices[idx]

        # 获取对应的文件名和标签
        img_name = self.labels_df.iloc[filtered_idx, 0]
        label_str = self.labels_df.iloc[filtered_idx, 1]  # 标签

        # MRI 文件路径
        mri_img_path = self.mri_dir / (img_name + '.nii')
        mri_img_numpy = nib.load(str(mri_img_path)).get_fdata()
        mri_img_torch = self.transform(mri_img_numpy)
        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1

        clinical_features = get_clinical(img_name, self.cli_dir)
        clin_tab_torch = torch.from_numpy(clinical_features).float()

        # 只有MRI,没有PET,用于eval阶段
        if self.pet_dir == '':
            return mri_img_torch.float(), label
        else:
            # PET 文件路径
            pet_img_path = self.pet_dir / (img_name + '.nii')
            # print('pet_img_path', pet_img_path)
            pet_img_numpy = nib.load(str(pet_img_path)).get_fdata()
            pet_img_torch = self.transform(pet_img_numpy)
            return mri_img_torch.float(), pet_img_torch.float(), clin_tab_torch,label

if __name__ == '__main__':
    mri_dir = r'D:\dataset\final\ADNI1\MRI\skull-bet'  # 替换为 MRI 文件的路径
    pet_dir = r'D:\dataset\final\ADNI1\PET'  # 替换为 PET 文件的路径
    cli_dir = r'C:\Users\admin\Desktop\ADNI_Clinical.csv'
    csv_file = 'ADNI1_match.csv'  # 替换为 CSV 文件路径
    batch_size = 8  # 设置批次大小

    dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file, valid_group=("pMCI", "sMCI"))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 测试读取数据
    print('dataloader', len(dataloader))
    print('dataset', len(dataloader.dataset))
    for i, (mri_imgs, pet_imgs, cli_tab, labels) in enumerate(dataloader):
        print(f"{i} MRI Images batch shape: {mri_imgs.shape}")
        print(f"{i} PET Images batch shape: {pet_imgs.shape}")
        print(f"{i} Clinical Table batch shape: {cli_tab.shape}")
        print(f"{i} Labels batch shape: {labels}")
