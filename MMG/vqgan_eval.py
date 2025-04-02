from transform import myTransform
import nibabel as nib
from config import config
import torch
import os
import cv2 as cv
import numpy as np



def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行环境

    source_path = "./test_eval"  # 图像文件夹路径
    recon_output_path = "./recon"

    model = torch.load("2024-10-20-e200-2vqgan.pth").to(device).eval()

    with torch.no_grad():
        for filename in os.listdir(source_path):
            img_path = os.path.join(source_path, filename)
            nii_img = nib.load(img_path)
            img = nii_img.get_fdata()
            img = myTransform["testTransform"](img).to(device)
            img = torch.unsqueeze(img, dim=0).to(device).float()
            if config.save_fig:
                ori = np.array(img.detach().to("cpu"))  # BCHW
                ori = np.squeeze(ori)  # HW
                ori = ori * 0.5 + 0.5
                ori = np.clip(ori, 0, 1)

                vis1 = ori[ori.shape[0] // 2]
                vis1 = np.flip(np.rot90(vis1, k=1, axes=(0, 1)), axis=1)

                vis2 = ori[:, ori.shape[1] // 2]
                vis2 = np.rot90(vis2, k=1, axes=(0, 1))

                vis3 = ori[:, :, ori.shape[2] // 2]
                vis3 = np.rot90(vis3, k=1, axes=(0, 1))

                vis1 = np.clip(vis1, 0, 1)
                vis2 = np.clip(vis2, 0, 1)
                vis3 = np.clip(vis3, 0, 1)
                vis1 *= 255
                vis2 *= 255
                vis3 *= 255
                cv.imwrite("ovis1.png", vis1)
                cv.imwrite("ovis2.png", vis2)
                cv.imwrite("ovis3.png", vis3)

            recon, _ = model(img)

            recon = np.array(recon.detach().to("cpu"))  # BCHW
            recon = np.squeeze(recon)  # HW
            recon = recon * 0.5 + 0.5
            recon = np.clip(recon, 0, 1)
            if not config.use_server:
                vis1 = recon[recon.shape[0] // 2]
                vis1 = np.flip(np.rot90(vis1, k=1, axes=(0, 1)), axis=1)

                vis2 = recon[:, recon.shape[1] // 2]
                vis2 = np.rot90(vis2, k=1, axes=(0, 1))

                vis3 = recon[:, :, recon.shape[2] // 2]
                vis3 = np.rot90(vis3, k=1, axes=(0, 1))

                cv.imshow("win1", vis1)
                cv.imshow("win2", vis2)
                cv.imshow("win3", vis3)
                cv.waitKey(0)
                if config.save_fig:
                    vis1 = np.clip(vis1, 0, 1)
                    vis2 = np.clip(vis2, 0, 1)
                    vis3 = np.clip(vis3, 0, 1)
                    vis1 *= 255
                    vis2 *= 255
                    vis3 *= 255
                    cv.imwrite("rvis1.png", vis1)
                    cv.imwrite("rvis2.png", vis2)
                    cv.imwrite("rvis3.png", vis3)

            nii_save = nib.Nifti1Image(recon, np.eye(4))
            nib.save(nii_save, os.path.join(recon_output_path, filename))

            if config.output_feature_map:
                compress = model.encode_stage_2_inputs(img).cpu().detach().numpy()
                print(np.mean(compress))
                compress = np.squeeze(compress)[0]
                compress = compress * 0.5 + 0.5
                compress = np.clip(compress, 0, 1)
                if not config.use_server:
                    vis1 = compress[compress.shape[0] // 2]
                    vis2 = compress[:, compress.shape[1] // 2]
                    vis2 = np.rot90(vis2, k=1, axes=(0, 1))
                    vis3 = compress[:, :, compress.shape[2] // 2]

                    cv.imshow("win1", vis1)
                    cv.imshow("win2", vis2)
                    cv.imshow("win3", vis3)
                    cv.waitKey(0)
                if config.save_fig:
                    vis1 = np.clip(vis1, 0, 1)
                    vis2 = np.clip(vis2, 0, 1)
                    vis3 = np.clip(vis3, 0, 1)
                    vis1 *= 255
                    vis2 *= 255
                    vis3 *= 255
                    cv.imwrite("cvis1.png", vis1)
                    cv.imwrite("cvis2.png", vis2)
                    cv.imwrite("cvis3.png", vis3)



if __name__ == "__main__":
    eval()
