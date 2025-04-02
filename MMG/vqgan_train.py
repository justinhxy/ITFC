import torch
import os
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler
from datetime import date
import time

from transform import myTransform
from dataset import myDataset
from model import vqgan

# for reproducibility purposes set a seed
set_determinism(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

adni1_trainset_list = "adni1_trainset.txt"
adni1_testset_list = "adni1_valset.txt"
adni2_trainset_list = "adni2_trainset.txt"
adni2_testset_list = "adni2_valset.txt"

adni1_mri_path = "/mntcephfs/med_dataset/huxiangyang/ADNI/ADNI1/MRI"
adni1_pet_path = "/mntcephfs/med_dataset/huxiangyang/ADNI/ADNI1/PET"
adni2_mri_path = "/mntcephfs/med_dataset/huxiangyang/ADNI/ADNI2/MRI"
adni2_pet_path = "/mntcephfs/med_dataset/huxiangyang/ADNI/ADNI2/PET"

myTrainSet = myDataset(adni1_trainset_list, adni1_mri_path, adni1_pet_path, myTransform['trainTransform'],
                       myTransform['trainTransform']) + myDataset(adni2_trainset_list, adni2_mri_path,
                                                                  adni2_pet_path, myTransform['trainTransform'],
                                                                  myTransform['trainTransform'])
myTestSet = myDataset(adni1_testset_list, adni1_mri_path, adni1_pet_path, myTransform['testTransform'],
                      myTransform['testTransform']) + myDataset(adni2_testset_list, adni2_mri_path, adni2_pet_path,
                                                                myTransform['testTransform'],
                                                                myTransform['testTransform'])
myTrainLoader = DataLoader(myTrainSet, batch_size=16, shuffle=True)
myTestLoader = DataLoader(myTestSet, batch_size=16, shuffle=True)

autoencoder = vqgan.to(device)

discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)
discriminator.to(device)

l1_loss = L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")
loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
loss_perceptual.to(device)

adv_weight = 0.01
perceptual_weight = 0.001

optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)

n_epochs = 100
autoencoder_warm_up_n_epochs = 5
val_interval = 10
epoch_recon_loss_list = []
epoch_gen_loss_list = []
epoch_disc_loss_list = []
val_recon_epoch_loss_list = []
total_start = time.time()

for epoch in range(n_epochs):
    autoencoder.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    progress_bar = tqdm(enumerate(myTrainLoader), total=len(myTrainLoader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        mri = batch[0].to(device=device, non_blocking=True).float()
        pet = batch[1].to(device=device, non_blocking=True).float()

        # Generator part
        optimizer_g.zero_grad(set_to_none=True)
        reconstruction, quantization_loss = autoencoder(mri)

        recons_loss = l1_loss(reconstruction.float(), pet.float())
        p_loss = loss_perceptual(reconstruction.float(), pet.float())
        loss_g = recons_loss + quantization_loss + perceptual_weight * p_loss

        if epoch > autoencoder_warm_up_n_epochs:
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()

        if epoch > autoencoder_warm_up_n_epochs:
            # Discriminator part
            optimizer_d.zero_grad(set_to_none=True)
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(pet.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = adv_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

        epoch_loss += recons_loss.item()
        if epoch > autoencoder_warm_up_n_epochs:
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )
    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
    epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        autoencoder.eval()
        discriminator.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(myTestLoader, start=1):
                mri = batch[0].to(device=device, non_blocking=True).float()
                pet = batch[1].to(device=device, non_blocking=True).float()

                reconstruction, quantization_loss = autoencoder(mri)
                recons_loss = l1_loss(reconstruction.float(), pet.float())

                val_loss += recons_loss.item()

        val_loss /= val_step
        val_recon_epoch_loss_list.append(val_loss)

        torch.save(autoencoder, str(date.today()) + "-e" + str(n_epochs) + "-vqgan.pth")

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")

plt.style.use("ggplot")
plt.title("Learning Curves", fontsize=20)
plt.plot(epoch_recon_loss_list)
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_recon_loss_list, color="C0",
         linewidth=2.0,
         label="Train")
plt.plot(
    np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
    val_recon_epoch_loss_list,
    color="C1",
    linewidth=2.0,
    label="Validation",
)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig("vae Learning.png")

plt.title("Adversarial Training Curves", fontsize=20)
plt.plot(epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
plt.plot(epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig("vae Adversarial.png")
