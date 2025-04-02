from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator, VQVAE
import torch

autoencoder = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 64),
    latent_channels=3,
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True),
)
unet = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=8,
    out_channels=8,
    num_res_blocks=2,
    num_channels=(32, 64, 128, 256),
    attention_levels=(False, True, True, True),
    num_head_channels=(0, 64, 128, 256),
)
vae = VQVAE(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 256),
    num_res_channels=256,
    num_res_layers=2,
    downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
    upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
    num_embeddings=1024,
    embedding_dim=4,
)
vqgan = VQVAE(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(256, 512),
    num_res_channels=512,
    num_res_layers=2,
    downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
    upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
    num_embeddings=1024,
    embedding_dim=32,
)
if __name__ == "__main__":
    print("Number of model parameters:", sum([p.numel() for p in vqgan.parameters()]))
    a = torch.randn([4, 1, 96, 128, 96])
    b,_ = vqgan(a)
