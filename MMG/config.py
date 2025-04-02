from dataclasses import dataclass
import torch


@dataclass
class config():
    use_server = True
    output_feature_map = True
    save_fig = True

    # dm
    batch_size = 4
    n_epochs = 200
    offset_noise = True
    offset_noise_coefficient = 0.1
