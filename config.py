import torch

config  = dict(
    device          = 'cuda' if torch.cuda.is_available() else 'cpu',

    data_path       = '/content/CelebA200k/', # Uses pytorch MNIST if None
    batch_size      = 512,

    img_size        = (32, 32),
    img_channels    = 3,

    timesteps       = 300,
    noise_schedule  = 'Linear',
    noise_limits    = [0.0001, 0.02],

    epochs          = 500,
    experiment_dir  = '/content/gdrive/MyDrive/Projects/DM_Expts/'
)