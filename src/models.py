import torch
import math

class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, positional_emb_dim, upsample= False):
        super().__init__()

        self.time_mlp =  torch.nn.Linear(positional_emb_dim, out_channels)

        if upsample:
            self.conv1      = torch.nn.Conv2d(2*in_channels, out_channels, kernel_size= 3, padding=1)
            self.transform  = torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size= 4, stride= 2, padding= 1)
        else: # downsampling
            self.conv1      = torch.nn.Conv2d(in_channels, out_channels, kernel_size= 3, padding=1)
            self.transform  = torch.nn.Conv2d(out_channels, out_channels, kernel_size= 4, stride= 2, padding= 1)

        self.conv2  = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bnorm1 = torch.nn.BatchNorm2d(out_channels)
        self.bnorm2 = torch.nn.BatchNorm2d(out_channels)
        self.relu   = torch.nn.ReLU()

    def forward(self, x, t):
        h           = self.bnorm1(self.relu(self.conv1(x)))
        time_emb    = self.relu(self.time_mlp(t))

        time_emb    = time_emb.reshape(time_emb.shape[0], time_emb.shape[1], 1, 1)

        h           = h + time_emb
        h           = self.bnorm2(self.relu(self.conv2(h)))

        return self.transform(h)


class SinusoidalPositionEmbeddings(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device      = time.device
        half_dim    = self.dim // 2
        embeddings  = math.log(10000) / (half_dim - 1)
        embeddings  = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings  = time[:, None] * embeddings[None, :]
        embeddings  = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class SimpleUnet(torch.nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self,
                 img_channels   = 3,
                 tim_emb_dim    = 32,
                 channels       = [64, 128, 256, 512]):

        super().__init__()
        down_channels   = channels
        up_channels     = list(reversed(down_channels))
        time_emb_dim    = 32

        # Time embedding
        self.time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            torch.nn.Linear(time_emb_dim, time_emb_dim),
            torch.nn.ReLU()
        )

        # Initial projection
        self.conv0      = torch.nn.Conv2d(img_channels, down_channels[0], 3, padding=1)


        self.encoder    = torch.nn.Sequential(
            *[Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels) - 1)]
        )

        # Upsample
        self.decoder    = torch.nn.Sequential(
            *[Block(up_channels[i], up_channels[i+1], time_emb_dim, upsample= True) for i in range(len(up_channels) - 1)]
        )

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output     = torch.nn.Conv2d(up_channels[-1], img_channels, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.encoder:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.decoder:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        out = self.output(x)

        return out