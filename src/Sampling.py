import torch
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

import utils
from config import config

class Sampling():
    # Algorithm 2 in the DDPM paper 
    def __init__(self, betas, img_channels= 3):

        self.betas              = betas
        self.alphas             = 1 - betas
        self.alphas_bar         = torch.cumprod(self.alphas, axis= 0)
        self.alphas_bar_prev    = torch.cat([torch.ones(1,), self.alphas_bar])[:-1]
        self.posterior_variance = betas * (1-self.alphas_bar_prev)/(1-self.alphas_bar)

        self.alphas_recip_sqrt  = (1/self.alphas)**0.5
        self.alphas_bar_sqrt    = self.alphas_bar**0.5
        self.alphas_bar_sqrt_1  = (1 - self.alphas_bar)**0.5

        self.img_channels       = img_channels

    def sample_timestep(self, model, x, t):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        with torch.no_grad():
        # no gradients while sampling else goes out of memory
            betas_t     = utils.get_index_from_list(self.betas, t, x.shape)
            alphas1_t   = utils.get_index_from_list(self.alphas_bar_sqrt_1, t, x.shape)
            alphas2_t   = utils.get_index_from_list(self.alphas_recip_sqrt, t, x.shape)

            # Call model (current image - noise prediction)
            predicted_noise         = model(x, t) # predicted noise from current sample x
            x_prev                  = alphas2_t * (x - betas_t * predicted_noise/alphas1_t) # betas = 1- alphas

            posterior_variance_t    = utils.get_index_from_list(self.posterior_variance, t, x.shape)

            if t == 0: # returning the image without any noise at t=0
                return x_prev
            else: # add noise according to the forward process
                noise = torch.randn_like(x)
                return x_prev + torch.sqrt(posterior_variance_t) * noise

    def sample_plot_image(self, model, save_path):
        ### saves a plot with denoising steps. epoch is used to name the file
        with torch.no_grad():
            # Sample noise
            img_size    = config['img_size'][0]
            img         = torch.randn((1, self.img_channels, img_size, img_size), device= config['device']) # starting with noise

            #plt.figure(figsize=(15,15))
            plt.axis('off')
            num_images  = 10
            stepsize    = int(config['timesteps']/num_images)

            for i in range(0, config['timesteps'])[::-1]:
                t   = torch.full((1,), i, device= config['device'], dtype=torch.long)
                img = self.sample_timestep(model, img, t)
                # Edit: This is to maintain the natural range of the distribution
                img = torch.clamp(img, -1.0, 1.0)
                if i % stepsize == 0:
                    plt.subplot(1, num_images, int(i/stepsize)+1)
                    img_t   = img * 0.5 + 0.5 # denormalize
                    img_t   = img_t[0].permute(1, 2, 0)
                    plt.imshow(img_t.detach().cpu().numpy())
                    plt.axis('off')
            plt.savefig(save_path)