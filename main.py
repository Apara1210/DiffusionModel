import torch
from torch.optim import optimizer
import torchvision

import os
import gc
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import config
from dataset import ImageDataset
import utils

from src import models, NoiseSchedules, ForwardDiffusion, Sampling

def main():
    print("Inside main function")
    print("Device: ", config['device'])

    img_channels            = config['img_channels']

    data_transforms         = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config['img_size']),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean= [0.5]*img_channels, std= [0.5]*img_channels) # normalize in range [-1, 1]
    ])

    if config['data_path'] == None:
        dataset = torchvision.datasets.MNIST(
            root        = "/content/",
            download    = True,
            transform   = data_transforms
        )
    else: # own data
        dataset = ImageDataset(
            data_root   = config['data_path'], 
            transforms  = data_transforms
        )
    
    dataloader      = torch.utils.data.DataLoader(
        dataset     = dataset,
        batch_size  = config['batch_size'],
        shuffle     = True,
        pin_memory  = True
    )

    print("\nNumber of samples in dataset         : ", len(dataset))
    print("Batch size: {}, Number of batches    :  {}".format(config['batch_size'], len(dataloader)))

    utils.data_visualization(
        dataloader  = dataloader, 
        save_path   = os.path.join(config['experiment_dir'], 'dataviz.png'))
    print("Data visualization saved at '{}'".format(config['experiment_dir']))

    model   = models.SimpleUnet(
        img_channels    = img_channels,
        channels        = [64, 128, 256, 512, 1024]
    ).to(config['device'])
    
    print("Num params: {:.4f}M".format(sum(p.numel() for p in model.parameters())/1000000))

    print("Setting up {} noise schedule for {} timesteps"
    .format(config['noise_schedule'], config['timesteps']))
    betas       = NoiseSchedules.get_noise_schedule(
        timesteps   = config['timesteps'],
        start       = config['noise_limits'][0],
        end         = config['noise_limits'][1],
        schedule    = config['noise_schedule']
    )
    
    diffusion   = ForwardDiffusion.ForwardDiffusion(betas= betas)
    sampler     = Sampling.Sampling(betas= betas, img_channels= img_channels)

    optimizer   = torch.optim.SGD(
        params          = model.parameters(), 
        lr              = 0.1,
        momentum        = 0.9,
        weight_decay    = 1e-4)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer   = optimizer,
        mode        = 'min',
        factor      = 0.75,
        patience    = 5,
        min_lr      = 5e-5,
        verbose     = True
    )
    criterion   = torch.nn.L1Loss()
    scaler      = torch.cuda.amp.GradScaler()

    logFile     = os.path.join(config['experiment_dir'], '0log.txt')
    f           = open(logFile, 'w')
    f.close()

    utils.logging(logFile, [str(config), str(model), str(optimizer), str(criterion)])

    print("\n Training started...")
    epochs  = config['epochs']

    best_loss   = 1e9
    for epoch in range(epochs):
        total_loss  = 0
        batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

        for batch_idx, data in enumerate(dataloader):

            optimizer.zero_grad()

            t   = torch.randint(0, config['timesteps'], (data[0].shape[0], ), device= config['device']).long()

            x_noisy, noise  = diffusion.add_noise(data[0], t)

            with torch.cuda.amp.autocast():
                pred_noise      = model(x_noisy, t)

            loss            = criterion(noise, pred_noise)
            total_loss      += loss.item()

            scaler.scale(loss).backward() # This is a replacement for loss.backward()
            scaler.step(optimizer) # This is a replacement for optimizer.step()
            scaler.update()

            batch_bar.set_postfix(
                loss="{:.04f}".format(float(total_loss / (batch_idx + 1))),
                iters="{}".format(batch_idx + 1 + epoch*len(dataloader)))
            batch_bar.update()

        batch_bar.close()
        total_loss  /= len(dataloader)

        utils.logging(logFile,
        ["Epoch {}: Iters over: {}, Loss: {:.4f}, Lr: {:.06f}".
        format(epoch+1, len(dataloader)*(epoch+1), total_loss, optimizer.param_groups[0]['lr'])])

        if total_loss < best_loss:
            best_loss  = total_loss
            utils.save_model(
                save_path   = os.path.join(config['experiment_dir'], 'model.pth'),
                model       = model,
                optimizer   = optimizer,
                scheduler   = scheduler,
                loss        = total_loss,
                epoch       = epoch
            )
            utils.logging(logFile, ['Best model saved'])
        
        if epoch%5 == 0:
            sampler.sample_plot_image(
                model, save_path= os.path.join(config['experiment_dir'], 'Epoch_{:04d}.png'.format(epoch+1)))

        scheduler.step(total_loss)

if __name__ == "__main__":
    main()