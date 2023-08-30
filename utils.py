import torch
import matplotlib.pyplot as plt

def data_visualization(dataloader, save_path):
    ### for vizualizing images
    batch_imgs  = next(iter(dataloader))[0]
    num_samples = min(20, batch_imgs.shape[0])
    plt.figure(figsize=(10,10))
    for i in range(num_samples):
        plt.subplot(int(num_samples/4) + 1, 4, i + 1)
        img = batch_imgs[i].permute(1, 2, 0) # selecting image
        img = (img+1)/2 # denormalize from [-1, 1] to [0, 1] for displaying
        plt.axis('off')
        plt.imshow(img)
    plt.savefig(save_path)

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def logging(txt_file, statements):
    f   = open(txt_file, 'a')
    for s in statements:
        print(s)
        f.write(s + '\n')
    f.close()

def save_model(save_path, model, optimizer, scheduler, loss, epoch):
    torch.save({
        'model_state_dict'      : model.state_dict(),
        'optimizer_state_dict'  : optimizer.state_dict(),
        'scheduler_state_dict'  : None if scheduler == None else scheduler.state_dict(),
        'loss'                  : loss,
        'epoch'                 : epoch
    }, save_path)

def load_model(ckpt_path, model, optimizer, scheduler):
    checkpoint  = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch       = checkpoint['epoch']
    loss        = checkpoint['loss']
    if scheduler is None:
        return model, optimizer, loss, epoch
    else: 
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return model, optimizer, scheduler, loss, epoch