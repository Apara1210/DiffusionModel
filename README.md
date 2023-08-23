
# Simple Diffusion Model for Image Generation
As the title suggests. The denoising network is a simple U-Net. Sinusoidal positional embeddings are used. References are given below, however I've modified the code structure to better suit my workflow.

### 1. Setting up the environment
Create a conda evnironment and install the requirements. Its better to go to https://pytorch.org/get-started/locally/ to install torch.
```
pip install -r requirements.txt
```

### 2. Setting up data
By default, the dataset is MNIST. You can use your own dataset as well by specifying the path to the folder containing images in **data_path** in **config.py**. The folder structure is as follows.
```
ImageDir
--Image1.ext
--Image2.ext
...
```
Where ext is the extension of the image (use whatever PyTorch supports).

The transforms normalize the images in the range [-1, 1] which is required in a diffusion model to add gaussian noise. Data visualization is done using ```data_visualization()``` from utils which is called in ```main.py```. It saves a grid of the images from the dataset.

### 3. Training
Run ```main.py``` to train the model. Data visualization, log file, best model checkpoint and image logs will be saved in the experiments directory specified in config.
* Diffusion models tend to require a lot of epochs for training
* Try using various data augmentations 

### 4. Results
Training for more epochc at the moment. Will update after completed.

### References
* Denoising Diffusion Probabilistic Models: https://arxiv.org/abs/2006.11239
* Diffusion Models - Math explained: https://youtu.be/HoKDTa5jHvg
* Diffusion Models - PyTorch implementation: https://youtu.be/a4Yfz2FxXiY