#%%
import matplotlib.pyplot as plt
import torch
from torch import nn
import sys
import os
import numpy as np
import argparse
import glob

sys.path.insert(1, '../Utils/')     # In case we run from Experiments/Training
import Unet_conditioned as Unet
import Plot
import Diffusion_conditioned as Diffusion
import loader_conditioned as loader
import cfg
from numpy.random import default_rng

#%% 

parser = argparse.ArgumentParser("Diffusion on CelebA dataset with U-net.")
parser.add_argument("-n", "--num", help="Number of training data", type=int)
parser.add_argument("-i", "--index", help="Index for the dataset (0 or 1)", type=int)
parser.add_argument("-s", "--img_size", help="Size of the images to use", type=int)
parser.add_argument("-LR", "--learning_rate", help="Learning rate for optimization", type=float)
parser.add_argument("-O", "--optim", help="Optimisation type (SGD_Momentum or Adam)", type=str)
parser.add_argument("-W", "--nbase", help="Number of base filters", type=str)
parser.add_argument("-t", "--time", help="Diffusion timestep", type=int)
parser.add_argument("-bal", "--balanced", help="Use balanced sampling (1) or random/unbalanced (0)", type=int, default=1)
args = vars(parser.parse_args())
print(args)

# Get arguments
n = args['num']
index = args['index']
size = args['img_size']
lr = args['learning_rate']
optim = args['optim']
n_base = int(args['nbase'])
time_step = args['time']
balanced = args['balanced']
if time_step == -1:
    mode = 'normal'
else:
    mode = 'fixed_time'

# Overwrite config with command line arguments
DATASET = 'ISIC' # Changed to ISIC
config = cfg.load_config(DATASET)
config.DATASET = 'ISIC_Conditioned' # Update dataset name to reflect conditioning
config.path_data = '../data/ISIC/' # Path to the preprocessed ISIC images
config.path_metadata = '../data/metadata_combined.csv' # <--- NEW: Add this line! Point it to your CSV.
config.IMG_SHAPE = (3, size, size) # Changed 1 to 3 for RGB channels
config.n_images = n
config.BATCH_SIZE = min(512, n)
config.OPTIM = optim
config.LR = lr
config.mode = mode
config.time_step = time_step
config.BALANCED = bool(balanced) # Store the flag in the config object

if config.mode == 'normal':
    suffix = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}_bal{:d}/'.format(config.DATASET, size,
                                        config.n_images, n_base, config.OPTIM, config.BATCH_SIZE,
                                        config.LR, index, int(config.BALANCED))
elif config.mode == 'fixed_time':
    suffix = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}_t{:d}_bal{:d}/'.format(config.DATASET, size,
                                        config.n_images, n_base, config.OPTIM, config.BATCH_SIZE,
                                        config.LR, index, time_step, int(config.BALANCED))
    print('Training at fixed diffusion time: {:d}'.format(config.time_step))

# Create path to images and model save
path_images = config.path_save + suffix + 'Images/'
path_models = config.path_save + suffix + 'Models/'
os.makedirs(path_images, exist_ok=True)
os.makedirs(path_models, exist_ok=True)

os.system('cp run_Unet.py {:s}'.format(path_models + '_run_Unet.py'))
os.system('cp ../Utils/loader.py {:s}'.format(path_models + '_loader.py'))
os.system('cp ../Utils/cfg.py {:s}'.format(path_models + '_cfg.py'))

# Raw images version
# loading_func = 'loader.load_{:s}(config, index={:d})'.format(config.DATASET, index)
# testset = None
# trainset, testset = eval(loading_func)

# # Test to put the full trainset on the device
# train_images = torch.zeros(size=(config.n_images, config.IMG_SHAPE[0], config.IMG_SHAPE[1], config.IMG_SHAPE[2]))
# for i in np.arange(config.n_images):
#     train_images[i, :, :] = trainset[i]
# train_images = train_images.to(config.DEVICE)

# Load raw ISIC images directly using our new loader
trainset, testset = loader.load_ISIC(config, loadtest=False, index=index)

# Convert the trainset subset into a full tensor (matching the expected format)
print("Loading images to tensor...")
train_images = torch.zeros(size=(config.n_images, config.IMG_SHAPE[0], config.IMG_SHAPE[1], config.IMG_SHAPE[2]))

train_labels = torch.zeros(size=(config.n_images,), dtype=torch.long)
for i in range(len(trainset)):
    img, label = trainset[i] # Unpack the tuple from our new loader
    train_images[i] = img
    train_labels[i] = label
train_images = train_images.to(config.DEVICE)
train_labels = train_labels.to(config.DEVICE)
print("Images loaded!")

if __name__ == '__main__':
    # Wrap both tensors in a TensorDataset
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=True)

# del trainset
# In[] Plot one random batch of training images

dataiter = iter(trainloader)
images, labels = next(dataiter)

Plot.imshow(images[0:32].cpu(), config.mean, config.std)
plt.savefig(path_images + 'Training_set.pdf', 
            bbox_inches='tight')

# In[] Model definition

if __name__ == '__main__':
    model = Unet.UNet(
        input_channels          = config.IMG_SHAPE[0],
        output_channels         = config.IMG_SHAPE[0],
        base_channels           = n_base,
        base_channels_multiples = (1, 2, 4),
        apply_attention         = (False, True, True),
        dropout_rate            = 0.1,
        num_classes               = 7
    )
    
    # Resume training from last weights in the folder
    weights_files = glob.glob(os.path.join(path_models + 'Model_' + '*'))
    if weights_files:   # If exist, use it
        offset = max([int(f.split('_')[-1]) for f in weights_files])
    else:               # If not, start from 0
        offset = 0
    
    if offset > 0:
        path_checkpoint = config.path_save + '/{:s}/Models/Model_{:d}'.format(suffix, offset)
        model = loader.load_model(model, path_checkpoint)
        model.to(config.DEVICE)
            
    model = nn.DataParallel(model)
    model.to(config.DEVICE)

if __name__ == '__main__':
    n_params = sum(p.numel() for p in model.parameters())
    print('{:.2f}M'.format(n_params/1e6))

# In[] Training and saving

if __name__ == '__main__':
    if config.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    elif config.OPTIM == 'SGD_Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.LR, momentum=0.95)
        
    df = Diffusion.DiffusionConfig(
        n_steps                 = config.TIMESTEPS,
        img_shape               = config.IMG_SHAPE,
        device                  = config.DEVICE,
    )
    loss_fn = nn.MSELoss()
    
    sweeping = 1.0
    
    # Saving times for the model during training
    times_save = cfg.get_training_times()
    
    Diffusion.train(model, trainloader, optimizer, config, df, 
                    loss_fn, sweeping, times_save, offset, suffix, generate=True)