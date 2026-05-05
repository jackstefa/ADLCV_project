import numpy as np
import torch
import torchvision.transforms as transforms

# For CelebA
import os
from PIL import Image
from torch.utils.data import Dataset
from natsort import natsorted


# ===================================================================
#   Helper functions for loading datasets
# ===================================================================

import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from natsort import natsorted
import torch
import torchvision.transforms as transforms
import numpy as np

# ===================================================================
#   ISIC Dataset Loading (Modified for Conditioning)
# ===================================================================
class ISICDataset(Dataset):
    def __init__(self, root_dir, metadata_path, transform=None):
        """
        Args:
          root_dir (string): Directory with all the 32x32 ISIC images
          metadata_path (string): Path to the metadata_combined.csv
          transform (callable, optional): transform to be applied
        """
        valid_ext = ('.jpg', '.jpeg', '.png')
        image_names = [f for f in os.listdir(root_dir) if f.lower().endswith(valid_ext)]
        
        self.root_dir = root_dir
        self.transform = transform 
        self.image_names = natsorted(image_names)
        
        # Load metadata and set isic_id as the index for fast lookups
        self.metadata = pd.read_csv(metadata_path)
        self.metadata.set_index('isic_id', inplace=True)
        
        # Mapping Roman Numerals to integers (0 will be our missing/null class)
        self.fitz_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}

    def __len__(self): 
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # Extract isic_id from filename (e.g. 'ISIC_0065973.jpg' -> 'ISIC_0065973')
        isic_id = os.path.splitext(img_name)[0]
        
        # Default label is 0 (null/unconditional class)
        label = 0 
        
        if isic_id in self.metadata.index:
            # Handle cases where multiple rows might exist for the same ID or missing values
            fitz_val = self.metadata.loc[isic_id, 'fitzpatrick_skin_type']
            if isinstance(fitz_val, pd.Series): 
                fitz_val = fitz_val.iloc[0] 
                
            if isinstance(fitz_val, str) and fitz_val in self.fitz_map:
                label = self.fitz_map[fitz_val]
                
        return img, label

def load_ISIC(config, loadtest=False, ntest=2048, index=0):
    '''
    Loads ISIC dataset keeping 3 RGB channels, perfectly balanced across 
    the 6 Fitzpatrick skin types for conditional DDPM training.
    '''
    metadata_path = config.path_metadata
    
    # Notice we removed Resize and CenterCrop here since the images are already 32x32.
    transform = transforms.Compose([
         transforms.ToTensor()
    ])
    
    isic_dataset = ISICDataset(config.path_data, metadata_path, transform)
    
    # ==========================================
    # NEW: BALANCED SAMPLING LOGIC
    # ==========================================
    print("Balancing dataset across Fitzpatrick classes...")
    rng = np.random.default_rng(seed=42) # Fixed seed for reproducibility
    
    # 1. Quickly extract all labels without loading the heavy image files
    all_labels = []
    for i in range(len(isic_dataset)):
        img_name = isic_dataset.image_names[i]
        isic_id = os.path.splitext(img_name)[0]
        
        label = 0 # Default null class
        if isic_id in isic_dataset.metadata.index:
            fitz_val = isic_dataset.metadata.loc[isic_id, 'fitzpatrick_skin_type']
            if isinstance(fitz_val, pd.Series): 
                fitz_val = fitz_val.iloc[0]
            if isinstance(fitz_val, str) and fitz_val in isic_dataset.fitz_map:
                label = isic_dataset.fitz_map[fitz_val]
        all_labels.append(label)
        
    all_labels = np.array(all_labels)
    
    # 2. Separate indices by valid skin type (classes 1 through 6)
    # We ignore class 0 here so the model explicitly learns the 6 skin tones.
    class_indices = {c: np.where(all_labels == c)[0] for c in range(1, 7)}
    
    # 3. Calculate how many images to draw per class
    n_classes = 6
    n_per_class = config.n_images // n_classes
    remainder = config.n_images % n_classes # Handle sizes that don't divide perfectly by 6
    
    subset_indices = []
    for c in range(1, 7):
        # Add 1 extra image to the first few classes if there's a remainder
        draw_count = n_per_class + (1 if c <= remainder else 0)
        
        available_indices = class_indices[c]
        if len(available_indices) < draw_count:
            raise ValueError(f"Not enough images for class {c}. Needed {draw_count}, found {len(available_indices)}.")
        
        # Sample without replacement
        sampled = rng.choice(available_indices, size=draw_count, replace=False)
        subset_indices.extend(sampled)
        
    # 4. Shuffle the final combined indices so the batches are mixed
    rng.shuffle(subset_indices)
    
    # Create the training subset
    trainset = torch.utils.data.Subset(isic_dataset, subset_indices)
    testset = None
    
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([1.0, 1.0, 1.0])
    
    if config.CENTER:
        tmploader = torch.utils.data.DataLoader(trainset, batch_size=min(len(trainset), 2000),
                                                  shuffle=False, num_workers=0)
        # tmploader now returns (images, labels)
        t_data, _ = next(iter(tmploader)) 
        
        mean = torch.mean(t_data, axis=[0, 2, 3])
        if config.STANDARDIZE:
            std = torch.std(t_data, axis=[0, 2, 3])
        
        transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean, std)
        ])
        
        # Reload data
        isic_dataset = ISICDataset(config.path_data, metadata_path, transform)
        trainset = torch.utils.data.Subset(isic_dataset, subset_indices)
        
        if loadtest:
            indices_test = np.arange(-ntest, 0)
            testset = torch.utils.data.Subset(isic_dataset, indices_test)
        
    config.mean = mean
    config.std = std

    # ==========================================
    # NEW: Print Class Distribution
    # ==========================================
    print("\nAnalyzing class distribution in the loaded subset...")
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    
    # We iterate through the subset to count the labels
    for i in range(len(trainset)):
        img, label = trainset[i]
        label_counts[label] += 1
        
    print("--- Fitzpatrick Skin Type Distribution ---")
    print(f"Null/Unclassified (0): {label_counts[0]}")
    print(f"Type I (1):            {label_counts[1]}")
    print(f"Type II (2):           {label_counts[2]}")
    print(f"Type III (3):          {label_counts[3]}")
    print(f"Type IV (4):           {label_counts[4]}")
    print(f"Type V (5):            {label_counts[5]}")
    print(f"Type VI (6):           {label_counts[6]}")
    print(f"Total Images:          {len(trainset)}")
    print("------------------------------------------\n")
    
    return trainset, testset

# Create a custom Dataset class
class CelebADataset(Dataset):
  def __init__(self, root_dir, transform=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Get image names
    image_names = os.listdir(root_dir)

    self.root_dir = root_dir
    self.transform = transform 
    self.image_names = natsorted(image_names)
    self.attr = np.loadtxt(root_dir+'/../list_attr_celeba.txt', skiprows=2,
                           usecols=np.arange(1, 41))

  def __len__(self): 
    return len(self.image_names)

  def __getitem__(self, idx):
    # Get the path to the image 
    img_path = os.path.join(self.root_dir, self.image_names[idx])
    # Load image and convert it to RGB
    img = Image.open(img_path).convert('RGB')
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)
      
    # label = self.attr[idx, 20]  # Male or fem
    # if label == -1:
    #     label = 0

    return img#, label



# ===================================================================
#   Loading the datasets
# ===================================================================
def load_CelebA(config, loadtest=False, ntest=2048, index=0):
    '''
    Parameters
    ----------
    config : class Diffusion.TrainingConfig()
        Contains all the training information
    loadtest : TYPE, optional
        Whether or not to load a test set. The default is False.
    ntest : int, optional
        Number of test images to load. The default is 2048.
    index : int, optional
        Index of the subset to load. The default is 0.

    Returns
    -------
    trainset : torchvision.datasets
        Subset of the training set containing config.n_images for each class.
    testset : torchvision.datasets
        Subset of the training set containing test images for each class.
    '''
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(config.IMG_SHAPE[1]),
         transforms.CenterCrop(config.IMG_SHAPE[1]),
         transforms.Grayscale(1),
         ])
    
    celeba_dataset = CelebADataset(config.path_data, transform)
    
    indices = np.arange(index*config.n_images, (index+1)*config.n_images) # Load images between index and index+1 times the number of data
    trainset = torch.utils.data.Subset(celeba_dataset, indices)
    testset = None
    
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([1.0, 1.0, 1.0])
    if config.CENTER:
        tmploader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                                  shuffle=False, num_workers=1)
        t_data = next(iter(tmploader))
        
        mean = torch.mean(t_data, axis=[0, 2, 3])
        if config.STANDARDIZE:
            std = torch.std(t_data, axis=[0, 2, 3])
        
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(config.IMG_SHAPE[1]),
             transforms.CenterCrop(config.IMG_SHAPE[1]),
             transforms.Normalize(mean, std),
             transforms.Grayscale(1),
             ])
        
        # Reload data
        celeba_dataset = CelebADataset(config.path_data, transform)
        trainset = torch.utils.data.Subset(celeba_dataset, indices)
        
        if loadtest:
            # indices_test = np.arange(config.n_images, config.n_images + ntest)
            indices_test = np.arange(-ntest, 0)
            testset = torch.utils.data.Subset(celeba_dataset, indices_test)
        
    # Store mean and std
    config.mean = mean
    config.std = std
    
    return trainset, testset


class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x = self.base[idx]
        if self.transform:
            x = self.transform(x)
        return x

def load_CelebA_pt(config, full_tensor, loadtest=False, ntest=2048, index=0):
    '''
    Parameters
    ----------
    config : class Diffusion.TrainingConfig()
        Contains all the training information
    full_tensor : torch.Tensor
        Tensor containing the full dataset.
    loadtest : TYPE, optional
        Whether or not to load a test set. The default is False.
    ntest : int, optional
        Number of test images to load. The default is 2048.
    index : int, optional
        Index of the subset to load. The default is 0.

    Returns
    -------
    trainset : Transformed tensor of 
        Subset of the training set containing config.n_images images.
    testset : torchvision.datasets
        Subset of the training set containing ntest test images.
    '''
    
    # Load training and test sets
    train_images = full_tensor[index*config.n_images:(index+1)*config.n_images]
    test_images = None
    if loadtest:
        test_images = full_tensor[-ntest:]
    
    # Center and standardize the data
    mean = torch.zeros(config.IMG_SHAPE[0])
    std = torch.ones(config.IMG_SHAPE[0])
    if config.CENTER:
        mean = torch.mean(train_images, axis=[0, 2, 3])
        if config.STANDARDIZE:
            std = torch.std(train_images, axis=[0, 2, 3])
        
        transform = transforms.Compose(
            [transforms.Normalize(mean, std),])
        
        # Transform the trainset and testset
        train = TransformedDataset(train_images, transform=transform)
        test = None
        if loadtest:
            test = TransformedDataset(test_images, transform=transform)
        
    # Store mean and std
    config.mean = mean
    config.std = std
    
    return train, test
    



# ===================================================================
#   Loading the model
# ===================================================================
def load_model(model: torch.nn.Module, path_checkpoint: str, verbose: bool = True):
    state_dict = torch.load(path_checkpoint, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # Remove 'module.'
        new_state_dict[k] = v
        
    model.load_state_dict(new_state_dict)
    if verbose:
        print('Loading initial state at {:s}'.format(path_checkpoint))
    return model