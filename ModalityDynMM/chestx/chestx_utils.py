import sys 
import os 
sys.path.append(os.getcwd())
sys.path.append('/home/bianca/Code/MultiBench/mm_health_bench/mmhb')

import torch
from torch.utils.data import DataLoader, Dataset
# from multi_health_bench.mmhb.loader import ChestXDataset
# from multi_health_bench.mmhb.utils import Config

class TextOnlyChestX(Dataset):
    def __init__(self, chestx_dataset):
        self.chestx_dataset = chestx_dataset
    
    def __getitem__(self, idx):
        (_, report_tensor), target = self.chestx_dataset[idx]

        return [report_tensor, target]
    
    def __len__(self):
        return len(self.chestx_dataset)



class ImageOnlyChestX(Dataset):
    def __init__(self, chestx_dataset):
        self.chestx_dataset = chestx_dataset
    
    def __getitem__(self, idx):
        (image_tensor, _), target = self.chestx_dataset[idx]

        return [image_tensor, target]
    
    def __len__(self):
        return len(self.chestx_dataset)


class RandomChestXDataset(Dataset):
    def __init__(self, num_samples=1000, max_seq_length=256, image_size=(256, 256, 3), num_labels=2):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            max_seq_length (int): Length of the token sequence.
            image_size (tuple): Shape of the image (H, W, C).
            num_labels (int): Number of binary labels.
        """
        self.num_samples = num_samples
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        self.num_labels = num_labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate a tokenized text sequence (e.g., from BERT) with random integers.
        # BERT's vocab size is around 30522, so we use that as an upper bound.
        text = torch.randint(0, 30522, (self.max_seq_length,))
        
        # Create a random image with shape (256, 256, 3)
        image = torch.randn(*self.image_size)
        image_nchw = image.permute(2, 0, 1)
        # Create a random multi-label target with 14 binary values.
        label = torch.randint(0, 2, (self.num_labels,))
        
        # Return a tuple: (text modality, image modality, label)
        return text, image_nchw, label
