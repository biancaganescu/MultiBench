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

def get_data(batch_size=32, num_workers=4):

    config = Config("mm-health-bench/config/config.yml").read()

    train_dataset = ChestXDataset(data_path="data/chestx", split="train", max_seq_length=256)
    val_dataset = ChestXDataset(data_path="data/chestx", split="val", max_seq_length=256)
    test_dataset = ChestXDataset(data_path="data/chestx", split="test", max_seq_length=256)

    # text_train_dataset = TextOnlyChestX(train_dataset)
    # text_val_dataset = TextOnlyChestX(val_dataset)
    # text_test_dataset = TextOnlyChestX(test_dataset)


    # image_train_dataset = ImageOnlyChestX(train_dataset)
    # image_val_dataset = ImageOnlyChestX(val_dataset)
    # image_test_dataset = ImageOnlyChestX(test_dataset)


    # text_train_loader = DataLoader(text_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # text_val_loader = DataLoader(text_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # text_test_loader = DataLoader(text_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # image_train_loader = DataLoader(image_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # image_val_loader = DataLoader(image_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # image_test_loader = DataLoader(image_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # return text_train_loader, text_val_loader, text_test_loader, image_train_loader, image_val_loader, image_test_loader


    rain_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def generate_random_data(batch_size=32, num_workers=0):
    # Define sample sizes for train, validation, and test splits.
    train_samples = 3000
    val_samples = 200
    test_samples = 200

    # Create dataset instances for each split.
    train_dataset = RandomChestXDataset(num_samples=train_samples, num_labels=2)
    val_dataset   = RandomChestXDataset(num_samples=val_samples, num_labels=2)
    test_dataset  = RandomChestXDataset(num_samples=test_samples, num_labels=2)

    # Create DataLoaders directly from the datasets.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader