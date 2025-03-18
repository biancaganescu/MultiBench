import sys 
import os 
sys.path.append(os.getcwd())
sys.path.append('/home/bianca/Code/MultiBench/mm_health_bench/mmhb')
from torch.utils.data import random_split
import torch
from torch.utils.data import DataLoader, Dataset
from mm_health_bench.mmhb.loader import ChestXDataset
from mm_health_bench.mmhb.utils import Config

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


def chestx_collate_fn(batch):
    """
    Each batch element is a tuple: ((image, report), target)
    where report is currently a 1D tensor of shape (s,).
    We want to convert each report to shape (s, 1)
    so that the batched reports have shape (batch, s, 1).
    """
    images = [sample[0][0].permute(2, 0, 1) for sample in batch]
    reports = [sample[0][1] for sample in batch]
    targets = [sample[1] for sample in batch]
    
    # Process each report: if it is 1D, unsqueeze to add a feature dimension.
    # processed_reports = []
    # for rep in reports:
    #     if rep.dim() == 1:
    #         rep = rep.unsqueeze(0)
    #     processed_reports.append(rep)
    
    images = torch.stack(images, dim=0)         # images: (batch, H, W, C)
    reports = torch.stack(reports, dim=0)  # reports: (batch, s)
    targets = torch.tensor(targets, dtype=torch.float) # targets: (batch, num_labels)
    return reports, images, targets


def get_data(batch_size=32, num_workers=4):

    config = Config("./mm_health_bench/config/config.yml").read()

    chestx_dataset = ChestXDataset(data_path="./mm_health_bench/data/chestx", max_seq_length=256)

    print(chestx_dataset[0][1])

    total_length = len(chestx_dataset)
    train_length = int(0.8 * total_length)
    val_length = int(0.1 * total_length)
    test_length = total_length - train_length - val_length


    train_dataset, val_dataset, test_dataset = random_split(chestx_dataset, [train_length, val_length, test_length])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=chestx_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=chestx_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=chestx_collate_fn)

    return train_loader, val_loader, test_loader


