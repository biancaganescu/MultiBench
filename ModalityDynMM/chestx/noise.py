import torch
import torch.nn.functional as F
import numpy as np
import random
import sys 
import os 
sys.path.append(os.getcwd())
sys.path.append('/home/bianca/Code/MultiBench/mm_health_bench/mmhb')
from torch.utils.data import random_split, Subset
import torch
from torch.utils.data import DataLoader, Dataset
from mm_health_bench.mmhb.loader import ChestXDataset
from mm_health_bench.mmhb.utils import Config
from torch.utils.data import WeightedRandomSampler, RandomSampler
import pickle
import hashlib
import json
import torchvision
class NoiseAugmenter:
   
    @staticmethod
    def add_gaussian_noise_to_image(image, mean=0.0, std=0.1):
        """
        Add Gaussian noise to images.
        
        Args:
            image (torch.Tensor): Image tensor of shape [batch, channels, height, width]
            mean (float): Mean of the Gaussian noise
            std (float): Standard deviation of the Gaussian noise
            
        Returns:
            torch.Tensor: Noisy image
        """
        noise = torch.randn_like(image) * std + mean
        noisy_image = image + noise
        # Clip to valid image range [0, 1]'
        return torch.clamp(noisy_image, 0, 1)
    
    @staticmethod
    def add_salt_and_pepper_noise(image, amount=0.05):
        """
        Add salt and pepper noise to images.
        
        Args:
            image (torch.Tensor): Image tensor of shape [batch, channels, height, width]
            amount (float): Proportion of the image to be affected by noise
            
        Returns:
            torch.Tensor: Image with salt and pepper noise
        """
        noisy_image = image.clone()
        
        # Salt noise
        salt_mask = torch.rand_like(image) < (amount/2)
        noisy_image[salt_mask] = 1.0
        
        # Pepper noise
        pepper_mask = torch.rand_like(image) < (amount/2)
        noisy_image[pepper_mask] = 0.0
        
        return noisy_image
    
    @staticmethod
    def add_poisson_noise(image, scale=1.0):
        """
        Add Poisson noise to images (simulates photon noise).
        
        Args:
            image (torch.Tensor): Image tensor of shape [batch, channels, height, width]
            scale (float): Scale factor for the Poisson process
            
        Returns:
            torch.Tensor: Image with Poisson noise
        """
        # Scale image to appropriate range for Poisson sampling
        scaled_image = image * 255.0 * scale
        
        # Apply Poisson noise (implementation depends on PyTorch version)
        try:
            # For newer PyTorch versions
            noisy_image = torch.poisson(scaled_image) / (255.0 * scale)
        except:
            # Fallback for older versions
            noisy_image = torch.FloatTensor(np.random.poisson(
                scaled_image.cpu().numpy()) / (255.0 * scale)).to(image.device)
        
        return torch.clamp(noisy_image, 0, 1)
    
    @staticmethod
    def reduce_image_quality(image, blur_factor=1.5, noise_level=0):
        """
        Reduce image quality by applying blur and noise.
        
        Args:
            image (torch.Tensor): Image tensor of shape [batch, channels, height, width]
            blur_factor (float): Amount of Gaussian blur to apply
            noise_level (float): Amount of noise to add
            
        Returns:
            torch.Tensor: Lower quality image
        """
        # Apply Gaussian blur
        kernel_size = int(blur_factor * 2) * 2 + 1  # Ensure odd kernel size
        blurred = torchvision.transforms.functional.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], 
                                 sigma=[blur_factor, blur_factor])
        
        # Add noise
        noisy_blurred = NoiseAugmenter.add_gaussian_noise_to_image(blurred, std=noise_level)
        
        return noisy_blurred
    
    @staticmethod
    def mask_image_regions(image, mask_size=0.2, num_masks=3):
        """
        Randomly mask regions of the image (simulates occlusions or artifacts).
        
        Args:
            image (torch.Tensor): Image tensor of shape [batch, channels, height, width]
            mask_size (float): Size of each mask as a fraction of image size
            num_masks (int): Number of masks to apply
            
        Returns:
            torch.Tensor: Image with masked regions
        """
        masked_image = image.clone()
        batch_size, channels, height, width = image.shape
        
        # Size of mask in pixels
        mask_h = int(height * mask_size)
        mask_w = int(width * mask_size)
        
        for b in range(batch_size):
            for _ in range(num_masks):
                # Random position for mask
                top = random.randint(0, height - mask_h)
                left = random.randint(0, width - mask_w)
                
                # Apply mask (set to mean value of the image)
                mean_value = image[b].mean()
                masked_image[b, :, top:top+mask_h, left:left+mask_w] = mean_value
        
        return masked_image
    
    
    # Text Noise Functions
    
    @staticmethod
    def add_word_dropout(text, dropout_prob=0.3):
        """
        Randomly replace words in text embeddings with zeros (word dropout).
        
        Args:
            text (torch.Tensor): Text tensor of shape [batch, seq_len, embedding_dim]
                                 or [batch, seq_len] for token indices
            dropout_prob (float): Probability of dropping each word
            
        Returns:
            torch.Tensor: Text with some words dropped
        """
        if len(text.shape) == 3:  # For word embeddings [batch, seq_len, embedding_dim]
            mask = torch.rand(text.shape[0], text.shape[1], 1, device=text.device) >= dropout_prob
            return text * mask
        else:  # For token indices [batch, seq_len]
            # Create a mask where 0 tokens (padding) remain 0, and other tokens have dropout applied
            non_padding_mask = (text != 0)
            dropout_mask = (torch.rand_like(text.float()) >= dropout_prob)
            combined_mask = non_padding_mask & dropout_mask
            
            # Apply mask, keeping padding at 0
            result = text * combined_mask.long()
            return result
    
    @staticmethod
    def swap_words(text, swap_prob=0.3):
        """
        Randomly swap adjacent words in text.
        
        Args:
            text (torch.Tensor): Text tensor of shape [batch, seq_len]
            swap_prob (float): Probability of swapping adjacent words
            
        Returns:
            torch.Tensor: text_noise_typeText with swapped words
        """
        result = text.clone()
        batch_size, seq_len = text.shape
        
        for b in range(batch_size):
            for i in range(seq_len - 1):
                # Skip padding tokens
                if text[b, i] == 0 or text[b, i+1] == 0:
                    continue
                    
                # Random swap with probability
                if random.random() < swap_prob:
                    result[b, i], result[b, i+1] = text[b, i+1], text[b, i]
        
        return result
    
    @staticmethod
    def corrupt_text(text, corruption_prob=0.5, pad_token_id=0):
        """
        Corrupt text by randomly replacing tokens with random tokens.
        
        Args:
            text (torch.Tensor): Text tensor of shape [batch, seq_len]
            corruption_prob (float): Probability of corrupting each token
            pad_token_id (int): ID of padding token to avoid replacing
            
        Returns:
            torch.Tensor: Corrupted text
        """
        corrupted = text.clone()
        batch_size, seq_len = text.shape
        
        # Get vocabulary size (maximum token ID + 1)
        vocab_size = torch.max(text) + 1
        
        for b in range(batch_size):
            for i in range(seq_len):
                # Skip padding tokens
                if text[b, i] == pad_token_id:
                    continue
                    
                # Corrupt with probability
                if random.random() < corruption_prob:
                    # Replace with a random token (excluding pad token)
                    random_token = random.randint(1, vocab_size - 1)
                    corrupted[b, i] = random_token
        
        return corrupted



# Generate a unique hash for a corruption configuration
def get_config_hash(config):
    """
    Generate a hash for a corruption configuration to use as a unique identifier.
    
    Args:
        config (dict): Corruption configuration dictionary
        
    Returns:
        str: A hash string representing the configuration
    """
    # Create a deterministic string representation of the config
    config_str = json.dumps(config, sort_keys=True)
    # Create a hash of the config string
    return hashlib.md5(config_str.encode()).hexdigest()


# Create directories for indices if they don't exist
def ensure_indices_dir():
    """Ensure that the directory for storing noise indices exists."""
    indices_dir = os.path.join(os.getcwd(), "noise_indices")
    if not os.path.exists(indices_dir):
        os.makedirs(indices_dir)
    return indices_dir


# Generate or load noise indices for a specific batch size and configuration
def get_noise_indices(dataset_size, batch_size, config, split_name, seed=42):
    """
    Get consistent noise indices for a specific dataset, batch size, and configuration.
    Will load existing indices if available, or generate and save new ones.
    
    Args:
        dataset_size (int): Total size of the dataset
        batch_size (int): Batch size used for data loading
        config (dict): Corruption configuration
        split_name (str): Name of the dataset split ('train', 'val', or 'test')
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of batch indices lists, where each inner list contains indices 
              of samples to apply noise to in that batch
    """
    # Set a seed for reproducibility
    random.seed(seed)
    
    # Get directory for indices
    indices_dir = ensure_indices_dir()
    
    # Generate a unique filename based on the configuration, dataset size, and batch size
    config_hash = get_config_hash(config)
    indices_filename = f"{split_name}_{config_hash}_{dataset_size}_{batch_size}.json"
    indices_path = os.path.join(indices_dir, indices_filename)
    
    # Check if indices already exist
    if os.path.exists(indices_path):
        # Load existing indices
        with open(indices_path, 'r') as f:
            batch_noise_indices = json.load(f)
        print(f"Loaded existing noise indices for {split_name} split from {indices_path}")
        return batch_noise_indices
    
    # Calculate parameters
    image_noise_percentage = config.get("image_noise_percentage", 0)
    text_noise_percentage = config.get("text_noise_percentage", 0)
    
    # Generate new indices
    num_batches = (dataset_size + batch_size - 1) // batch_size  # Ceiling division
    batch_noise_indices = []
    
    for i in range(num_batches):
        # For the last batch, handle potential partial batch
        current_batch_size = min(batch_size, dataset_size - i * batch_size)
        batch_indices = list(range(current_batch_size))
        
        # Generate noise indices for this batch
        batch_dict = {
            "image_indices": [],
            "text_indices": []
        }
        
        # Image noise indices
        if image_noise_percentage > 0:
            num_noisy_images = max(1, int(current_batch_size * image_noise_percentage / 100))
            batch_dict["image_indices"] = sorted(random.sample(batch_indices, num_noisy_images))
        
        # Text noise indices
        if text_noise_percentage > 0:
            num_noisy_texts = max(1, int(current_batch_size * text_noise_percentage / 100))
            batch_dict["text_indices"] = sorted(random.sample(batch_indices, num_noisy_texts))
        
        batch_noise_indices.append(batch_dict)
    
    # Save indices for future use
    with open(indices_path, 'w') as f:
        json.dump(batch_noise_indices, f)
    
    print(f"Generated and saved new noise indices for {split_name} split to {indices_path}")
    return batch_noise_indices


# Modified collate function that uses consistent noise indices
def noisy_chestx_collate_fn(batch, corruption_config=None, batch_idx=0, noise_indices=None):
    """
    Custom collate function that adds noise to chest X-ray images and reports
    based on a corruption configuration and consistent noise indices.
    
    Args:
        batch: Batch of data from ChestXDataset
        corruption_config (dict): Configuration for the corruption
        batch_idx (int): Index of the current batch
        noise_indices (list): List of batch noise indices dictionaries
            
    Returns:
        tuple: (reports, images, targets) with noise applied
    """
    # Apply regular collate function first
    images = [sample[0][0] for sample in batch]
    reports = [sample[0][1] for sample in batch]
    targets = [sample[1] for sample in batch]
    
    images = torch.stack(images, dim=0)
    reports = torch.stack(reports, dim=0)  # (batch, seq)
    targets = torch.tensor(targets, dtype=torch.float)  # (batch, num_labels)
    
    # If no corruption config or noise_indices are None, return as is
    if corruption_config is None or noise_indices is None:
        return reports, images, targets
    
    # Get the current batch's noise indices
    if batch_idx >= len(noise_indices):
        print(f"Warning: batch_idx {batch_idx} out of range for noise_indices of length {len(noise_indices)}")
        return reports, images, targets
    
    current_noise_indices = noise_indices[batch_idx]
    image_indices = current_noise_indices["image_indices"]
    text_indices = current_noise_indices["text_indices"]
    
    # Extract parameters from the corruption config
    image_noise_type = corruption_config.get("image_noise_type")
    text_noise_type = corruption_config.get("text_noise_type")
    image_noise_params = corruption_config.get("image_noise_params", {})
    text_noise_params = corruption_config.get("text_noise_params", {})
    
    # Apply image noise if specified and indices exist
    if image_noise_type and image_indices:
        # Create a noisy version of the entire batch
        if image_noise_type == 'gaussian':
            noisy_images = NoiseAugmenter.add_gaussian_noise_to_image(images, **image_noise_params)
        elif image_noise_type == 'salt_pepper':
            noisy_images = NoiseAugmenter.add_salt_and_pepper_noise(images, **image_noise_params)
        elif image_noise_type == 'poisson':
            noisy_images = NoiseAugmenter.add_poisson_noise(images, **image_noise_params)
        elif image_noise_type == 'quality_reduction':
            noisy_images = NoiseAugmenter.reduce_image_quality(images, **image_noise_params)
        elif image_noise_type == 'masking':
            noisy_images = NoiseAugmenter.mask_image_regions(images, **image_noise_params)
        
        # Apply noise only to selected samples
        for idx in image_indices:
            if idx < len(images):  # Ensure index is valid
                images[idx] = noisy_images[idx]
    
    # Apply text noise if specified and indices exist
    if text_noise_type and text_indices:
        # Create a noisy version of the entire batch
        if text_noise_type == 'dropout':
            noisy_reports = NoiseAugmenter.add_word_dropout(reports, **text_noise_params)
        elif text_noise_type == 'swap':
            noisy_reports = NoiseAugmenter.swap_words(reports, **text_noise_params)
        elif text_noise_type == 'corruption':
            noisy_reports = NoiseAugmenter.corrupt_text(reports, **text_noise_params)
        
        # Apply noise only to selected samples
        for idx in text_indices:
            if idx < len(reports):  # Ensure index is valid
                reports[idx] = noisy_reports[idx]
    
    return reports, images, targets

CORRUPTION =  {
            "name": "corruption_03",
            "image_noise_type": None,
            "text_noise_type": "corruption",
            "image_noise_params": None,
            "text_noise_params": {"corruption_prob": 0.5},
            "image_noise_percentage": 0,
            "text_noise_percentage": 100
        }

class ConsistentNoiseDataLoader(DataLoader):
    """
    DataLoader that applies consistent noise to the same samples across epochs.
    """
    
    def __init__(self, dataset, batch_size, shuffle, corruption_config, split_name, 
                 num_workers=0, collate_fn=None, seed=42, **kwargs):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.corruption_config = corruption_config
        self.split_name = split_name
        self.noise_seed = seed
        
        # Generate or load noise indices
        if corruption_config:
            self.noise_indices = get_noise_indices(
                dataset_size=len(dataset),
                batch_size=batch_size,
                config=corruption_config,
                split_name=split_name,
                seed=seed
            )
        else:
            self.noise_indices = None
        
        # Create a custom collate function that tracks batch index
        self.batch_idx = 0
        
        def noise_tracking_collate(batch):
            if collate_fn:
                result = collate_fn(batch, corruption_config, self.batch_idx, self.noise_indices)
                self.batch_idx = (self.batch_idx + 1) % ((len(dataset) + batch_size - 1) // batch_size)
                return result
            return torch.utils.data.dataloader.default_collate(batch)
        
        # Initialize the parent DataLoader
        super(ConsistentNoiseDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=noise_tracking_collate,
            **kwargs
        )
    
    def __iter__(self):
        # Reset batch index at the start of each iteration
        self.batch_idx = 0
        return super(ConsistentNoiseDataLoader, self).__iter__()


def get_noisy_data_loaders(
    corruption_config=CORRUPTION,
    corruption_name=None,
    indices_path="split_indices.pth",
    data_path="./mm_health_bench/data/chestx",
    batch_size=32,
    num_workers=4,
    apply_to_train=True,
    apply_to_val=False,
    apply_to_test=False,
    max_seq_length=256,
    noise_seed=42
):
    """
    Get data loaders with specified noise applied on-the-fly to images and/or text.
    Uses consistent noise indices across epochs.
    
    Args:
        corruption_config (dict, optional): Configuration for the corruption to apply
        corruption_name (str, optional): Name of a predefined corruption config to use
        indices_path (str): Path to the file containing dataset split indices
        data_path (str): Path to the ChestX dataset
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        apply_to_train (bool): Whether to apply noise to training data
        apply_to_val (bool): Whether to apply noise to validation data
        apply_to_test (bool): Whether to apply noise to test data
        max_seq_length (int): Maximum sequence length for text
        noise_seed (int): Seed for consistent noise generation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader) with noise applied as configured
    """
    # # Resolve the corruption configuration
    # if corruption_name is not None:
    #     if corruption_name not in CORRUPTION_CONFIGS:
    #         raise ValueError(f"Unknown corruption name: {corruption_name}. "
    #                         f"Available configurations: {list(CORRUPTION_CONFIGS.keys())}")
    #     config_to_use = CORRUPTION_CONFIGS[corruption_name]
    #     print(f"Using predefined corruption config: {corruption_name}")
    # else:
    config_to_use = corruption_config
    
    # Import ChestXDataset and Config from the original imports
    from mm_health_bench.mmhb.loader import ChestXDataset
    from mm_health_bench.mmhb.utils import Config
    
    config = Config("./mm_health_bench/config/config.yml").read()
    chestx_dataset = ChestXDataset(data_path=data_path, max_seq_length=max_seq_length)
    
    total_length = len(chestx_dataset)
    train_length = int(0.8 * total_length)
    val_length = int(0.1 * total_length)
    test_length = total_length - train_length - val_length
    
    # Check if a saved split exists
    if os.path.exists(indices_path):
        indices = torch.load(indices_path)
        train_indices = indices['train']
        val_indices = indices['val']
        test_indices = indices['test']
        
        # Create subsets based on the saved indices
        train_dataset = Subset(chestx_dataset, train_indices)
        val_dataset = Subset(chestx_dataset, val_indices)
        test_dataset = Subset(chestx_dataset, test_indices)
        print(f"Loaded split indices from {indices_path}")
    else:
        # Generate new splits and save the indices
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            chestx_dataset, [train_length, val_length, test_length]
        )
        indices = {
            'train': train_dataset.indices,
            'val': val_dataset.indices,
            'test': test_dataset.indices,
        }
        torch.save(indices, indices_path)
        print(f"Saved split indices to {indices_path}")
    
    print("Creating noisy data loaders with consistent indices")
    
    # Create data loaders with consistent noise
    train_loader = ConsistentNoiseDataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        corruption_config=config_to_use if apply_to_train else None,
        split_name="train",
        num_workers=num_workers, 
        collate_fn=noisy_chestx_collate_fn,
        seed=noise_seed
    )
    
    val_loader = ConsistentNoiseDataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        corruption_config=config_to_use if apply_to_val else None,
        split_name="val",
        num_workers=num_workers, 
        collate_fn=noisy_chestx_collate_fn,
        seed=noise_seed
    )
    
    test_loader = ConsistentNoiseDataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        corruption_config=config_to_use if apply_to_test else None,
        split_name="test",
        num_workers=num_workers, 
        collate_fn=noisy_chestx_collate_fn,
        seed=noise_seed
    )
    
    return train_loader, val_loader, test_loader

# # Example usage:
# if __name__ == "__main__":
#     # Define corruption configurations
#     corruption_configs = {
#         # Gaussian noise on images only
#          "gaussian_25": {
#             "name": "gaussian_25",
#             "image_noise_type": "gaussian",
#             "text_noise_type": None,
#             "image_noise_params": {"mean": 0.0, "std": 0.1},
#             "text_noise_params": None,
#             "image_noise_percentage": 25,
#             "text_noise_percentage": 0
#         },
#          "gaussian_50":{
#             "name": "gaussian_50",
#             "image_noise_type": "gaussian",
#             "text_noise_type": None,
#             "image_noise_params": {"mean": 0.0, "std": 0.1},
#             "text_noise_params": None,
#             "image_noise_percentage": 50,
#             "text_noise_percentage": 0
#         },
#          "gaussian_75":{
#             "name": "gaussian_75",
#             "image_noise_type": "gaussian",
#             "text_noise_type": None,
#             "image_noise_params": {"mean": 0.0, "std": 0.1},
#             "text_noise_params": None,
#             "image_noise_percentage": 75,
#             "text_noise_percentage": 0
#         },
#          "gaussian_100":{
#             "name": "gaussian_100",
#             "image_noise_type": "gaussian",
#             "text_noise_type": None,
#             "image_noise_params": {"mean": 0.0, "std": 0.1},
#             "text_noise_params": None,
#             "image_noise_percentage": 100,
#             "text_noise_percentage": 0
#         },
#         # {
#         #     "name": "sp_0_25",
#         #     "image_noise_type": "salt_pepper",
#         #     "text_noise_type": None,
#         #     "image_noise_params": {"amount": 0.25},
#         #     "text_noise_params": None,
#         #     "image_noise_percentage": 100,
#         #     "text_noise_percentage": 0
#         # },
#         # {
#         #     "name": "sp_0_50",
#         #     "image_noise_type": "salt_pepper",
#         #     "text_noise_type": None,
#         #     "image_noise_params": {"amount": 0.5},
#         #     "text_noise_params": None,
#         #     "image_noise_percentage": 100,
#         #     "text_noise_percentage": 0
#         # },
#         # {
#         #     "name": "sp_0_75",
#         #     "image_noise_type": "salt_pepper",
#         #     "text_noise_type": None,
#         #     "image_noise_params": {"amount": 0.75},
#         #     "text_noise_params": None,
#         #     "image_noise_percentage": 100,
#         #     "text_noise_percentage": 0
#         # },
#         # {
#         #     "name": "sp_0_100",
#         #     "image_noise_type": "salt_pepper",
#         #     "text_noise_type": None,
#         #     "image_noise_params": {"amount": 1},
#         #     "text_noise_params": None,
#         #     "image_noise_percentage": 100,
#         #     "text_noise_percentage": 0
#         # },
#         # {
#         #     "name": "blur_1_5",
#         #     "image_noise_type": "quality_reduction",
#         #     "text_noise_type": None,
#         #     "image_noise_params": {"blur_factor": 1.5},
#         #     "text_noise_params": None,
#         #     "image_noise_percentage": 100,
#         #     "text_noise_percentage": 0
#         # },
#         # {
#         #     "name": "blur_3",
#         #     "image_noise_type": "quality_reduction",
#         #     "text_noise_type": None,
#         #     "image_noise_params": {"blur_factor": 3},
#         #     "text_noise_params": None,
#         #     "image_noise_percentage": 100,
#         #     "text_noise_percentage": 0
#         # },
#         # {
#         #     "name": "blur_5",
#         #     "image_noise_type": "quality_reduction",
#         #     "text_noise_type": None,
#         #     "image_noise_params": {"blur_factor": 5},
#         #     "text_noise_params": None,
#         #     "image_noise_percentage": 100,
#         #     "text_noise_percentage": 0
#         # },
#         # {
#         #     "name": "blur_10",
#         #     "image_noise_type": "quality_reduction",
#         #     "text_noise_type": None,
#         #     "image_noise_params": {"blur_factor": 10},
#         #     "text_noise_params": None,
#         #     "image_noise_percentage": 100,
#         #     "text_noise_percentage": 0
#         # },
#         # {
#         #     "name": "mask_02_3",
#         #     "image_noise_type": "masking",
#         #     "text_noise_type": None,
#         #     "image_noise_params": {},
#         #     "text_noise_params": None,
#         #     "image_noise_percentage": 100,
#         #     "text_noise_percentage": 0
#         # },
#         # # Word dropout on text only
#         # {
#         #     "name": "dropout_01",
#         #     "image_noise_type": None,
#         #     "text_noise_type": "dropout",
#         #     "image_noise_params": None,
#         #     "text_noise_params": {"dropout_prob": 0.1},
#         #     "image_noise_percentage": 0,
#         #     "text_noise_percentage": 100
#         # },
#         {
#             "name": "dropout_03",
#             "image_noise_type": None,
#             "text_noise_type": "dropout",
#             "image_noise_params": None,
#             "text_noise_params": {"dropout_prob": 0.3},
#             "image_noise_percentage": 0,
#             "text_noise_percentage": 100
#         }}
#         # {
#         #     "name": "dropout_05",
#         #     "image_noise_type": None,
#         #     "text_noise_type": "dropout",
#         #     "image_noise_params": None,
#         #     "text_noise_params": {"dropout_prob": 0.5},
#         #     "image_noise_percentage": 0,
#         #     "text_noise_percentage": 100
#         # },
#         # {
#         #     "name": "swap_01",
#         #     "image_noise_type": None,
#         #     "text_noise_type": "dropout",
#         #     "image_noise_params": None,
#         #     "text_noise_params": {"swap_prob": 0.1},
#         #     "image_noise_percentage": 0,
#         #     "text_noise_percentage": 100
#         # },
#         # {
#         #     "name": "swap_03",
#         #     "image_noise_type": None,
#         #     "text_noise_type": "dropout",
#         #     "image_noise_params": None,
#         #     "text_noise_params": {"swap_prob": 0.3},
#         #     "image_noise_percentage": 0,
#         #     "text_noise_percentage": 100
#         # },
#         # {
#         #     "name": "swap_05",
#         #     "image_noise_type": None,
#         #     "text_noise_type": "dropout",
#         #     "image_noise_params": None,
#         #     "text_noise_params": {"dropout_prob": 0.5},
#         #     "image_noise_percentage": 0,
#         #     "text_noise_percentage": 100
#         # },
#         # # Combined corruption
#         # {
#         #     "name": "combined_corruption",
#         #     "image_noise_type": "quality_reduction",
#         #     "text_noise_type": "corruption",
#         #     "image_noise_params": {"blur_factor": 1.5, "noise_level": 0.1},
#         #     "text_noise_params": {"corruption_prob": 0.2},
#         #     "image_noise_percentage": 100,
#         #     "text_noise_percentage": 100
#         # }
    
    
#     # Create and save corrupted datasets
#     create_and_save_corrupted_datasets(
#         corruption_configs=corruption_configs,
#         output_dir="./corrupted_data",
#         apply_to_train=True,
#         apply_to_val=True,
#         apply_to_test=True
#     )
    
#     # Load corrupted data loaders
#     train_loader, val_loader, test_loader = load_corrupted_data_loaders(
#         corruption_name="gaussian_noise_images",
#         output_dir="./corrupted_data",
#         batch_size=32
#     )