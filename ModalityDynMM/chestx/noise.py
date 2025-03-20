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
    def reduce_image_quality(image, blur_factor=1.5, noise_level=0.1):
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
        blurred = F.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], 
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
    def add_word_dropout(text, dropout_prob=0.1):
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
    def swap_words(text, swap_prob=0.05):
        """
        Randomly swap adjacent words in text.
        
        Args:
            text (torch.Tensor): Text tensor of shape [batch, seq_len]
            swap_prob (float): Probability of swapping adjacent words
            
        Returns:
            torch.Tensor: Text with swapped words
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
    def corrupt_text(text, corruption_prob=0.3, pad_token_id=0):
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


def noisy_chestx_collate_fn(batch, image_noise_type=None, text_noise_type=None,
                           image_noise_params=None, text_noise_params=None,
                           image_noise_percentage=100, text_noise_percentage=100):
    """
    Custom collate function that adds noise to chest X-ray images and reports.
    
    Args:
        batch: Batch of data from ChestXDataset
        image_noise_type (str, optional): Type of noise to add to images
            Options: 'gaussian', 'salt_pepper', 'poisson', 'quality_reduction', 'masking', None
        text_noise_type (str, optional): Type of noise to add to text
            Options: 'dropout', 'swap', 'corruption', None
        image_noise_params (dict, optional): Parameters for image noise function
        text_noise_params (dict, optional): Parameters for text noise function
        image_noise_percentage (int): Percentage of images to apply noise to (0-100)
        text_noise_percentage (int): Percentage of text samples to apply noise to (0-100)
        
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
    
    batch_size = images.shape[0]
    
    # Apply image noise if specified
    if image_noise_type:
        image_noise_params = image_noise_params or {}
        
        # Determine which samples to add noise to
        num_noisy_images = int(batch_size * image_noise_percentage / 100)
        noisy_image_indices = random.sample(range(batch_size), num_noisy_images)
        
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
        for idx in noisy_image_indices:
            images[idx] = noisy_images[idx]
    
    # Apply text noise if specified
    if text_noise_type:
        text_noise_params = text_noise_params or {}
        
        # Determine which samples to add noise to
        num_noisy_texts = int(batch_size * text_noise_percentage / 100)
        noisy_text_indices = random.sample(range(batch_size), num_noisy_texts)
        
        # Create a noisy version of the entire batch
        if text_noise_type == 'dropout':
            noisy_reports = NoiseAugmenter.add_word_dropout(reports, **text_noise_params)
        elif text_noise_type == 'swap':
            noisy_reports = NoiseAugmenter.swap_words(reports, **text_noise_params)
        elif text_noise_type == 'corruption':
            noisy_reports = NoiseAugmenter.corrupt_text(reports, **text_noise_params)
        
        # Apply noise only to selected samples
        for idx in noisy_text_indices:
            reports[idx] = noisy_reports[idx]
    
    return reports, images, targets


def get_data_with_noise(batch_size=32, num_workers=4,
                       image_noise_type=None, text_noise_type=None,
                       image_noise_params=None, text_noise_params=None,
                       image_noise_percentage=100, text_noise_percentage=100,
                       apply_to_train=True, apply_to_val=True, apply_to_test=True,
                       indices_path="split_indices.pth"):
    """
    Get data loaders with specified noise applied to images and/or text.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        image_noise_type (str, optional): Type of noise to add to images
        text_noise_type (str, optional): Type of noise to add to text
        image_noise_params (dict, optional): Parameters for image noise function
        text_noise_params (dict, optional): Parameters for text noise function
        image_noise_percentage (int): Percentage of images to apply noise to (0-100)
        text_noise_percentage (int): Percentage of text samples to apply noise to (0-100)
        apply_to_train (bool): Whether to apply noise to training data
        apply_to_val (bool): Whether to apply noise to validation data
        apply_to_test (bool): Whether to apply noise to test data
        
    Returns:
        tuple: (train_loader, val_loader, test_loader) with noise applied as configured
    """
    config = Config("./mm_health_bench/config/config.yml").read()
    chestx_dataset = ChestXDataset(data_path="./mm_health_bench/data/chestx", max_seq_length=256)
    
    total_length = len(chestx_dataset)
    train_length = int(0.8 * total_length)
    val_length = int(0.1 * total_length)
    test_length = total_length - train_length - val_length
    
    
    # Check if a saved split exists.
    if os.path.exists(indices_path):
        indices = torch.load(indices_path)
        train_indices = indices['train']
        val_indices = indices['val']
        test_indices = indices['test']
        
        # Create subsets based on the saved indices.
        train_dataset = Subset(chestx_dataset, train_indices)
        val_dataset = Subset(chestx_dataset, val_indices)
        test_dataset = Subset(chestx_dataset, test_indices)
        print(f"Loaded split indices from {indices_path}")
    else:
        # Generate new splits and save the indices.
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
    
    # Regular collate function (no noise)
    regular_collate = lambda batch: noisy_chestx_collate_fn(
        batch,
        image_noise_type=None,
        text_noise_type=None
    )
    
    # Noisy collate function
    noisy_collate = lambda batch: noisy_chestx_collate_fn(
        batch, 
        image_noise_type=image_noise_type,
        text_noise_type=text_noise_type,
        image_noise_params=image_noise_params,
        text_noise_params=text_noise_params,
        image_noise_percentage=image_noise_percentage,
        text_noise_percentage=text_noise_percentage
    )
    
    # Create loaders with appropriate noise settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        collate_fn=noisy_collate if apply_to_train else regular_collate
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        collate_fn=noisy_collate if apply_to_val else regular_collate
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        collate_fn=noisy_collate if apply_to_test else regular_collate
    )
    
    return train_loader, val_loader, test_loader

