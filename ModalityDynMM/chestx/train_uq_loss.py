import torch
from torch import nn
import time
from eval_scripts.performance import f1_score
from tqdm import tqdm

def train_dynmm_multilabel(
    train_dataloader,
    valid_dataloader,
    moe_model,
    total_epochs,
    lr=1e-4,
    weight_decay=0.01,
    optimtype=torch.optim.AdamW,
    early_stop=True,
    objective=nn.BCEWithLogitsLoss(),
    save='best.pt',
    lambda_weight=0.01
):
    """
    Simplified training function specifically for DynMM model with multilabel classification
    and quality-uncertainty weighted loss.
    
    Parameters:
    -----------
    train_dataloader : DataLoader
        Training data loader
    valid_dataloader : DataLoader
        Validation data loader
    moe_model : DynMMNet
        The pre-initialized DynMMNet model
    total_epochs : int
        Number of training epochs
    lr : float
        Learning rate for optimizer
    weight_decay : float
        Weight decay for optimizer
    optimtype : torch.optim
        Optimizer class
        Whether to use early stopping
    objective : torch.nn.Module
        Primary task loss function (default: BCEWithLogitsLoss)
    save : str
        Path to save the best model
    lambda_weight : float
        Weight for the resource penalty component of the loss
    """
    model = moe_model.cuda()
    
    # Initialize optimizer
    optimizer = optimtype(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Initialize tracking variables for early stopping
    best_f1_macro = 0
    patience = 0
    patience_limit = 7  # Number of epochs to wait before early stopping
    
    # Training loop
    for epoch in range(total_epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        task_losses = 0.0
        resource_losses = 0.0
        total_samples = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Prepare inputs and targets
            inputs = [i.cuda() for i in batch[:-1]]
            targets = batch[-1].cuda()
            
            # Forward pass through DynMMNet
            # This will store quality_scores, uncertainty_scores, and branch_weights in the model
            outputs, fusion_weight = model(inputs)
            
            # Calculate task loss
            task_loss = objective(outputs, targets)
            
            # Calculate resource loss component based on quality and uncertainty
            quality_scores = model.batch_quality_scores
            uncertainty_scores = model.batch_uncertainty_scores
            branch_weights = model.batch_branch_weights
            
            # Calculate resource usage based on branch selection
            per_sample_resource = torch.matmul(
                branch_weights, 
                model.flop.to(branch_weights.device)
            )
            
            # Quality-uncertainty modifier: High quality and low uncertainty increases penalty
            quality_uncertainty_modifier = quality_scores * (1.0 - uncertainty_scores)
            
            # Apply the quality-uncertainty weighted resource penalty
            resource_loss = per_sample_resource * quality_uncertainty_modifier.squeeze()
            mean_resource_loss = resource_loss.mean()
            
            # Combined loss
            total_batch_loss = task_loss + lambda_weight * mean_resource_loss
            
            # Backward pass and optimization
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)  # Gradient clipping
            optimizer.step()
            
            # Track losses
            batch_size = targets.size(0)
            total_loss += total_batch_loss.item() * batch_size
            task_losses += task_loss.item() * batch_size
            resource_losses += mean_resource_loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / total_samples,
                'task_loss': task_losses / total_samples,
                'resource_loss': resource_losses / total_samples
            })
        
        # Validation phase
        model.eval()
        model.reset_weight()  # Reset weight tracking for validation
        
        val_loss = 0.0
        val_samples = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in valid_dataloader:
                # Prepare inputs and targets
                inputs = [i.cuda() for i in batch[:-1]]
                targets = batch[-1].cuda()
                
                # Forward pass
                outputs, _ = model(inputs)
                
                # Calculate loss
                loss = objective(outputs, targets)
                
                # Track validation loss
                batch_size = targets.size(0)
                val_loss += loss.item() * batch_size
                val_samples += batch_size
                
                # Store predictions and targets for metrics
                all_preds.append(torch.sigmoid(outputs).round())
                all_targets.append(targets)
        
        # Get branch selection statistics
        branch_stats = model.get_selection_stats() if hasattr(model, 'get_selection_stats') else "Stats not available"
        branch_weights = model.weight_stat() if hasattr(model, 'weight_stat') else 0
        
        # Calculate validation metrics
        all_preds = torch.cat(all_preds, 0)
        all_targets = torch.cat(all_targets, 0)
        
        f1_micro = f1_score(all_targets, all_preds, average="micro")
        f1_macro = f1_score(all_targets, all_preds, average="macro")
        
        # Print epoch summary
        print('-' * 70)
        print(f'Epoch {epoch+1}/{total_epochs}:')
        print(f'Train loss: {total_loss/total_samples:.4f} | '
              f'Task loss: {task_losses/total_samples:.4f} | '
              f'Resource loss: {resource_losses/total_samples:.4f}')
        print(f'Val loss: {val_loss/val_samples:.4f} | '
              f'F1 micro: {f1_micro:.4f} | F1 macro: {f1_macro:.4f}')
        print(f'Branch weights: {branch_weights:.4f}')
        print(branch_stats)
        
        # Save best model and check early stopping
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            patience = 0
            print(f"New best F1 macro: {best_f1_macro:.4f}, saving model to {save}")
            torch.save(model, save)
        else:
            patience += 1
            print(f"No improvement, patience: {patience}/{patience_limit}")
            
        if early_stop and patience >= patience_limit:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    print(f"Training completed. Best F1 macro: {best_f1_macro:.4f}")
    return model


def test_dynmm_multilabel(model, test_dataloader, objective=nn.BCEWithLogitsLoss()):
    """
    Test function for DynMMNet multilabel classification.
    
    Parameters:
    -----------
    model : DynMMNet
        The trained DynMMNet model
    test_dataloader : DataLoader
        Test data loader
    objective : torch.nn.Module
        Loss function
        
    Returns:
    --------
    dict
        Dictionary with test metrics
    """
    model.eval()
    model.reset_weight()  # Reset branch selection tracking
    model.reset_selection_stats()  # Reset selection statistics
    
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Prepare inputs and targets
            inputs = [i.cuda() for i in batch[:-1]]
            targets = batch[-1].cuda()
            
            # Forward pass
            outputs, _ = model(inputs)
            
            # Calculate loss
            loss = objective(outputs, targets)
            
            # Track test loss
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Store predictions and targets for metrics
            all_preds.append(torch.sigmoid(outputs).round())
            all_targets.append(targets)
    
    # Calculate test metrics
    all_preds = torch.cat(all_preds, 0)
    all_targets = torch.cat(all_targets, 0)
    
    f1_micro = f1_score(all_targets, all_preds, average="micro")
    f1_macro = f1_score(all_targets, all_preds, average="macro")
    
    # Get branch selection statistics
    branch_stats = model.get_selection_stats()
    fusion_weight = model.weight_stat()
    flops = model.cal_flop()
    
    # Print test results
    print('-' * 70)
    print('Test Results:')
    print(f'Loss: {total_loss/total_samples:.4f} | '
          f'F1 micro: {f1_micro:.4f} | F1 macro: {f1_macro:.4f}')
    print(f'Average branch fusion weight: {fusion_weight:.4f}')
    print(f'Effective FLOPs: {flops:.2f}M')
    print(branch_stats)
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'loss': total_loss/total_samples,
        'fusion_weight': fusion_weight,
        'flops': flops
    }


# Example of how to use the functions:
"""
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser("DynMM Training")
    parser.add_argument("--n-epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--reg", type=float, default=0.01, help="resource loss weight")
    args = parser.parse_args()
    
    # Get data loaders
    train_data, val_data, test_data = get_data(32)
    
    # Initialize model
    model = DynMMNet(pretrain=True, freeze=True)
    
    # Train model
    filename = './log/dynmm_model.pt'
    train_dynmm_multilabel(
        train_data, 
        val_data, 
        model,
        args.n_epochs,
        lr=args.lr,
        weight_decay=args.wd,
        early_stop=True,
        objective=torch.nn.BCEWithLogitsLoss(),
        save=filename,
        lambda_weight=args.reg
    )
    
    # Test model
    model = torch.load(filename).cuda()
    test_dynmm_multilabel(model, test_data)
"""