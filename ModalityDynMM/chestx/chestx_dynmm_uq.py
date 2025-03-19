import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.getcwd())
from chestx_utils import *
from unimodals.common_models import MLP, Linear, MaxOut_MLP
from fusions.common_fusions import Concat
from ModalityDynMM.training_structures_dynmm.Supervised_Learning import train, test, MMDL


def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class LightTransformerBlock(nn.Module):
    """Lightweight transformer block for efficient processing"""
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super(LightTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class QualityAssessor(nn.Module):
    """Quality assessment module using lightweight transformer"""
    def __init__(self, input_dim, hidden_dim=128, num_heads=2, depth=1):
        super(QualityAssessor, self).__init__()
        
        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional embedding for transformer
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            LightTransformerBlock(hidden_dim, num_heads=num_heads) 
            for _ in range(depth)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Add batch dimension if needed for single sample processing
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, F] -> [B, 1, F]
            
        # Project input
        x = self.input_proj(x)  # [B, 1, F] -> [B, 1, H]
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            
        # Global average pooling
        x = x.mean(dim=1)  # [B, 1, H] -> [B, H]
        
        # Project to output score and apply sigmoid
        return torch.sigmoid(self.output_proj(x))  # [B, 1]


class UncertaintyEstimator(nn.Module):
    """Uncertainty estimation using lightweight transformer"""
    def __init__(self, feature_dim, hidden_dim=128, num_heads=2, depth=1):
        super(UncertaintyEstimator, self).__init__()
        
        # Project input to hidden dimension
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Positional embedding for transformer
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            LightTransformerBlock(hidden_dim, num_heads=num_heads) 
            for _ in range(depth)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, features):
        # Add batch dimension if needed for single sample processing
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [B, F] -> [B, 1, F]
            
        # Project input
        x = self.input_proj(features)  # [B, 1, F] -> [B, 1, H]
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            
        # Global average pooling
        x = x.mean(dim=1)  # [B, 1, H] -> [B, H]
        
        # Project to output score and apply sigmoid
        return torch.sigmoid(self.output_proj(x))  # [B, 1]


class DynMMNet(nn.Module):
    def __init__(self, branch_num=2, pretrain=True, freeze=True, directory="chestx/"):
        # Add branch selection counters
        self.branch_selections = torch.zeros(branch_num)
        self.total_samples = 0
        super(DynMMNet, self).__init__()
        self.branch_num = branch_num
        self.dir = directory
        self.image_preprocess = nn.Linear(196864, 4396)
        
        # branch 1: text network
        self.text_encoder = torch.load('./log/' + self.dir + 'model_text.pt') if pretrain else MLP(256, 256, 256)
        self.text_head = torch.load('./log/' + self.dir + 'head_text.pt') if pretrain else MLP(256, 256, 14)
        
        # branch2: late fusion of text+image
        if pretrain:
            self.branch2 = torch.load('./log/' + self.dir + 'best_lf.pt')
            
        if freeze:
            self.freeze_branch(self.text_encoder)
            self.freeze_branch(self.text_head)
            self.freeze_branch(self.branch2)

        # Quality assessment modules
        self.text_quality = QualityAssessor(256)  # Assumes text feature dimension is 256
        self.image_quality = QualityAssessor(3072)  # For flattened 32x32x3 images
        
        # Uncertainty estimators
        self.text_uncertainty = UncertaintyEstimator(256)
        self.fusion_uncertainty = UncertaintyEstimator(512)
        
        # Enhanced gating network with quality and uncertainty inputs using a lightweight transformer
        # Input: text features, image features, quality scores, uncertainty estimates
        self.gate_input_dim = 3328 + 4  # +4 for 2 quality scores and 2 uncertainty scores
        self.gate_hidden_dim = 256
        
        # Input projection for gating network
        self.gate_input_proj = nn.Linear(self.gate_input_dim, self.gate_hidden_dim)
        
        # Transformer blocks for gating
        self.gate_transformer = LightTransformerBlock(self.gate_hidden_dim, num_heads=4)
        
        # Output projection for branch selection
        self.gate_output = nn.Linear(self.gate_hidden_dim, branch_num)
        
        self.temp = 1
        self.hard_gate = True
        self.weight_list = torch.Tensor()
        self.store_weight = False
        self.infer_mode = 0
        self.flop = torch.Tensor([0.400384, 119644.54])
        
        # Hyperparameter for resource penalty
        self.lambda_resource = 0.1
    
    def reset_selection_stats(self):
        self.branch_selections = torch.zeros_like(self.branch_selections)
        self.total_samples = 0

    def freeze_branch(self, m):
        for param in m.parameters():
            param.requires_grad = False

    def reset_weight(self):
        self.weight_list = torch.Tensor()
        self.store_weight = True

    def weight_stat(self):
        print(self.weight_list)
        tmp = torch.mean(self.weight_list, dim=0)
        print(f'mean branch weight {tmp[0].item():.4f}, {tmp[1].item():.4f}')
        self.store_weight = False
        return tmp[1].item()  # Return the weight for the fusion branch

    def cal_flop(self):
        tmp = torch.mean(self.weight_list, dim=0)
        total_flop = (self.flop * tmp).sum()
        print(f'Total Flops {total_flop.item():.2f}M')
        return total_flop.item()

    def forward(self, inputs):
        text_input = inputs[0]  # Shape: [batch, seq]
        image_input = inputs[1]  # Shape: [batch, h, w, c]
        
        # Get batch size
        batch_size = text_input.size(0)
        
        # Reduce image dimensions through pooling or resizing
        reduced_image = F.adaptive_avg_pool2d(image_input, (32, 32))  # Reduce to 32x32
        image_flattened = reduced_image.view(batch_size, -1)  # Now much smaller
        
        # Get text features for quality assessment
        text_features = self.text_encoder(text_input)
        
        # Assess quality for each modality
        text_quality = self.text_quality(text_features)
        image_quality = self.image_quality(image_flattened)
        
        # Estimate uncertainty for each expert
        text_uncertainty = self.text_uncertainty(text_features)
        
        # For fusion uncertainty, we need features from both modalities
        # We can use a simple concatenation of features
        fusion_features = torch.cat([text_features, image_flattened[:, :256]], dim=1)
        fusion_uncertainty = self.fusion_uncertainty(fusion_features)
        
        # Concatenate all inputs for the gating network
        # Original inputs + quality scores + uncertainty estimates
        x = torch.cat([
            text_input, 
            image_flattened, 
            text_quality, 
            image_quality, 
            text_uncertainty, 
            fusion_uncertainty
        ], dim=1)
        
        # Process through the transformer-based gating network
        gate_input = self.gate_input_proj(x).unsqueeze(1)  # Add sequence dimension
        gate_features = self.gate_transformer(gate_input).squeeze(1)  # Remove sequence dimension
        gate_logits = self.gate_output(gate_features)
        
        # Get weights from gating network
        weight = DiffSoftmax(gate_logits, tau=self.temp, hard=self.hard_gate)

        if self.store_weight:
            self.weight_list = torch.cat((self.weight_list, weight.cpu()))
        
        if self.hard_gate:
            # Get the index of the maximum weight for each sample in the batch
            selected_branches = torch.argmax(weight, dim=1)
            
            # Count selections for each branch
            for i in range(self.branch_num):
                self.branch_selections[i] += (selected_branches == i).sum().item()
            
            # Track total number of samples
            self.total_samples += weight.size(0)

        # Get predictions from both branches
        pred_list = [
            self.text_head(text_features),                 # Branch 1: Text only
            self.branch2(inputs)                           # Branch 2: Late fusion
        ]
        
        if self.infer_mode > 0:
            return pred_list[self.infer_mode - 1], 0

        # Combine predictions using weights
        output = (weight[:, 0:1] * pred_list[0] + 
                  weight[:, 1:2] * pred_list[1])
                
        # Return output and average weight of fusion branch for monitoring
        return output, weight[:, 1].mean()

    def forward_separate_branch(self, inputs, path, weight_enable):  # see separate branch performance
        text_input = inputs[0]
        image_input = inputs[1]
        
        # Get batch size
        batch_size = text_input.size(0)
        
        # Reduce image dimensions
        reduced_image = F.adaptive_avg_pool2d(image_input, (32, 32))
        image_flattened = reduced_image.view(batch_size, -1)
        
        # Get text features
        text_features = self.text_encoder(text_input)
        
        # Quality and uncertainty assessment
        text_quality = self.text_quality(text_features)
        image_quality = self.image_quality(image_flattened)
        text_uncertainty = self.text_uncertainty(text_features)
        fusion_features = torch.cat([text_features, image_flattened[:, :256]], dim=1)
        fusion_uncertainty = self.fusion_uncertainty(fusion_features)
        
        if weight_enable:
            x = torch.cat([
                text_input, 
                image_flattened, 
                text_quality, 
                image_quality, 
                text_uncertainty, 
                fusion_uncertainty
            ], dim=1)
            
            # Process through the transformer-based gating network
            gate_input = self.gate_input_proj(x).unsqueeze(1)
            gate_features = self.gate_transformer(gate_input).squeeze(1)
            gate_logits = self.gate_output(gate_features)
            
            weight = DiffSoftmax(gate_logits, tau=self.temp, hard=self.hard_gate)
        
        if path == 1:
            output = self.text_head(text_features)
        elif path == 2:
            output = self.branch2(inputs)

        return output

    def get_selection_stats(self):
        if self.total_samples == 0:
            return "No samples processed yet."
        
        percentages = (self.branch_selections / self.total_samples) * 100
        
        stats = "Branch selection statistics:\n"
        for i in range(self.branch_num):
            stats += f"Branch {i+1}: selected {self.branch_selections[i]} times " \
                    f"({percentages[i]:.2f}% of samples)\n"
        
        return stats

    def quality_uncertainty_loss(self, pred, target, quality_scores, uncertainty_scores):
        """
        Custom loss that incorporates quality and uncertainty awareness:
        L = L_task + λ × L_resource × Q × (1 - U)
        
        Where:
        - L_task: Task loss (e.g., BCE)
        - λ: Resource penalty weight
        - L_resource: Resource usage loss (based on computation used)
        - Q: Quality score (higher for better quality)
        - U: Uncertainty score (higher for more uncertain predictions)
        """
        # Base task loss (e.g., BCE)
        task_loss = F.binary_cross_entropy_with_logits(pred, target)
        
        # Resource usage loss based on which branch was selected
        # Higher weight for fusion branch (more compute intensive)
        resource_loss = torch.mean(self.weight_list[:, 1])
        
        # Average quality and uncertainty scores across batch
        avg_quality = torch.mean(quality_scores)
        avg_uncertainty = torch.mean(uncertainty_scores)
        
        # Quality and uncertainty weighted resource penalty
        # When quality is high and uncertainty is low, penalize resource usage more
        # When quality is low or uncertainty is high, allow more computation
        resource_penalty = self.lambda_resource * resource_loss * avg_quality * (1 - avg_uncertainty)
        
        return task_loss + resource_penalty


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("chestx",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    argparser.add_argument("--n-runs", type=int, default=1, help="number of runs")
    argparser.add_argument("--data", type=str, default='chestx', help="dataset name")
    argparser.add_argument("--n-epochs", type=int, default=50, help="number of epochs")
    argparser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    argparser.add_argument("--wd", type=float, default=1e-2, help="weight decay")
    argparser.add_argument("--reg", type=float, default=0.1, help="reg loss weight")
    argparser.add_argument("--freeze", action='store_true', help='freeze branch weights')
    argparser.add_argument("--eval-only", action='store_true', help='no training')
    argparser.add_argument("--hard", action='store_true', help='hard labels')
    argparser.add_argument("--no-pretrain", action='store_true', help='train from scratch')
    argparser.add_argument("--infer-mode", type=int, default=0, help="infer mode")
    argparser.add_argument("--balanced", action='store_true', help='balanced dataset')
    # Add new arguments for quality-aware features
    argparser.add_argument("--lambda-resource", type=float, default=0.1, help="resource penalty weight")

    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    if args.balanced:
        train_data, val_data, test_data = get_data(64)
        # Init Model
        model = DynMMNet(pretrain=1-args.no_pretrain, freeze=args.freeze)
        model.lambda_resource = args.lambda_resource
        filename = os.path.join('./log', args.data, 'QualityAwareDynMMNet_freeze' + str(args.freeze) + '_reg_' + str(args.reg) + '_lambda_' + str(args.lambda_resource) + '.pt')

        if not args.eval_only:
            model.hard_gate = args.hard
            train(None, None, None, train_data, val_data, args.n_epochs, task="multilabel", optimtype=torch.optim.AdamW,
                  is_packed=False, early_stop=True, lr=args.lr, save=filename, weight_decay=args.wd,
                  objective=torch.nn.BCEWithLogitsLoss(), moe_model=model, additional_loss=True, lossw=args.reg)

        # Test
        print(f"Testing model {filename}:")
        model = torch.load(filename).cuda()
        model.hard_gate = True

        # print('-' * 30 + 'Val data' + '-' * 30)
        model.infer_mode = args.infer_mode
        # tmp = test(model=model, test_dataloaders_all=validdata, dataset=args.data, is_packed=False,
        #            criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel", no_robust=True, additional_loss=True)

        print('-' * 30 + 'Test data' + '-' * 30)
        model.reset_weight()
        model.reset_selection_stats()
        tmp = test(model=model, test_dataloaders_all=test_data, dataset=args.data, is_packed=False,
                   criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel", no_robust=True, additional_loss=True)
        print(model.get_selection_stats())
        log1.append(model.weight_stat())
        log2.append(tmp['f1_micro'], tmp['f1_macro'], model.cal_flop())

    print(log1)
    print(log2)
    print('-' * 60)
    print(f'Finish {args.n_runs} runs')
    # print(f'Val f1 micro {np.mean(log1[:, 0]) * 100:.2f} ± {np.std(log1[:, 0]) * 100:.2f} | f1 macro {np.mean(log1[:, 1]) * 100:.2f} ± {np.std(log1[:, 0]) * 100:.2f}')
    print(f'Test f1 micro {np.mean(log2[:, 0]) * 100:.2f} ± {np.std(log2[:, 0]) * 100:.2f} | '
          f'f1 macro {np.mean(log2[:, 1]) * 100:.2f} ± {np.std(log2[:, 0]) * 100:.2f} | ' 
          f'Flop saving {np.mean(log2[:, 2]):.2f} ± {np.std(log2[:, 2]):.2f}M | '
          f'Branch selection ratio {np.mean(log1):.3f} ± {np.std(log1):.3f}')
    idx = np.argmax(log2[:, 1])
    print('Best result', log2[idx, :])