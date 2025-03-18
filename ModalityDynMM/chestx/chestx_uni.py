import sys 
import os 
sys.path.append(os.getcwd())
sys.path.append('/home/bianca/Code/MultiBench/mm_health_bench/mmhb')
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from unimodals.common_models import VGG11Slim, MLP, ReportTransformer
from datasets.imdb.get_data import get_dataloader
from training_structures_dynmm.unimodal import train, test
from torch.utils.data import DataLoader, TensorDataset
# from mm_health_bench.mmhb.loader import *
# from mm_health_bench.mmhb.utils import Config
from chestx_utils import *

def compute_class_weights(train_loader):
    class_counts = torch.zeros(14)
    total_samples = 0
    
    for batch in train_loader:
        _, _, targets = batch
        class_counts += targets.sum(dim=0)
        total_samples += targets.size(0)
    
    # Calculate inverse frequency weights, capped to prevent extreme values
    class_weights = total_samples / (class_counts + 1)  # Add 1 to prevent division by zero
    
    # Normalize weights to have mean of 1
    class_weights = class_weights / class_weights.mean()
    
    # Cap extremely high weights to 10
    class_weights = torch.clamp(class_weights, max=10.0)
    
    return class_weights.cuda()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("chestx",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    argparser.add_argument("--n-runs", type=int, default=1, help="number of runs")
    argparser.add_argument("--mod", type=int, default=0, help="0: text; 1: image")
    argparser.add_argument("--eval-only", action='store_true', help='no training')
    argparser.add_argument("--measure", action='store_true', help='no training')
    argparser.add_argument("--dir", default='chestx/', help='folder to store results')
    argparser.add_argument("--balanced", action='store_true', help='balanced dataset')
    args = argparser.parse_args()

    
    if not os.path.exists("./log/" + args.dir):
        os.makedirs("./log/" + args.dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    modality = 'image' if args.mod == 1 else 'text'
    model_file = "./log/" + args.dir + "model_" + modality + ".pt"
    head_file = "./log/" + args.dir + "head_" + modality + ".pt"

    log1, log2 = [], []
    for n in range(args.n_runs):
        if args.mod == 0:
            model = MLP(256, 256, 256).cuda()
            head = MLP(256, 256, 14).cuda() 
        else:
            model = VGG11Slim(256).cuda()
            head = MLP(256, 256, 14).cuda()

        if args.balanced:
            train_data, val_data, test_data = get_data_balanced(batch_size=64, num_workers=4)
        else:
            train_data, val_data, test_data = get_data(batch_size=64, num_workers=4)

        if not args.eval_only:
            train(model, head, train_data, val_data, 100, early_stop=True, task="multilabel",
                    save_encoder=model_file, save_head=head_file,
                    modalnum=args.mod, optimtype=torch.optim.AdamW, lr=1e-4, weight_decay=0.01,
                    criterion=torch.nn.BCEWithLogitsLoss())

        print(f"Testing model {model_file} and {head_file}:")
        model = torch.load(model_file).cuda()
        head = torch.load(head_file).cuda()

        tmp = test(model, head, test_data, "default", modality, task="multilabel", modalnum=args.mod, no_robust=True)
        log1.append(tmp['f1_micro'])
        log2.append(tmp['f1_macro'])

    print(log1, log2)
    print(f'Finish {args.n_runs} runs')
    print(f'f1 micro {np.mean(log1) * 100:.2f} ± {np.std(log1) * 100:.2f}')
    print(f'f1 macro {np.mean(log2) * 100:.2f} ± {np.std(log2) * 100:.2f}')
    
    
        
