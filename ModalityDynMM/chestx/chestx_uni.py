import sys 
import os 
sys.path.append(os.getcwd())
sys.path.append('/home/bianca/Code/MultiBench/mm_health_bench/mmhb')
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from unimodals.common_models import GRUWithLinear, VGG11Slim, MLP
from datasets.imdb.get_data import get_dataloader
from training_structures.unimodal import train, test
from torch.utils.data import DataLoader, TensorDataset
# from mm_health_bench.mmhb.loader import *
# from mm_health_bench.mmhb.utils import Config
from chestx_utils import *


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("chestx",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    argparser.add_argument("--n-runs", type=int, default=1, help="number of runs")
    argparser.add_argument("--mod", type=int, default=0, help="0: text; 1: image")
    argparser.add_argument("--eval-only", action='store_true', help='no training')
    argparser.add_argument("--measure", action='store_true', help='no training')
    args = argparser.parse_args()

    
    if not os.path.exists("./log/chestx/"):
        os.makedirs("./log/chestx/")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    modality = 'image' if args.mod == 1 else 'text'
    model_file = "./log/chestx/model_" + modality + ".pt"
    head_file = "./log/chestx/head_" + modality + ".pt"

    log1, log2 = [], []
    for n in range(args.n_runs):
        if args.mod == 0:
            model = GRUWithLinear(256, 256, 256).cuda()
            head = MLP(256, 256, 14).cuda()
            train_data, val_data, test_data = get_data(batch_size=32, num_workers=4)
        else:
            model = VGG11Slim(128).cuda()
            head = MLP(128, 128, 14).cuda()
            train_data, val_data, test_data = get_data(batch_size=32, num_workers=4)

        # train_data, val_data, test_data = generate_random_data()

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
    
    
        
