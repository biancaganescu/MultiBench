import sys
import os
sys.path.append(os.getcwd())

import argparse
import numpy as np
import torch

from unimodals.common_models import VGG11Slim, Linear, GRUWithLinear
from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import Concat, LowRankTensorFusion, MultiplicativeInteractions2Modal
from training_structures.Supervised_Learning import train, test
from chestx_utils import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("chestx", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    argparser.add_argument("--n-runs", type=int, default=1, help="number of runs")
    argparser.add_argument("--fuse", type=int, default=0, help="fusion model")
    argparser.add_argument("--eval-only", action='store_true', help='no training')
    argparser.add_argument("--measure", action='store_true', help='no training')
    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    fusion_dict = {0: 'ef', 1: 'lf', 2: 'lrtf', 3: 'mim'}
    filename = "./log/chestx/fused_models_" + fusion_dict[args.fuse] + ".pt"

    train_data, val_data, test_data = generate_random_data(batch_size=32)

    log1, log2 = [], []
    for n in range(args.n_runs):
        models = [GRUWithLinear(256, 256, 256).cuda(), VGG11Slim(256).cuda()]
        lr = 1e-5

        if args.fuse in set([0, 1]):
            fusion = Concat().cuda()
            head= Linear(512, 2).cuda()
        elif args.fuse == 2:
            fusion = LowRankTensorFusion([256, 256], 256, 256).cuda()
            head= Linear(256, 2).cuda()
        elif args.fuse == 3:
            fusion = MultiplicativeInteractions2Modal([256, 256], 512, 'matrix').cuda()
            head= Linear(512, 2).cuda()

        if not args.eval_only:
            train(models, fusion, head, train_data, val_data, 1000, early_stop=False, task="multilabel",
                    save=filename, optimtype=torch.optim.AdamW, lr=lr, weight_decay=0.001,
                    objective=torch.nn.BCEWithLogitsLoss())

        print(f"Testing {filename}")
        model = torch.load(filename).cuda()

        tmp = test(model, test_data, method_name=fusion_dict[args.fuse], dataset="default", criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel", no_robust=True)

        log1.append(tmp['f1_micro'])
        log2.append(tmp['f1_macro'])

    print(log1, log2)
    print(f'Finish {args.n_runs} runs')
    print(f'f1 micro {np.mean(log1) * 100:.2f} ± {np.std(log1) * 100:.2f}')
    print(f'f1 macro {np.mean(log2) * 100:.2f} ± {np.std(log2) * 100:.2f}')



