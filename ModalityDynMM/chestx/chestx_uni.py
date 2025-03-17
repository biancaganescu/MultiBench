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

    return train_loader, val_loader, test_loader,


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
            head = MLP(256, 256, 2).cuda()
            # train_data, val_data, test_data, _, _, _ = get_data(batch_size=32, num_workers=4)
        else:
            model = VGG11Slim(128).cuda()
            head = MLP(128, 128, 2).cuda()
            # _, _, _, train_data, val_data, test_data = get_data(batch_size=32, num_workers=4)

        train_data, val_data, test_data = generate_random_data()

        if not args.eval_only:
            train(model, head, train_data, val_data, 10, early_stop=True, task="multilabel",
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
    
    
        
