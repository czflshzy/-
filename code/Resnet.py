import torch
import argparse
import tqdm
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import copy
import os
import pandas as pd
from torchvision import transforms, models

"""超参数部分"""

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--bs', default=4, type=int, help='在train集上的batchSize')
parser.add_argument('--lr', default=1e-4, type=float, help='')
parser.add_argument('--gpu', default=0, type=float, help='')
parser.add_argument('--model', default="resnet18", type=str)
# help='resnet18 | mobilenet_v2 | resnet50 | convnext_base | regnet_x_400mf | convnext_tiny |')
parser.add_argument('--scheduler', default="linear")
opt = parser.parse_args()


if __name__ == '__main__':
    best_acc = 0
    resnet = models.resnet18(pretrained=True)
    input_dim = 512





