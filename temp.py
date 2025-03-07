import argparse
import os
import sys
# sys.path.append('/content/drive/My Drive/Colab Notebooks/prediction/')
from natsort import natsorted
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from lib import Pre_dataset
from  network import  Seq2seqGRU, SASTANGen
from config import ParseGRU
from torch.utils.data import DataLoader

parse  = ParseGRU()
opt    = parse.args
print ( opt.trainset )
print("training")