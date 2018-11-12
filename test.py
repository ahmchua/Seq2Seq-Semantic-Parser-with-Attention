import argparse
import random
import numpy as np
import time
import torch
from torch import optim
from lf_evaluator import *
from models import *
from data import *
from utils import *
from train import *

def test_seq2seq(train_data_indexed, input_indexer, args):
    epochs = 5
    beam_length = 1
    out = len(input_indexer)
    train_iters(train_data_indexed, args.epochs, input_indexer, input_indexer, args, beam_length, out)
