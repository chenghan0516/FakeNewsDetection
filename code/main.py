from util import preprocess_input
import os
import pandas as pd
import numpy as np
import json
import heapq
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import globals
from util import preprocess, whats_running, Config
import train

# subject="politifact"
subject = "gossipcop"
path = "Bert"
save_every_pt = 1000
if not os.path.isdir('../{}/current'.format(path)):
    os.makedirs('../{}/current'.format(path))

do_preprocess = 0
do_train = 0
do_evaluate = 0

if __name__ == "__main__":

    globals.init()

    if do_preprocess:
        preprocess(input)
    if do_train:
        print("Start training")
        train.train()
    if do_evaluate:
        print("evaluate")
