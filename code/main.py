import os
import pandas as pd
import numpy as np
import json
import heapq
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import globals
from util import preprocess
import train

save_every_pt = 1000

do_preprocess = 1
do_train = 0
do_evaluate = 0

if __name__ == "__main__":

    globals.init()

    if do_preprocess:
        preprocess()
    if do_train:
        print("Start training")
        if not os.path.isdir('../{}/current'.format(globals.config.model_type)):
            os.makedirs('../{}/current'.format(globals.config.model_type))
        train.train()
    if do_evaluate:
        print("evaluate")
