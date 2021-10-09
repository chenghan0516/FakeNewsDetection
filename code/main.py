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


do_preprocess = 0
do_train = 0
do_evaluate = 0

if __name__ == "__main__":

    globals.init()

    if do_preprocess:
        print("Start preprocess")
        preprocess()
    if do_train:
        print("Start training")
        train.train()
    if do_evaluate:
        print("Start evaluating")
