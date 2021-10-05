from util import Config
import os
import numpy as np
import pandas as pd
import torch
import json


def init():

    global config
    config = Config(subject='politifact',
                    model_type="Bert",
                    with_sentiment=False,

                    epoch=40,
                    end_warmup=15,
                    lr=0.001,
                    scheduler_gamma=0.85,
                    save_every_pt=1000,

                    freezed_bert_layer_num=10,
                    progressive_unfreeze=False,
                    progressive_unfreeze_step=10,
                    max_unfreeze_layer_num=3,

                    evaluate_or_train=1,
                    eval_best_n=50,
                    eval_waiting_queue_begin=0)

    #paths in preprocessing
    global raw_data_path, train_data_path, test_data_path
    raw_data_path = "../{}_clean.csv".format(config.subject)
    train_data_path = "../{}/{}_{}_token_data_train.csv".format(
        config.model_type, config.model_type, config.subject)
    test_data_path = "../{}/{}_{}_token_data_test.csv".format(
        config.model_type, config.model_type, config.subject)

    #paths in training
    global current_folder, train_progress_path, random_file_path
    if config.with_sentiment:
        current_folder = '../{}_with_sentiment/current'.format(
            config.model_type)
    else:
        current_folder = '../{}/current'.format(config.model_type)
    train_progress_path = '{}/train_progress.json'.format(current_folder)
    random_file_path = '{}/random.txt'.format(current_folder)


def get_random_index(random_length):
    global random_index
    if os.path.isfile(random_file_path):
        print("old random")
        f = open(random_file_path, 'r')
        random_index = [int(i) for i in list(f.read().split("\n")[:-1])]
        f.close()
    else:
        print("new random")
        random_index = list(range(random_length))
        np.random.shuffle(random_index)
        f = open(random_file_path, 'w')
        for i in random_index:
            f.write(str(i)+"\n")
        f.close()
