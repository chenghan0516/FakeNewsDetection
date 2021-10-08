import os
import numpy as np
import pandas as pd
import torch
import json


class Config:
    def __init__(
        self,
        subject,
        model_type,
        with_sentiment,
        epoch,
        end_warmup,
        lr,
        scheduler_gamma,
        save_every_pt,
        freezed_bert_layer_num,
        progressive_unfreeze,
        progressive_unfreeze_step,
        max_unfreeze_layer_num,
        evaluate_or_train,
        eval_best_n,
        eval_waiting_queue_begin,
    ):

        # 現在在執行甚麼
        self.subject = subject
        self.model_type = model_type
        self.with_sentiment = with_sentiment

        # 訓練過程參數
        self.epoch = epoch  # 訓練總epoch
        self.end_warmup = end_warmup  # 暖身完畢的epoch
        self.lr = lr  # learning rate
        self.scheduler_gamma = scheduler_gamma
        self.save_every_pt = save_every_pt  # 每幾epoch存一次模型

        # pretrain model相關
        # freeze前幾層pretrained model(bert共12層)
        self.freezed_bert_layer_num = freezed_bert_layer_num
        self.progressive_unfreeze = progressive_unfreeze  # 是否一層一層解凍pretrained model
        self.progressive_unfreeze_step = progressive_unfreeze_step  # 每幾epoch解凍一層模型
        self.max_unfreeze_layer_num = max_unfreeze_layer_num  # 最多解凍幾層

        # evaluate用參數
        # 0 = training set, 1 = evaluating set
        self.evaluate_or_train = evaluate_or_train
        self.eval_best_n = eval_best_n
        self.eval_waiting_queue_begin = eval_waiting_queue_begin
        self.eval_waiting_queue_end = self.eval_best_n + self.eval_waiting_queue_begin


def init():

    global config
    config = Config(
        subject="politifact",
        model_type="Bert",
        with_sentiment=False,
        epoch=40,
        end_warmup=50,
        lr=0.001,
        scheduler_gamma=0.85,
        save_every_pt=1000,
        freezed_bert_layer_num=10,
        progressive_unfreeze=False,
        progressive_unfreeze_step=10,
        max_unfreeze_layer_num=3,
        evaluate_or_train=1,
        eval_best_n=50,
        eval_waiting_queue_begin=0,
    )

    # paths in preprocessing
    global raw_data_path, train_data_path, test_data_path
    raw_data_path = "../{}_clean.csv".format(config.subject)
    if not os.path.isdir("../{}".format(config.model_type)):
        os.makedirs("../{}".format(config.model_type))
    train_data_path = "../{}/{}_{}_token_data_train.csv".format(
        config.model_type, config.model_type, config.subject
    )
    test_data_path = "../{}/{}_{}_token_data_test.csv".format(
        config.model_type, config.model_type, config.subject
    )

    # paths in training
    global current_folder, train_progress_path, random_file_path
    if config.with_sentiment:
        current_folder = "../{}_with_sentiment/current".format(config.model_type)
    else:
        current_folder = "../{}/current".format(config.model_type)
    if not os.path.isdir(current_folder):
        os.makedirs(current_folder)
    train_progress_path = "{}/train_progress.json".format(current_folder)
    random_file_path = "{}/random.txt".format(current_folder)


def get_random_index(random_length):
    global random_index
    if os.path.isfile(random_file_path):
        print("old random")
        with open(random_file_path, "r") as f:
            random_index = [int(i) for i in list(f.read().split("\n")[:-1])]
    else:
        print("new random")
        random_index = list(range(random_length))
        np.random.shuffle(random_index)
        f = open(random_file_path, "w")
        for i in random_index:
            f.write(str(i) + "\n")
        f.close()
