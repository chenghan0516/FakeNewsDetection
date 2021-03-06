import os
import numpy as np
import torch
import json


class Config:
    def __init__(self):
        with open("config.json", "r") as f:
            config_json = json.load(f)
            # 現在在執行甚麼
            self.subject = config_json["subject"]
            self.embedding_type = config_json["embedding_type"]
            self.model_type = config_json["model_type"]
            self.with_sentiment = config_json["with_sentiment"]

            # 訓練過程參數
            self.epoch = config_json["epoch"]  # 訓練總epoch
            self.end_warmup = config_json["end_warmup"]  # 暖身完畢的epoch
            self.lr = config_json["lr"]  # learning rate
            self.scheduler_gamma = config_json["scheduler_gamma"]
            self.save_every_pt = config_json["save_every_pt"]  # 每幾epoch存一次模型

            # pretrain model相關
            # freeze前幾層pretrained model(e.g. bert共12層)
            self.freezed_pretrain_layer_num = config_json["freezed_pretrain_layer_num"]
            # 若有用到sentiment，是否finetune
            self.freeze_sentiment_model = config_json["freeze_sentiment_model"]
            # 是否一層一層解凍pretrained model
            self.progressive_unfreeze = config_json["progressive_unfreeze"]
            # 每幾epoch解凍一層模型
            self.progressive_unfreeze_step = config_json["progressive_unfreeze_step"]
            # 最多解凍幾層
            self.max_unfreeze_layer_num = config_json["max_unfreeze_layer_num"]

            # evaluate用參數
            # 0 = training set, 1 = evaluating set
            self.evaluate_or_train = config_json["evaluate_or_train"]
            self.eval_best_n = config_json["eval_best_n"]
            self.eval_waiting_queue_begin = config_json["eval_waiting_queue_begin"]
            self.eval_waiting_queue_end = (
                self.eval_best_n + self.eval_waiting_queue_begin
            )


def init():

    global config
    config = Config()

    # paths in preprocessing
    # raw data路徑
    global raw_data_path, raw_data_random_path_real, raw_data_random_path_fake
    raw_data_path = "../{}_clean.csv".format(config.subject)
    # 切分檔案用random
    raw_data_random_path_real = "../{}_random_real.txt".format(config.subject)
    raw_data_random_path_fake = "../{}_random_fake.txt".format(config.subject)

    # data paths
    global train_data_path, test_data_path
    if not os.path.isdir("../{}".format(config.embedding_type)):
        os.makedirs("../{}".format(config.embedding_type))
    train_data_path = "../{}/{}_{}_token_data_train.csv".format(
        config.embedding_type, config.embedding_type, config.subject
    )
    test_data_path = "../{}/{}_{}_token_data_test.csv".format(
        config.embedding_type, config.embedding_type, config.subject
    )
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # paths in training
    global current_folder, train_progress_path, random_file_path, eval_progress_path
    if config.with_sentiment:
        current_folder = "../{}_with_sentiment/current".format(
            config.embedding_type)
    else:
        current_folder = "../{}/current".format(config.embedding_type)
    if not os.path.isdir(current_folder):
        os.makedirs(current_folder)
    train_progress_path = "{}/train_progress.json".format(current_folder)
    random_file_path = "{}/random.txt".format(current_folder)
    eval_progress_path = "{}/eval_progress.json".format(current_folder)


def get_random_index(random_length):
    global random_index, random_file_path
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
