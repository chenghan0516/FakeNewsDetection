from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import globals
import os
import json
import time
import util
import matplotlib.pyplot as plt


class Train_Core:
    def __init__(self):
        self.FND_model = util.create_desired_model()
        self.optimizer = torch.optim.AdamW(
            self.FND_model.parameters(), lr=globals.config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=globals.config.scheduler_gamma
        )
        self.loss_func = nn.BCELoss()

        self.losses = []
        self.cur_news = 0
        self.freezed_pretrain_layer_num_temp = globals.config.freezed_pretrain_layer_num
        self.cur_unfreezed_layer_num = 1

    def load_state_dict(self, checkpoint):
        self.FND_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def into_train_state(self):
        self.FND_model.train()

    def start_time(self):
        self.start = time.time()

    def open_loss_file_to_write(self, if_continue, fold):
        if if_continue:  # 若接續訓練
            self.f = open(
                "{}/fold_{}/loss.txt".format(globals.current_folder, fold), "a"
            )
        else:  # 若非接續訓練
            self.f = open(
                "{}/fold_{}/loss.txt".format(globals.current_folder, fold), "w"
            )

    def close_loss_file(self):
        self.f.close()

    def freeze_layers(self, cur_epoch):
        modules = [self.FND_model.myEmbed.embedding]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        if cur_epoch >= globals.config.end_warmup:
            modules = [
                self.FND_model.myEmbed.embedding.encoder.layer[
                    self.freezed_pretrain_layer_num_temp :
                ]
            ]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = True
            if (
                globals.config.progressive_unfreeze
                and self.cur_unfreezed_layer_num < globals.config.max_unfreeze_layer_num
                and (cur_epoch - globals.config.end_warmup + 1)
                % globals.config.progressive_unfreeze_step
                == 0
            ):
                self.freezed_pretrain_layer_num_temp -= 1
                self.cur_unfreezed_layer_num = (
                    globals.config.freezed_pretrain_layer_num
                    - self.freezed_pretrain_layer_num_temp
                    + 1
                )

    def in_old_progress(self, target):
        return self.cur_news <= target

    def next_news(self):
        self.cur_news += 1

    def train_iter(self, X, y):
        self.optimizer.zero_grad()
        title_token = torch.tensor(eval(X[0])).to(globals.device)
        title_mask = torch.tensor(eval(X[1])).to(globals.device)
        text_token = torch.tensor(eval(X[2])).to(globals.device)
        text_mask = torch.tensor(eval(X[3])).to(globals.device)
        predict = self.FND_model(title_token, title_mask, text_token, text_mask)
        loss = self.loss_func(predict.view(1), torch.tensor([y]).to(globals.device))
        # print("predict: {}\nans:     {}\n loss: {}".format(predict,torch.tensor([y]),loss))
        loss.backward()
        self.optimizer.step()
        return predict, loss.item()

    def push_loss(self, loss):
        self.losses.append(loss)

    def if_save_state_dict(self):
        return self.cur_news != 0 and self.cur_news % globals.config.save_every_pt == 0

    def save_progress(self, fold):
        # 模型
        print("start save checkpoint")
        torch.save(
            {
                "cur": self.cur_news,
                "model_state_dict": self.FND_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            "{}/fold_{}/FND_model_{}.pt".format(
                globals.current_folder, fold, self.cur_news
            ),
        )
        # loss
        print("start save loss")
        for i in self.losses:
            self.f.write(str(i) + "\n")
        self.losses = []
        # 更新progress
        print("start save progress")
        with open(globals.train_progress_path, "w") as progress_json_file:
            json.dump({"fold": fold, "cur": self.cur_news}, progress_json_file)
        print("done saving")

    def print_progress(self, total_news_count, loss, old_progress):
        if self.cur_news % 10 == 0:
            util.progress(
                self.start, self.cur_news, total_news_count, loss, old_progress
            )

    def scheduler_step(self, old_progress):
        if self.cur_news > old_progress:
            self.scheduler.step()

    def if_skip_long_news(self):
        return self.cur_unfreezed_layer_num > 2


class Trainer:
    def __init__(self):
        self.token_data_read = pd.read_csv(globals.train_data_path).to_numpy()
        globals.get_random_index(self.token_data_read.shape[0])
        self.if_continue = os.path.isfile(globals.train_progress_path)
        if self.if_continue:
            with open(globals.train_progress_path, "r") as progress_json_file:
                self.progress_json = json.load(progress_json_file)
                self.checkpoint = torch.load(
                    "{}/fold_{}/FND_model_{}.pt".format(
                        globals.current_folder,
                        self.progress_json["fold"],
                        self.progress_json["cur"],
                    )
                )
        else:
            self.progress_json = {"fold": 1, "cur": -1}
            self.checkpoint = None
        self.fold = 1

    def next_fold(self):
        self.fold += 1

    def in_old_fold(self):
        return self.fold < self.progress_json["fold"]

    def create_fold_folder(self):
        if not os.path.isdir("{}/fold_{}".format(globals.current_folder, self.fold)):
            os.makedirs("{}/fold_{}".format(globals.current_folder, self.fold))

    def init_fold_progress(self):
        self.progress_json["cur"] = -1
        self.if_continue = False

    def plot_loss(self):
        f = open("{}/fold_{}/loss.txt".format(globals.current_folder, self.fold), "r")
        losses = [float(i) for i in list(f.read().split("\n")[:-1])]
        f.close()
        plt.plot(range(len(losses)), losses)
        plt.savefig("{}/fold_{}/loss.png".format(globals.current_folder, self.fold))
        plt.show()

    def train_model(self, train_index):
        train_core = Train_Core()
        if self.if_continue:
            train_core.load_state_dict(self.checkpoint)

        train_core.into_train_state()
        train_core.start_time()
        train_core.open_loss_file_to_write(self.if_continue, self.fold)

        for epoch in range(globals.config.epoch):
            # Freeze layers
            train_core.freeze_layers(epoch)
            # 迭代訓練資料
            for i in train_index:
                # 訓練pretrained model時跳過過長的文章
                if (
                    epoch >= globals.config.end_warmup
                    and train_core.if_skip_long_news()
                    and len(eval(self.token_data_read[globals.random_index[i]][2]))
                    > 500
                ):
                    print(
                        "skip: {}".format(
                            len(eval(self.token_data_read[globals.random_index[i]][2]))
                        )
                    )
                    continue
                # 走到上次進度
                if train_core.in_old_progress(self.progress_json["cur"]):
                    train_core.next_news()
                    continue

                # 訓練
                predict, loss = train_core.train_iter(
                    self.token_data_read[globals.random_index[i]][:4],
                    float(self.token_data_read[globals.random_index[i]][4]),
                )
                train_core.push_loss(loss)

                # 紀錄模型
                if train_core.if_save_state_dict():
                    train_core.save_progress(self.fold)

                # 顯示進度
                train_core.print_progress(
                    train_index.shape[0] * globals.config.epoch,
                    loss,
                    self.progress_json["cur"],
                )
                train_core.next_news()

            train_core.scheduler_step(self.progress_json["cur"])
        train_core.close_loss_file()

        self.plot_loss()


def train():
    trainer = Trainer()
    kf = KFold(n_splits=5, shuffle=False)
    for train_index, _ in kf.split(globals.random_index):
        # training-----------------------------------------------------------------------------------------------------------------------------
        if trainer.in_old_fold():
            trainer.next_fold()
            continue
        trainer.create_fold_folder()

        trainer.train_model(train_index)

        trainer.init_fold_progress()
        trainer.next_fold()

        break

    print("end")
