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

import heapq


class Evaluate_Core:
    def __init__(self):
        self.confusion_matrix = [[0, 0], [0, 0]]
        self.eval_model = util.create_desired_model()
        self.eval_model.eval()

        self.loss_func = nn.BCELoss()
        self.total_loss = 0
        self.eval_data_length = 1
        self.cur_news = 0

    def load_state_dict(self, checkpoint):
        self.eval_model.load_state_dict(checkpoint["model_state_dict"])

        self.eval_model.eval()

    def start_time(self):
        self.start = time.time()

    def set_total_length(self, length):
        self.eval_data_length = length

    def evaluate_iter(self, X, y):
        title_token = torch.tensor(eval(X[0])).to(globals.device)
        title_mask = torch.tensor(eval(X[1])).to(globals.device)
        text_token = torch.tensor(eval(X[2])).to(globals.device)
        text_mask = torch.tensor(eval(X[3])).to(globals.device)
        with torch.no_grad():
            predict = self.eval_model(title_token, title_mask,
                                      text_token, text_mask)
            loss = self.loss_func(predict.view(
                1), torch.tensor([y]).to(globals.device))
        return predict, loss.item()

    def update_total_loss(self, loss):
        self.total_loss += loss

    def update_confusion_matrix(self, truth, predict):
        self.confusion_matrix[int(truth)][predict] += 1

    def print_progress(self):
        if self.cur_news % 10 == 0:
            util.progress(
                self.start, self.cur_news, self.eval_data_length,
                self.total_loss / (self.cur_news + 1), 0
            )

    def next_news(self):
        self.cur_news += 1

    def confusion_matrix_process(self):
        avg_loss = self.total_loss / self.eval_data_length
        accuracy = (self.confusion_matrix[0][0] + self.confusion_matrix[1][1]) / (
            self.confusion_matrix[0][0]
            + self.confusion_matrix[0][1]
            + self.confusion_matrix[1][0]
            + self.confusion_matrix[1][1]
        )
        if self.confusion_matrix[1][1] + self.confusion_matrix[0][1] == 0:
            precision = "undefine"
        else:
            precision = self.confusion_matrix[1][1] / (
                self.confusion_matrix[1][1] + self.confusion_matrix[0][1]
            )
        if self.confusion_matrix[1][1] + self.confusion_matrix[1][0] == 0:
            recall = "undefine"
        else:
            recall = self.confusion_matrix[1][1] / (
                self.confusion_matrix[1][1] + self.confusion_matrix[1][0]
            )
        if precision == "undefine" or recall == "undefine":
            F1 = "undefine"
        else:
            F1 = 2 * precision * recall / (precision + recall)
        print("average loss: {}".format(avg_loss))
        print("confusion_matrix: {}".format(self.confusion_matrix))
        print(" Accuracy: {}".format(accuracy))
        print(" Precision: {}".format(precision))
        print(" Recall: {}".format(recall))
        print("F1: {}".format(F1))

        return avg_loss, accuracy, precision, recall, F1


class Evaluator:
    def __init__(self):
        self.token_data_read = pd.read_csv(globals.train_data_path).to_numpy()
        globals.get_random_index(self.token_data_read.shape[0])

        self.if_continue = os.path.isfile(globals.eval_progress_path)
        if self.if_continue:
            with open(globals.eval_progress_path, "r") as progress_json_file:
                self.progress_json = json.load(progress_json_file)
        else:
            self.progress_json = {"fold": 1, "candidate": 0}

        self.results_buffer_init()
        self.result_file_path = "{}/fold_{}/eva_result.csv".format(
            globals.current_folder, self.progress_json["fold"])

    def next_fold(self):
        self.progress_json["fold"] += 1

    def next_candidate(self):
        self.progress_json["candidate"] += 1

    def init_candidate(self):
        self.progress_json["candidate"] = 0

    def in_old_fold(self):
        return self.progress_json["fold"] < self.progress_json["fold"]

    def create_result_file(self):
        if not os.path.isfile(self.result_file_path):
            pd.DataFrame(
                {
                    "num_pt": [],
                    "Avg_Loss": [],
                    "Accuracy": [],
                    "Precision": [],
                    "Recall": [],
                    "F1": [],
                }
            ).to_csv(self.result_file_path,
                     index=False,
                     )

    def get_candidates(self):
        with open(
            "{}/fold_{}/loss.txt".format(globals.current_folder,
                                         self.progress_json["fold"]), "r"
        ) as f:
            losses = [float(i) for i in list(f.read().split("\n")[:-1])]
        best_n = globals.config.eval_best_n + globals.config.eval_waiting_queue_begin

        plot_loss = []
        sum = 0
        # 取平均Loss
        for i, loss in enumerate(losses):
            sum += loss
            if i != 0 and i % globals.config.save_every_pt == 0:
                avg_loss = sum / globals.config.save_every_pt
                plot_loss.append(avg_loss)
                sum = 0
        # 圖像化
        plt.plot(range(len(plot_loss)), plot_loss)
        plt.savefig(
            "{}/fold_{}/loss_batched.png".format(
                globals.current_folder, self.progress_json["fold"])
        )
        plt.show()
        buffer = [
            int(i) * globals.config.save_every_pt
            for i in list(map(plot_loss.index, heapq.nsmallest(best_n, plot_loss)))
        ]

        self.candidates = buffer[
            globals.config.eval_waiting_queue_begin
            + self.progress_json["candidate"]: globals.config.eval_waiting_queue_end
        ]

    def results_buffer_init(self):
        self.results_buffer = pd.DataFrame(
            {
                "num_pt": [],
                "Avg_Loss": [],
                "Accuracy": [],
                "Precision": [],
                "Recall": [],
                "F1": [],
            }
        )

    def evaluate_model(self, eval_index, eval_pt):
        checkpoint = torch.load(
            "{}/fold_{}/FND_model_{}.pt".format(
                globals.current_folder, self.progress_json["fold"], eval_pt)
        )
        evaluate_core = Evaluate_Core()
        evaluate_core.load_state_dict(checkpoint)
        evaluate_core.set_total_length(len(eval_index))

        evaluate_core.start_time()

        for i in eval_index:

            cur_news_index = globals.random_index[i]

            predict, loss = evaluate_core.evaluate_iter(
                self.token_data_read[cur_news_index][:4],
                float(self.token_data_read[cur_news_index][4])
            )
            evaluate_core.update_total_loss(loss)
            evaluate_core.update_confusion_matrix(
                self.token_data_read[cur_news_index][4], predict > 0.5)

            evaluate_core.print_progress()
            evaluate_core.next_news()

        avg_loss, accuracy, precision, recall, F1 = evaluate_core.confusion_matrix_process()
        return pd.DataFrame(
            {
                "num_pt": [eval_pt],
                "Avg_Loss": [avg_loss],
                "Accuracy": [accuracy],
                "Precision": [precision],
                "Recall": [recall],
                "F1": [F1],
            }
        )

    def save_progress(self, result):
        print("start writing result")
        result.to_csv(
            self.result_file_path,
            mode="a",
            header=False,
            index=False,
        )

        print("start saving progress")
        with open(globals.eval_progress_path, "w"
                  ) as progress_json_file:
            json.dump(
                {"fold": self.progress_json["fold"],
                    "candidate": self.progress_json["candidate"]},
                progress_json_file,
            )
        print("done saving")

    def results_buffer_append(self, result):
        self.results_buffer = self.results_buffer.append(
            result, ignore_index=True)

    def evaluate_candidates(self, eva_index):
        self.results_buffer_init()

        # 迭代候選節點
        for eval_pt in self.candidates:
            # evaluate error
            # 測試
            result = self.evaluate_model(eva_index, eval_pt)
            # 結果輸出+更新進度
            self.save_progress(result)

            self.results_buffer_append(result)
            self.next_candidate()
        print(self.results_buffer)


def evaluate():
    evaluator = Evaluator()
    kf = KFold(n_splits=5, shuffle=False)
    for _, eva_index in kf.split(globals.random_index):
        # evaluating-----------------------------------------------------------------------------------------------------------------------------
        if evaluator.in_old_fold():
            evaluator.next_fold()
            continue
        # 建立當前fold的輸出檔案
        evaluator.create_result_file()
        # 讀取loss並選出較好的節點
        evaluator.get_candidates()

        evaluator.evaluate_candidates(eva_index)

        evaluator.next_fold()
        evaluator.init_candidate()
        break

    print("end")
