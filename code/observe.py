from math import nan
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
import pickle
import math

from tqdm import tqdm
import torch.nn.functional as F
from transformers import BertModel

GRU_HIDDEN_SIZE_1 = 128
GRU_HIDDEN_SIZE_2 = 64
GRU_HIDDEN_SIZE_3 = 16


class Sentiment(nn.Module):
    def __init__(self):
        super(Sentiment, self).__init__()
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=GRU_HIDDEN_SIZE_1,
            dropout=0.3,
            num_layers=1,
            bidirectional=True,
        )
        self.FC_1 = nn.Linear(GRU_HIDDEN_SIZE_1, GRU_HIDDEN_SIZE_2)
        self.FC_2 = nn.Linear(GRU_HIDDEN_SIZE_2, GRU_HIDDEN_SIZE_3)
        self.FC_3 = nn.Linear(GRU_HIDDEN_SIZE_3, 1)

    def forward(self, cls_vector):
        _, hidden = self.gru(cls_vector)
        embed = hidden[-1]

        # print(newsEmbed)
        output = self.FC_1(embed.to(globals.device))
        output = self.FC_2(F.relu(output))
        output = self.FC_3(F.relu(output))

        return F.sigmoid(output)*10


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


class Observer:
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
        self.result_file_path = "{}/fold_{}/observe_result.csv".format(
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

    def get_candidates(self, pt):
        self.candidate = pt

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
        print("done saving")

    def results_buffer_append(self, result):
        self.results_buffer = self.results_buffer.append(
            result, ignore_index=True)

    def evaluate_candidates(self, eva_index):
        self.results_buffer_init()

        # evaluate error
        # 測試
        result = self.evaluate_model(eva_index, self.candidate)
        # 結果輸出
        self.save_progress(result)

        self.results_buffer_append(result)
        self.next_candidate()
        print(self.results_buffer)

    def categorize_news(self, eva_index):
        embedding = BertModel.from_pretrained(
            'bert-base-uncased').to(globals.device)
        sentiment_pt = torch.load(
            "./model/Bert_sentiment/Sentiment_model_301000.pt", map_location=globals.device)
        eval_model = Sentiment().to(globals.device)
        eval_model.load_state_dict(sentiment_pt["model_state_dict"])
        eval_model.eval()

        news_sentiment_score_dict = {}
        for i in tqdm(eva_index):
            cur_news_index = globals.random_index[i]

            title_token = torch.tensor(
                eval(self.token_data_read[cur_news_index][0])).to(globals.device)
            title_mask = torch.tensor(
                eval(self.token_data_read[cur_news_index][1])).to(globals.device)
            text_token = torch.tensor(
                eval(self.token_data_read[cur_news_index][2])).to(globals.device)
            text_mask = torch.tensor(
                eval(self.token_data_read[cur_news_index][3])).to(globals.device)
            with torch.no_grad():
                total = 0
                total_num = 0
                predict_title = -1
                if len(title_token) != 0:
                    embedded_title = embedding(title_token, attention_mask=title_mask)[
                        "last_hidden_state"]
                    cls_vector_title = embedded_title[:,
                                                      0, :].reshape(-1, 1, 768)
                    predict_title = eval_model(
                        cls_vector_title.to(globals.device))
                    total += predict_title
                    total_num += 1

                predict_text = -1
                if len(text_token) != 0:
                    embedded_text = embedding(text_token, attention_mask=text_mask)[
                        "last_hidden_state"]
                    cls_vector_text = embedded_text[:,
                                                    0, :].reshape(-1, 1, 768)
                    predict_text = eval_model(
                        cls_vector_text.to(globals.device))
                    total += predict_text
                    total_num += 1

                if total_num != 0:
                    total /= total_num
            news_sentiment_score_dict[cur_news_index] = {
                "title_score": predict_title, "text_score": predict_text, "avg_score": total}

        with open('{}/fold_1/news_sentiment_score_dict.pickle'.format(globals.current_folder), 'wb') as f:
            pickle.dump(news_sentiment_score_dict, f)


def observe():
    observer = Observer()
    kf = KFold(n_splits=5, shuffle=False)
    for train_index, eva_index in kf.split(globals.random_index):
        # evaluating-----------------------------------------------------------------------------------------------------------------------------
        if observer.in_old_fold():
            observer.next_fold()
            continue
        # observer.get_candidates(pt)
        if not os.path.isfile('{}/fold_1/news_sentiment_score_dict_eval.pickle'.format(globals.current_folder)):
            observer.categorize_news(eva_index)
        with open('{}/fold_1/news_sentiment_score_dict_eval.pickle'.format(globals.current_folder), 'rb') as f:
            news_sentiment_score_dict = pickle.load(f)
        with open('{}/fold_1/eval_result_ID.pickle'.format(globals.current_folder), 'rb') as f:
            eval_result_ID = pickle.load(f)

        token_data_read = pd.read_csv(globals.train_data_path).to_numpy()
        sentiment_distribution = [[0, 0] for _ in range(10)]
        fake_sentiment_distribution = [[0, 0] for _ in range(10)]
        real_sentiment_distribution = [[0, 0] for _ in range(10)]
        total = len(news_sentiment_score_dict)

        for news_index in tqdm(news_sentiment_score_dict):
            score = math.floor(news_sentiment_score_dict[news_index]["avg_score"])
            evaluate_result = news_index in eval_result_ID[1]

            sentiment_distribution[score][evaluate_result] += 1
            if token_data_read[news_index][4]:
                real_sentiment_distribution[score][evaluate_result] += 1
            else:
                fake_sentiment_distribution[score][evaluate_result] += 1
        real_block = [[0, 0], [0, 0], [0, 0]]
        fake_block = [[0, 0], [0, 0], [0, 0]]
        for i in range(1, 10):
            print("score: {}, real: {}, fake: {}, total: {} ({} + {})".format(i, sum(real_sentiment_distribution[i]), sum(fake_sentiment_distribution[i]), sum(sentiment_distribution[i]), sum(real_sentiment_distribution[i])/sum(sentiment_distribution[i]), sum(fake_sentiment_distribution[i])/sum(sentiment_distribution[i])))
            if i < 4:
                real_block[0][0] += real_sentiment_distribution[i][0]
                real_block[0][1] += real_sentiment_distribution[i][1]
                fake_block[0][0] += fake_sentiment_distribution[i][0]
                fake_block[0][1] += fake_sentiment_distribution[i][1]
            elif i < 7:
                real_block[1][0] += real_sentiment_distribution[i][0]
                real_block[1][1] += real_sentiment_distribution[i][1]
                fake_block[1][0] += fake_sentiment_distribution[i][0]
                fake_block[1][1] += fake_sentiment_distribution[i][1]
            else:
                real_block[2][0] += real_sentiment_distribution[i][0]
                real_block[2][1] += real_sentiment_distribution[i][1]
                fake_block[2][0] += fake_sentiment_distribution[i][0]
                fake_block[2][1] += fake_sentiment_distribution[i][1]

        print("\nproportion")
        tp_total = 0
        fp_total = 0
        tn_total = 0
        fn_total = 0
        for i in range(3):
            local_sum = sum(real_block[i]+fake_block[i])
            tp = real_block[i][1]
            fp = fake_block[i][1]
            tn = fake_block[i][0]
            fn = real_block[i][0]
            tp_total += tp
            fp_total += fp
            tn_total += tn
            fn_total += fn
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            print("score: {}~{}, real: {}, fake: {}, total: {} ({} + {})".format(i*3+1, i*3+3, (tp+fp)/total, (tn+fn)/total,
                                                                                 local_sum/total, (tp+fp)/local_sum, (tn+fn)/local_sum))
            print("  tp: {}, fp: {}, tn: {}, fn: {}".format(tp/local_sum, fp/local_sum, tn/local_sum, fn/local_sum))
            print("  accuracy: {}, precision: {}, recall: {}, F1: {}".format((tp+tn)/local_sum, precision, recall, 2*precision*recall/(precision+recall)))
            print("")

        local_sum = tp_total+fp_total+tn_total+fn_total
        precision = tp_total/(tp_total+fp_total)
        recall = tp_total/(tp_total+fn_total)
        print("score: {}~{}, real: {}, fake: {}, total: {} ({} + {})".format(1, 9, (tp_total+fp_total)/total, (tn_total+fn_total)/total,
                                                                             local_sum/total, (tp_total+fp_total)/local_sum, (tn_total+fn_total)/local_sum))
        print("  tp: {}, fp: {}, tn: {}, fn: {}".format(tp_total/local_sum, fp_total/local_sum, tn_total/local_sum, fn_total/local_sum))
        print("  accuracy: {}, precision: {}, recall: {}, F1: {}".format((tp_total+tn_total)/local_sum, precision, recall, 2*precision*recall/(precision+recall)))
        print("")
        print("end")

        return news_sentiment_score_dict, sentiment_distribution
