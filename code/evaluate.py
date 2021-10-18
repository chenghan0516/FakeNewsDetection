import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os
import json
import torch
import globals
import heapq


def evaluate_iter(eval_model, X, y, loss_func):
    title_token = torch.tensor(eval(X[0])).to(device)
    title_mask = torch.tensor(eval(X[1])).to(device)
    text_token = torch.tensor(eval(X[2])).to(device)
    text_mask = torch.tensor(eval(X[3])).to(device)
    with torch.no_grad():
        predict = eval_model(title_token, title_mask, text_token, text_mask)
        loss = loss_func(predict.view(1), torch.tensor([y]).to(device))
    return predict, loss.item()


def evaluate_model(token_data_read, eval_index, eval_pt):
    confusion_matrix = [[0, 0], [0, 0]]

    checkpoint = torch.load(
        "{}/current/fold_{}/FND_model_{}.pt".format(path, fold, eval_pt)
    )
    eval_model = FakeNewsDetection().to(device)
    eval_model.load_state_dict(checkpoint["model_state_dict"])

    loss_func = nn.BCELoss()
    eval_model.eval()
    total_loss = 0
    cur_eval = 0
    start_eval = time.time()

    for i in eval_index:
        predict, loss = evaluate_iter(
            eval_model,
            token_data_read[random_index_read[i]][:4],
            float(token_data_read[random_index_read[i]][4]),
            loss_func,
        )
        total_loss += loss
        confusion_matrix[int(token_data_read[random_index_read[i]][4])][
            predict > 0.5
        ] += 1
        if cur_eval % 10 == 0:
            print("num in waiting_queue: {}".format(temp_waiting_queue_num))
            progress(
                start_eval, cur_eval, len(eval_index), total_loss / (cur_eval + 1), 0
            )
        cur_eval += 1

    accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (
        confusion_matrix[0][0]
        + confusion_matrix[0][1]
        + confusion_matrix[1][0]
        + confusion_matrix[1][1]
    )
    if confusion_matrix[1][1] + confusion_matrix[0][1] == 0:
        precision = "undefine"
    else:
        precision = confusion_matrix[1][1] / (
            confusion_matrix[1][1] + confusion_matrix[0][1]
        )
    if confusion_matrix[1][1] + confusion_matrix[1][0] == 0:
        recall = "undefine"
    else:
        recall = confusion_matrix[1][1] / (
            confusion_matrix[1][1] + confusion_matrix[1][0]
        )
    if precision == "undefine" or recall == "undefine":
        F1 = "undefine"
    else:
        F1 = 2 * precision * recall / (precision + recall)
    print("average loss: {}".format(total_loss / len(eval_index)))
    print("confusion_matrix: {}".format(confusion_matrix))
    print(" Accuracy: {}".format(accuracy))
    print(" Precision: {}".format(precision))
    print(" Recall: {}".format(recall))
    print("F1: {}".format(F1))
    return pd.DataFrame(
        {
            "num_pt": [eval_pt],
            "Avg_Loss": [total_loss / len(eval_index)],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1": [F1],
        }
    )


class Evaluate_Core:
    def __init__():
        print("")


class Evaluator:
    def __init__(self):
        self.token_data_read = pd.read_csv(globals.train_data_path).to_numpy()
        globals.get_random_index(self.token_data_read.shape[0])

        self.if_continue = os.path.isfile(globals.eval_progress_path)
        if self.if_continue:
            with open(globals.eval_progress_path, "r") as progress_json_file:
                self.progress_json = json.load(progress_json_file)
                self.checkpoint = torch.load(
                    "{}/fold_{}/FND_model_{}.pt".format(
                        globals.current_folder,
                        self.progress_json["fold"],
                        self.progress_json["temp_waiting_queue_num"],
                    )
                )
        else:
            self.progress_json = {"fold": 1, "temp_waiting_queue_num": 0}
            self.checkpoint = None
        self.results = pd.DataFrame(
            {
                "num_pt": [],
                "Avg_Loss": [],
                "Accuracy": [],
                "Precision": [],
                "Recall": [],
                "F1": [],
            }
        )
        self.fold = 1

    def next_fold(self):
        self.fold += 1

    def in_old_fold(self):
        return self.fold < self.progress_json["fold"]

    def create_result_file(self):
        if not os.path.isfile(
            "{}/fold_{}/eva_result.csv".format(globals.current_folder, self.fold)
        ):
            pd.DataFrame(
                {
                    "num_pt": [],
                    "Avg_Loss": [],
                    "Accuracy": [],
                    "Precision": [],
                    "Recall": [],
                    "F1": [],
                }
            ).to_csv(
                "{}/fold_{}/eva_result.csv".format(globals.current_folder, self.fold),
                index=False,
            )

    def get_candidates(self):
        with open(
            "{}/fold_{}/loss.txt".format(globals.current_folder, self.fold), "r"
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
            "{}/fold_{}/loss_batched.png".format(globals.current_folder, self.fold)
        )
        plt.show()
        self.waiting_queue = [
            int(i) * globals.config.save_every_pt
            for i in list(map(plot_loss.index, heapq.nsmallest(best_n, plot_loss)))
        ]

    def evaluate(self, eva_index):
        # 迭代候選節點

        for eval_pt in self.waiting_queue[
            globals.config.eval_waiting_queue_begin
            + self.progress_json[
                "temp_waiting_queue_num"
            ] : globals.config.eval_waiting_queue_end
        ]:
            # evaluate error
            # 測試
            result = evaluate_model(token_data_read, eva_index, eval_pt)
            # 結果輸出
            print("start writing result")
            result.to_csv(
                "{}/current/fold_{}/eva_result.csv".format(path, fold),
                mode="a",
                header=False,
                index=False,
            )
            # 更新進度
            print("start saving progress")
            with open(
                "{}/current/eva_progress.json".format(path), "w"
            ) as progress_json_file:
                json.dump(
                    {"fold": fold, "temp_waiting_queue_num": temp_waiting_queue_num},
                    progress_json_file,
                )
            print("done saving")

            results = results.append(result, ignore_index=True)
        print(results)


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

        break

    print("end")
