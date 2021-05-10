#%%
import os
import torch
import pandas as pd
import json
import random
import copy
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

import matplotlib.pyplot as plt

plt.switch_backend("agg")
import matplotlib.ticker as ticker
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

path = {
    "politifact_real.txt": "politifact\\real",
    "politifact_fake.txt": "politifact\\fake",
    "gossipcop_real.txt": "gossipcop\\real",
    "gossipcop_fake.txt": "gossipcop\\fake",
}

text_list = {
    "politifact_real.txt": [],
    "politifact_fake.txt": [],
    "gossipcop_real.txt": [],
    "gossipcop_fake.txt": [],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenized_paragraphs = []

result_path = "myFile\\modify_result\\"
EMBEDDING_DIM = 32
CONTEXT_SIZE = 11
PARAGRAPH_MAX_LENGTH = 2000
PARAGRAPH_PARSE_NUM = 3
HIDDEN_SIZE = 64
realize_learning_rate = (1e-3) * 2
judge_learning_rate = 1e-3
EPOCH = 5
EXTRA_JUDGE_TRAIN = 1
REALIZE_BATCH_SIZE = 5
JUDGE_BATCH_SIZE = 10
JUDGE_HIDDEN_SIZE = 64


def load_content(fileName):
    for content in open(fileName, "r"):
        j = json.load(
            open(path[fileName] + "\\" + content[:-1] + "\\news content.json", "r")
        )
        text_buffer = (j["title"] + "\n" + j["text"]).lower()
        text_buffer = text_buffer.replace("‘", "'")
        text_buffer = text_buffer.replace("’", "'")
        text_buffer = text_buffer.replace("“", '"')
        text_buffer = text_buffer.replace("”", '"')
        text_list[fileName].append(text_buffer)


def make_realize_context_and_ans(index_list, context_size, upper_bound):
    context = []
    ans = []
    tag = []
    cut = (context_size - 1) / 2
    for paragraph_index in index_list:
        if (
            len(tokenized_paragraphs[paragraph_index]["content"]) < context_size
            or len(tokenized_paragraphs[paragraph_index]["content"]) > upper_bound
        ):
            continue
        context.append([])
        ans.append([])
        for i in range(
            int(cut), len(tokenized_paragraphs[paragraph_index]["content"]) - int(cut)
        ):
            context[-1].append(
                [
                    vocab_to_ix[w]
                    for w in tokenized_paragraphs[paragraph_index]["content"][
                        i - int(cut) : i
                    ]
                ]
                + [
                    vocab_to_ix[w]
                    for w in tokenized_paragraphs[paragraph_index]["content"][
                        i + 1 : i + int(cut) + 1
                    ]
                ]
            )
            ans[-1].append(
                [vocab_to_ix[tokenized_paragraphs[paragraph_index]["content"][i]]]
            )
        tag.append(tokenized_paragraphs[paragraph_index]["tag"])
    return context, ans, tag


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


def showPlot(points, str, min_y, max_y):
    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(str)
    ax.set_ylabel("loss")
    ax.set_ylim([min_y, max_y])
    # loc = ticker.MultipleLocator(base=0.05)
    # ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(result_path + str + ".png")
    plt.savefig(result_path + str + ".jpg")


def progress(item_name, item, start, input_list, input, cur_epoch):
    print(
        "%s (%f%%)\naverage %s in current epoch(%d/%d) so far: %.4f"
        % (
            timeSince(
                start,
                (len(input_list) * cur_epoch + input_list.index(input) + 1)
                / (len(input_list) * EPOCH),
            ),
            (
                (len(input_list) * cur_epoch + input_list.index(input) + 1)
                / (len(input_list) * EPOCH)
            )
            * 100,
            item_name,
            cur_epoch,
            EPOCH,
            sum(item) / len(item),
        )
    )


def trainRealize(
    input_tensor, target_tensor, realize_model, realize_optimizer, realize_loss_func,
):
    # hn = realize_model.initHidden()
    realize_optimizer.zero_grad()
    predict, hn = realize_model(input_tensor)  # , hn)
    loss_r = realize_loss_func(
        predict.view(-1, predict.size()[-1]), target_tensor.view(-1)
    )
    loss_r.backward()
    realize_optimizer.step()
    return loss_r.item()


def trainJudge(
    input_list,  # JUDGE_BATCH_SIZE paragraph included
    tag_list,
    realize_model,
    judge_model,
    judge_optimizer,
    judge_loss_func,
    pre_train_judge_table,
):
    judge_optimizer.zero_grad()
    buffer = []
    for input_paragraph in input_list:
        temp_buffer = [[[0] * HIDDEN_SIZE] * REALIZE_BATCH_SIZE] * 2
        hn = None
        parse_len = int(len(input_paragraph) / PARAGRAPH_PARSE_NUM)
        parse_num = 0
        i = 0
        while parse_len != 0 and (i + parse_len) < (len(input_paragraph) - 1):
            parse_num += 1
            temp_paragraph = input_paragraph[i : i + parse_len]
            temp_paragraph = torch.tensor(temp_paragraph).cuda()
            dummy, hn = realize_model(temp_paragraph, hn)
            for j in range(len(hn)):
                for k in range(len(hn[0])):
                    for m in range(len(hn[0][0])):
                        temp_buffer[j][k][m] += hn[j][k][m]
            i += parse_len
        parse_num += 1
        temp_paragraph = input_paragraph[i : len(input_paragraph)]
        temp_paragraph = torch.tensor(temp_paragraph).cuda()
        dummy, hn = realize_model(temp_paragraph, hn)
        for j in range(len(hn)):
            for k in range(len(hn[0])):
                for m in range(len(hn[0][0])):
                    temp_buffer[j][k][m] += hn[j][k][m]
                    temp_buffer[j][k][m] = temp_buffer[j][k][m] / parse_num
        buffer.append(temp_buffer)

    judge = judge_model(torch.tensor(buffer).view(JUDGE_BATCH_SIZE, -1).cuda().detach())
    train_judge_table = pre_train_judge_table
    for i, j in zip(judge.tolist(), tag_list):
        train_judge_table[int(j)][int(i[0] > 0.5)] += 1
    loss_j = judge_loss_func(judge, torch.tensor([[i] for i in tag_list]).cuda()).cuda()
    loss_j.backward()
    judge_optimizer.step()
    return loss_j.item(), train_judge_table


def trainIters(
    realize_model,
    judge_model,
    train_realize_input_list,
    train_realize_target_list,
    train_judge_input_list,
    train_judge_tag_lis,
    start,
):
    realize_loss_func = nn.NLLLoss().cuda()
    judge_loss_func = nn.MSELoss().cuda()
    realize_optimizer = optim.Adam(realize_model.parameters(), lr=realize_learning_rate)
    judge_optimizer = optim.Adam(judge_model.parameters(), lr=judge_learning_rate)

    realize_losses = []
    judge_losses = []
    train_judge_table = [[0, 0], [0, 0]]
    for i in range(EPOCH):
        for train_realize_input, train_realize_target in zip(
            train_realize_input_list, train_realize_target_list
        ):
            # print(len(train_realize_input))
            loss_r = trainRealize(
                torch.tensor(train_realize_input).cuda(),
                torch.tensor(train_realize_target).cuda(),
                realize_model,
                realize_optimizer,
                realize_loss_func,
            )
            realize_losses.append(loss_r)
            if (
                train_realize_input_list.index(train_realize_input)
                % (5 * REALIZE_BATCH_SIZE)
                == 0
            ):
                progress(
                    "realize_loss",
                    realize_losses,
                    start,
                    train_realize_input_list,
                    train_realize_input,
                    i,
                )
        for j in range(EXTRA_JUDGE_TRAIN):
            for train_judge_input, train_judge_tag in zip(
                train_judge_input_list, train_judge_tag_list
            ):
                loss_j, train_judge_table = trainJudge(
                    train_judge_input,
                    train_judge_tag,
                    realize_model,
                    judge_model,
                    judge_optimizer,
                    judge_loss_func,
                    train_judge_table,
                )
                judge_losses.append(loss_j)
                if (
                    train_judge_input_list.index(train_judge_input)
                    % (1 * JUDGE_BATCH_SIZE)
                    == 0
                ):
                    progress(
                        "judge_loss",
                        judge_losses,
                        start,
                        train_judge_input_list,
                        train_judge_input,
                        i,
                    )
    return realize_losses, judge_losses, train_judge_table


def test(
    realize_model,
    judge_model,
    test_realize_input_list,
    test_realize_target_list,
    test_judge_input_list,
    test_judge_tag_list,
):
    with torch.no_grad():
        realize_loss_func = nn.NLLLoss().cuda()
        judge_loss_func = nn.MSELoss().cuda()
        loss_j = 0
        result_judge_table = [[0, 0], [0, 0]]
        realize_losses = []
        judge_losses = []
        for test_realize_input, test_realize_target in zip(
            test_realize_input_list, test_realize_target_list
        ):
            # hn = realize_model.initHidden()
            predict, hn = realize_model(
                torch.tensor(test_realize_input).cuda()
            )  # , hn)
            loss_r = realize_loss_func(
                predict.view(-1, predict.size()[-1]),
                torch.tensor(test_realize_target).view(-1).cuda(),
            )
            realize_losses.append(loss_r)

        for test_judge_input, test_judge_tag in zip(
            test_judge_input_list, test_judge_tag_list
        ):
            buffer = []
            for input_paragraph in test_judge_input:
                temp_buffer = [[[0] * HIDDEN_SIZE] * REALIZE_BATCH_SIZE] * 2
                hn = None
                parse_len = int(len(input_paragraph) / PARAGRAPH_PARSE_NUM)
                parse_num = 0
                i = 0
                while parse_len != 0 and (i + parse_len) < (len(input_paragraph) - 1):
                    parse_num += 1
                    temp_paragraph = torch.tensor(
                        input_paragraph[i : i + parse_len]
                    ).cuda()
                    dummy, hn = realize_model(temp_paragraph, hn)
                    for j in range(len(hn)):
                        for k in range(len(hn[0])):
                            for m in range(len(hn[0][0])):
                                temp_buffer[j][k][m] += hn[j][k][m]
                    i += parse_len
                parse_num += 1
                temp_paragraph = torch.tensor(
                    input_paragraph[i : len(input_paragraph)]
                ).cuda()
                dummy, hn = realize_model(temp_paragraph, hn)
                for j in range(len(hn)):
                    for k in range(len(hn[0])):
                        for m in range(len(hn[0][0])):
                            temp_buffer[j][k][m] += hn[j][k][m]
                            temp_buffer[j][k][m] = temp_buffer[j][k][m] / parse_num
                buffer.append(temp_buffer)

            judge = judge_model(torch.tensor(buffer).view(JUDGE_BATCH_SIZE, -1).cuda())
            for judge_temp, tag_temp in zip(judge, test_judge_tag):
                result_judge_table[int(tag_temp)][int(judge_temp > 0.5)] += 1
            loss_j = judge_loss_func(
                judge, torch.tensor([[i] for i in test_judge_tag]).cuda()
            ).cuda()
            judge_losses.append(loss_j)

        return realize_losses, judge_losses, result_judge_table


def result_record(
    file_num,
    r_losses,
    j_losses,
    train_table,
    test_realize_losses,
    test_judge_losses,
    result_table,
):
    # text
    train_r = open(result_path + "train_r_" + str(file_num) + ".txt", "w")
    train_j = open(result_path + "train_j_" + str(file_num) + ".txt", "w")
    test_r = open(result_path + "test_r_" + str(file_num) + ".txt", "w")
    test_j = open(result_path + "test_j_" + str(file_num) + ".txt", "w")
    result = open(result_path + "result_" + str(file_num) + ".txt", "w")

    for r_loss in r_losses:
        train_r.write(str(r_loss) + "\n")
    for j_loss in j_losses:
        train_j.write(str(j_loss) + "\n")
    for r_loss in test_realize_losses:
        test_r.write(str(r_loss) + "\n")
    for j_loss in test_judge_losses:
        test_j.write(str(j_loss) + "\n")

    actual_total_train_num = sum(train_table[0]) + sum(train_table[1])
    result.write("train record: \n")
    result.write("  actual total train number: %d\n" % actual_total_train_num)
    result.write("  average r_loss: %.4f\n" % (sum(r_losses) / len(r_losses)))
    result.write("  average j_loss: %.4f\n" % (sum(j_losses) / len(j_losses)))

    result.write(
        "    True Positive: %.2f (%d)\n"
        % (train_table[1][1] / actual_total_train_num, train_table[1][1],)
    )
    result.write(
        "    False Positive: %.2f (%d)\n"
        % (train_table[0][1] / actual_total_train_num, train_table[0][1],)
    )
    result.write(
        "    True Negative:  %.2f (%d)\n"
        % (train_table[0][0] / actual_total_train_num, train_table[0][0],)
    )
    result.write(
        "    False Negative:  %.2f (%d)\n"
        % (train_table[1][0] / actual_total_train_num, train_table[1][0],)
    )

    actual_total_test_num = sum(result_table[0]) + sum(result_table[1])
    result.write("test record: \n")
    result.write("  actual total test number: %d\n" % actual_total_test_num)
    result.write(
        "  average r_loss: %.4f\n"
        % (sum(test_realize_losses) / len(test_realize_losses))
    )
    result.write(
        "  average j_loss: %.4f\n" % (sum(test_judge_losses) / len(test_judge_losses))
    )
    result.write(
        "    True Positive: %.2f (%d)\n"
        % (result_table[1][1] / actual_total_test_num, result_table[1][1],)
    )
    result.write(
        "    False Positive: %.2f (%d)\n"
        % (result_table[0][1] / actual_total_test_num, result_table[0][1],)
    )
    result.write(
        "    True Negative: %.2f (%d)\n"
        % (result_table[0][0] / actual_total_test_num, result_table[0][0],)
    )
    result.write(
        "    False Negative: %.2f (%d)\n"
        % (result_table[1][0] / actual_total_test_num, result_table[1][0],)
    )

    train_r.close()
    train_j.close()
    test_r.close()
    test_j.close()
    result.close()

    # graph
    plot_r = []
    plot_r_temp = []
    for r_loss in r_losses:
        plot_r_temp.append(r_loss)
        if len(plot_r_temp) == 5:
            plot_r.append(sum(plot_r_temp) / 5)
            plot_r_temp = []
    showPlot(plot_r, "train realize loss_" + str(file_num), 0, 15)
    plot_j = []
    plot_j_temp = []
    for j_loss in j_losses:
        plot_j_temp.append(j_loss)
        if len(plot_j_temp) == 5:
            plot_j.append(sum(plot_j_temp) / 5)
            plot_j_temp = []
    showPlot(plot_j, "train judge loss_" + str(file_num), 0, 0.4)


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_size):
        super(NGramLanguageModeler, self).__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru1 = nn.GRU(
            (context_size - 1) * embedding_dim, hidden_size, bidirectional=True
        )
        self.linear2 = nn.Linear(hidden_size * 2, vocab_size)
        self.normalization = nn.LayerNorm(hidden_size)

    def forward(self, inputs, hidden=None):
        embeds = self.embeddings(inputs)
        out, hidden = self.gru1(
            embeds.view(-1, REALIZE_BATCH_SIZE, (CONTEXT_SIZE - 1) * EMBEDDING_DIM),
            hidden,
        )
        hidden = self.normalization(hidden)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.log_softmax(out, dim=2)
        return out, hidden

    def initHidden(self):
        return torch.randn(2, REALIZE_BATCH_SIZE, self.hidden_size, device=device)


class IsFake(nn.Module):
    def __init__(
        self, hidden_size, judge_batch_size, judge_hidden_size=JUDGE_HIDDEN_SIZE
    ):
        super(IsFake, self).__init__()
        self.input_size = hidden_size * 2 * REALIZE_BATCH_SIZE
        self.hidden12_size = judge_hidden_size
        self.hidden23_size = int(judge_hidden_size / 4)
        self.linear1 = nn.Linear(self.input_size, self.hidden12_size)
        self.linear2 = nn.Linear(self.hidden12_size, self.hidden23_size)
        self.linear3 = nn.Linear(self.hidden23_size, 1)

    def forward(self, inputs):
        out = self.linear1(inputs)
        out = self.linear2(F.relu(out))
        out = self.linear3(F.relu(out))
        out = F.sigmoid(out)
        return out

    def initHidden(self):
        return torch.randn(2, REALIZE_BATCH_SIZE, self.hidden_size, device=device)


if __name__ == "__main__":
    load_content("politifact_real.txt")
    load_content("politifact_fake.txt")
    # load_content("gossipcop_real.txt")
    # load_content("gossipcop_fake.txt")

    true_content_list = text_list[
        "politifact_real.txt"
    ]  # + text_list["gossipcop_real.txt"]
    false_content_list = text_list[
        "politifact_fake.txt"
    ]  # + text_list["gossipcop_fake.txt"]

    vocab_set = set()
    tokenized_paragraphs = []
    for paragraph in true_content_list:
        tokenized_paragraphs.append({"content": [], "tag": True})
        for word in word_tokenize(paragraph):
            if word.isalpha():
                tokenized_paragraphs[-1]["content"].append(word)
                vocab_set.add(word)
    for paragraph in false_content_list:
        tokenized_paragraphs.append({"content": [], "tag": False})
        for word in word_tokenize(paragraph):
            if word.isalpha():
                tokenized_paragraphs[-1]["content"].append(word)
                vocab_set.add(word)
    vocab_to_ix = {word: i for i, word in enumerate(vocab_set)}

    # with torch.autograd.set_detect_anomaly(True):

    file_num = 0
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(1, 1000))
    # KFold
    for train_index, test_index in kf.split(tokenized_paragraphs):
        file_num += 1

        realize = NGramLanguageModeler(
            len(vocab_set), EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_SIZE
        ).cuda()
        judge = IsFake(HIDDEN_SIZE, JUDGE_BATCH_SIZE, JUDGE_HIDDEN_SIZE).cuda()

        random.shuffle(train_index)
        r_losses = []
        j_losses = []

        # make train tensors
        context_list, target_list, tag_list = make_realize_context_and_ans(
            train_index, CONTEXT_SIZE, PARAGRAPH_MAX_LENGTH
        )
        train_realize_input_list = []
        train_realize_target_list = []
        train_judge_input_list = []
        train_judge_tag_list = []
        batch_tag = []
        for context_paragraph, target_paragraph, tag in zip(
            context_list, target_list, tag_list
        ):
            if len(context_paragraph) < REALIZE_BATCH_SIZE:
                continue
            train_realize_input_list.append([])
            train_realize_target_list.append([])
            batch_context = []
            batch_target = []
            for context, target in zip(context_paragraph, target_paragraph):
                batch_context.append(context)
                batch_target.append(target)
                if len(batch_context) == REALIZE_BATCH_SIZE:
                    train_realize_input_list[-1].append(batch_context)
                    train_realize_target_list[-1].append(batch_target)
                    batch_context = []
                    batch_target = []
            batch_tag.append(float(tag))
            if len(batch_tag) == JUDGE_BATCH_SIZE:
                train_judge_tag_list.append(batch_tag)
                train_judge_input_list.append(
                    train_realize_input_list[-JUDGE_BATCH_SIZE:]
                )
                batch_tag = []

        # make test tensors
        context_list, target_list, tag_list = make_realize_context_and_ans(
            test_index, CONTEXT_SIZE, PARAGRAPH_MAX_LENGTH
        )
        test_realize_input_list = []
        test_realize_target_list = []
        test_judge_input_list = []
        test_judge_tag_list = []
        batch_tag = []
        for context_paragraph, target_paragraph, tag in zip(
            context_list, target_list, tag_list
        ):
            if len(context_paragraph) < REALIZE_BATCH_SIZE:
                continue
            test_realize_input_list.append([])
            test_realize_target_list.append([])
            batch_context = []
            batch_target = []
            for context, target in zip(context_paragraph, target_paragraph):
                batch_context.append(context)
                batch_target.append(target)
                if len(batch_context) == REALIZE_BATCH_SIZE:
                    test_realize_input_list[-1].append(batch_context)
                    test_realize_target_list[-1].append(batch_target)
                    batch_context = []
                    batch_target = []
            batch_tag.append(float(tag))
            if len(batch_tag) == JUDGE_BATCH_SIZE:
                test_judge_tag_list.append(batch_tag)
                test_judge_input_list.append(
                    test_realize_input_list[-JUDGE_BATCH_SIZE:]
                )
                batch_tag = []

        start = time.time()

        r_losses, j_losses, train_table = trainIters(
            realize,
            judge,
            train_realize_input_list,
            train_realize_target_list,
            train_judge_input_list,
            train_judge_tag_list,
            start,
        )

        test_realize_losses, test_judge_losses, result_table = test(
            realize,
            judge,
            test_realize_input_list,
            test_realize_target_list,
            test_judge_input_list,
            test_judge_tag_list,
        )

        result_record(
            file_num,
            r_losses,
            j_losses,
            train_table,
            test_realize_losses,
            test_judge_losses,
            result_table,
        )
        # break
# %%
