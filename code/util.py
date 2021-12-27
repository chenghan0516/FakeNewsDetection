import pandas as pd
import numpy as np
import nltk
import time
import math
import globals
import os


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    if percent != 0:
        es = s / (percent)
        rs = es - s
        return "%s (- %s)" % (asMinutes(s), asMinutes(rs))
    else:
        return "%s (- INF)" % (asMinutes(s))


def progress(start, cur, total_len, loss, cur_read=0):
    print(
        "{} ({}%)\nloss = {}".format(
            timeSince(start, (cur - cur_read) / (total_len - cur_read)),
            (cur / total_len) * 100,
            loss,
        )
    )


# 取得對應的tokenizer
def get_tokenizer():
    print(
        "Preprocessing {} on {}.".format(
            globals.config.embedding_type, globals.config.subject
        )
    )
    if globals.config.embedding_type == "Bert":
        print("Get tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased').")

        from transformers import BertTokenizerFast

        return BertTokenizerFast.from_pretrained("bert-base-uncased")


# 輸入一段文字，返回每句的token和attention mask
def tokenize_string(input_string, tokenizer):
    if pd.notna(input_string):
        sentence_list = nltk.sent_tokenize(input_string)
        result = tokenizer(sentence_list, truncation=True,
                           padding=True, max_length=50)
        return result["input_ids"], result["attention_mask"]
    return [], []


# 預處理用亂數產生
def get_raw_random_index(real_data_num, total_data_num):
    # real
    if os.path.isfile(globals.raw_data_random_path_real):
        print("real: old random")
        with open(globals.raw_data_random_path_real, "r") as f:
            random_index_real = [int(i)
                                 for i in list(f.read().split("\n")[:-1])]
    else:
        print("real: new random")
        random_index_real = list(range(0, real_data_num))
        np.random.shuffle(random_index_real)
        f = open(globals.raw_data_random_path_real, "w")
        for i in random_index_real:
            f.write(str(i) + "\n")
        f.close()
    # fake
    if os.path.isfile(globals.raw_data_random_path_fake):
        print("fake: old random")
        with open(globals.raw_data_random_path_fake, "r") as f:
            random_index_fake = [int(i)
                                 for i in list(f.read().split("\n")[:-1])]
    else:
        print("fake: new random")
        random_index_fake = list(range(real_data_num, total_data_num))
        np.random.shuffle(random_index_fake)
        f = open(globals.raw_data_random_path_fake, "w")
        for i in random_index_fake:
            f.write(str(i) + "\n")
        f.close()

    return random_index_real, random_index_fake


# 預處理
def preprocess():
    nltk.download("popular")
    data_raw = pd.read_csv(globals.raw_data_path)
    tokenizer = get_tokenizer()
    # ------------------------------------------------------------------------------- Tokenize articles -------------------------------------------------------------------

    total_rows = data_raw.shape[0]
    token_data = pd.DataFrame(
        {
            "title_token": [],
            "title_mask": [],
            "text_token": [],
            "text_mask": [],
            "tag": [],
            "title_len": [],
            "text_len": [],
        }
    )

    real_data_num = -1

    start = time.time()
    for i, temp in data_raw.iterrows():
        if i % 10 == 0:
            progress(start, i, total_rows, "N/A")

        title_token, title_mask = tokenize_string(temp["title"], tokenizer)
        text_token, text_mask = tokenize_string(temp["text"], tokenizer)

        token_data = token_data.append(
            {
                "title_token": title_token,
                "title_mask": title_mask,
                "text_token": text_token,
                "text_mask": text_mask,
                "tag": temp["label"],
                "title_len": len(title_token),
                "text_len": len(text_token),
            },
            ignore_index=True,
        )

        if real_data_num == -1 and temp["label"] == 0:
            real_data_num = i
    fake_data_num = token_data.shape[0] - real_data_num
    print("total data number: {}".format(token_data.shape[0]))
    print("real news number: {}".format(real_data_num))
    print("fake news number: {}".format(fake_data_num))

    # ------------------------------------------------------------------------------- 切分training set跟testing set -------------------------------------------------------------------
    real_random_index, fake_random_index = get_raw_random_index(
        real_data_num, token_data.shape[0]
    )

    use_data_num = min(real_data_num, fake_data_num)
    random_index_test = (
        real_random_index[: int(use_data_num / 5)]
        + fake_random_index[: int(use_data_num / 5)]
    )

    random_index_train = (
        real_random_index[int(use_data_num / 5): use_data_num]
        + fake_random_index[int(use_data_num / 5): use_data_num]
    )

    testing_set = []
    training_set = []
    for i in range(token_data.shape[0]):
        if i in random_index_test:
            testing_set.append(True)
        else:
            testing_set.append(False)
        if i in random_index_train:
            training_set.append(True)
        else:
            training_set.append(False)

    print(token_data[testing_set])
    print(token_data[training_set])

    token_data[testing_set].to_csv(globals.test_data_path, index=False)
    token_data[training_set].to_csv(globals.train_data_path, index=False)


# 取得相對應的model
def create_desired_model():
    if globals.config.embedding_type == "Bert":
        if globals.config.model_type == "rnn":
            if globals.config.with_sentiment:
                print("import model Bert-rnn with sentiment")
                from model.Bert_RNN_sentiment_v1 import FakeNewsDetection
                return FakeNewsDetection().to(globals.device)
            else:
                print("import model Bert-rnn")
                from model.Bert_RNN import FakeNewsDetection
                return FakeNewsDetection().to(globals.device)

        elif globals.config.model_type == "cnn":
            if globals.config.with_sentiment:
                print("import model Bert-cnn with sentiment")
            else:
                print("import model Bert-cnn")
                from model.Bert_CNN import FakeNewsDetection

                return FakeNewsDetection().to(globals.device)
