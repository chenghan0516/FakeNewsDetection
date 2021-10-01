import pandas as pd
import numpy as np
import nltk
import time
import math
import globals


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
    print("{} ({}%)\nloss = {}".format(
        timeSince(start, (cur-cur_read) / (total_len-cur_read)),
        (cur / total_len) * 100,
        loss
    )
    )


class Config:
    def __init__(self, subject, model_type, with_sentiment, epoch, end_warmup, lr, scheduler_gamma, save_every_pt, freezed_bert_layer_num, progressive_unfreeze, progressive_unfreeze_step, evaluate_or_train, eval_best_n, eval_waiting_queue_begin):

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

        # evaluate用參數
        # 0 = training set, 1 = evaluating set
        self.evaluate_or_train = evaluate_or_train
        self.eval_best_n = eval_best_n
        self.eval_waiting_queue_begin = eval_waiting_queue_begin
        self.eval_waiting_queue_end = self.eval_best_n+self.eval_waiting_queue_begin


# 輸入一段文字，返回每句的token和attention mask
def tokenize_string_list(input_string, tokenizer):
    if pd.notna(input_string):
        sentence_list = nltk.sent_tokenize(input_string)
        result = tokenizer(sentence_list, truncation=True,
                           padding=True, max_length=50)
        return result["input_ids"], result["attention_mask"]


# 取得對應的tokenizer
def get_tokenizer():
    print("Preprocessing {} on {}.".format(
        globals.config.model_type, globals.config.subject))
    if globals.config.model_type == "Bert":
        print(
            "Get tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased'.")

        from transformers import BertTokenizerFast
        return BertTokenizerFast.from_pretrained('bert-base-uncased')


# 開始預處理
def preprocess():
    nltk.download("popular")
    data_raw = pd.read_csv(globals.raw_data_path)
    tokenizer = get_tokenizer()
    # ------------------------------------------------------------------------------- Tokenize articles -------------------------------------------------------------------

    total_rows = data_raw.shape[0]
    token_data = pd.DataFrame({"title_token": [], "title_mask": [], "text_token": [
    ], "text_mask": [], "tag": [], "title_len": [], "text_len": []})

    real_data_num = -1

    start = time.time()
    for i, temp in data_raw.iterrows():
        if(i % 10 == 0):
            progress(start, i, total_rows, "N/A")

        title_token, title_mask = tokenize_string_list(
            temp["title"], tokenizer)
        text_token, text_mask = tokenize_string_list(temp["text"], tokenizer)

        token_data = token_data.append({"title_token": title_token, "title_mask": title_mask,
                                        "text_token": text_token, "text_mask": text_mask,
                                        "tag": temp["label"], "title_len": len(title_token), "text_len": len(text_token)}, ignore_index=True)

        if real_data_num == -1 and temp["label"] == 0:
            real_data_num = i
    fake_data_num = token_data.shape[0]-real_data_num
    print("total data number: {}".format(token_data.shape[0]))
    print("real news number: {}".format(real_data_num))
    print("fake news number: {}".format(fake_data_num))

    # ------------------------------------------------------------------------------- 切分training set跟testing set -------------------------------------------------------------------
    real_random_index = list(range(0, real_data_num))
    np.random.shuffle(real_random_index)

    fake_random_index = list(range(real_data_num, token_data.shape[0]))
    np.random.shuffle(fake_random_index)

    use_data_num = min(real_data_num, fake_data_num)
    random_index_test = real_random_index[:int(use_data_num/5)] + \
        fake_random_index[:int(use_data_num/5)]

    random_index_train = real_random_index[int(use_data_num/5):use_data_num] + \
        fake_random_index[int(use_data_num/5):use_data_num]

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