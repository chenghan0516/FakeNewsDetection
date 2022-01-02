import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SENT_EMBED_SIZE = 16
GRU_HIDDEN_SIZE = 128


class Sentiment(nn.Module):
    def __init__(self):
        super(Sentiment, self).__init__()
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=128,
            dropout=0.3,
            num_layers=1,
            bidirectional=True,
        )
        self.FC_1 = nn.Linear(128, 64)
        self.FC_2 = nn.Linear(64, 16)
        self.FC_3 = nn.Linear(16, 1)

    def forward(self, cls_vector, h_0):
        _, hidden = self.gru(cls_vector, h_0)
        embed = hidden[-1]

        # print(newsEmbed)
        output = self.FC_1(embed.to(device))
        output = self.FC_2(F.relu(output))
        # output = self.FC_3(F.relu(output))

        return output

# Bert-Embedding


class BiGRU(nn.Module):
    def __init__(self):
        super(BiGRU, self).__init__()
        self.embedding = BertModel.from_pretrained('bert-base-uncased')
        sentiment_pt = torch.load(
            "./model/Bert_sentiment/Sentiment_model_301000.pt", map_location=device)
        self.sentiment_embed = Sentiment()
        self.sentiment_embed.load_state_dict(sentiment_pt["model_state_dict"])
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=GRU_HIDDEN_SIZE,
            dropout=0.3,
            num_layers=1,
            bidirectional=True,
        )

    def forward(self, tokens, masks=None):
        # BERT
        embedded = self.embedding(tokens, attention_mask=masks)[
            "last_hidden_state"]
        cls_vector = embedded[:, 0, :].reshape(-1, 1, 768)
        sentiment_hidden = torch.zeros(2, 1, 128).to(device)

        # GRU
        s_embed = self.sentiment_embed(cls_vector, sentiment_hidden)
        _, hidden = self.gru(cls_vector)
        hidden = torch.cat((hidden[-1], s_embed.to(device)), dim=1)

        return hidden


class FakeNewsDetection(nn.Module):
    def __init__(self):
        super(FakeNewsDetection, self).__init__()
        self.myEmbed = BiGRU()
        self.FC_1 = nn.Linear(GRU_HIDDEN_SIZE+SENT_EMBED_SIZE, 64)
        self.FC_2 = nn.Linear(64, 16)
        self.FC_3 = nn.Linear(16, 1)
        self.Dropout = nn.Dropout(p=0.3)

    def forward(self, titles, title_mask, texts, text_mask):
        count = 0
        titleEmbed = torch.zeros(GRU_HIDDEN_SIZE+SENT_EMBED_SIZE).to(device)
        textEmbed = torch.zeros(GRU_HIDDEN_SIZE+SENT_EMBED_SIZE).to(device)
        newsEmbed = torch.zeros(GRU_HIDDEN_SIZE+SENT_EMBED_SIZE).to(device)
        if len(titles) != 0:
            titleEmbed = self.myEmbed(titles, title_mask)
            count += 1
        if len(texts) != 0:
            textEmbed = self.myEmbed(texts, text_mask)
            count += 1
        if count != 0:
            newsEmbed = (titleEmbed+textEmbed)/count

        output = self.FC_1(newsEmbed.to(device))
        output = self.FC_2(F.relu(output))
        output = self.FC_3(F.relu(output))

        return F.sigmoid(output)
