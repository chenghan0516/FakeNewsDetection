import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import HingeEmbeddingLoss
from transformers import BertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 128


# Bert-Embedding
class BiGRU(nn.Module):
    def __init__(self):
        super(BiGRU, self).__init__()
        self.embedding = BertModel.from_pretrained('bert-base-uncased')
        self.cnn1 = nn.Conv1d(
            in_channels=768, out_channels=512, kernel_size=20, stride=5)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)

        self.cnn2 = nn.Conv1d(in_channels=512, out_channels=HIDDEN_SIZE)

    def forward(self, tokens, masks=None):
        # BERT
        embedded = self.embedding(tokens, attention_mask=masks)[
            "last_hidden_state"]
        cls_vector = embedded[:, 0, :].reshape(-1, 1, 768)
        # CNN
        output = self.cnn1(cls_vector)
        output = self.relu1(output)
        output = self.dropout1(output)
        # i=0
        # while i<=cls_vector.shape[0]:

        return output


class FakeNewsDetection(nn.Module):
    def __init__(self):
        super(FakeNewsDetection, self).__init__()
        self.myEmbed = BiGRU()
        self.FC_1 = nn.Linear(HIDDEN_SIZE, 64)
        self.FC_2 = nn.Linear(64, 16)
        self.FC_3 = nn.Linear(16, 1)
        self.Dropout = nn.Dropout(p=0.3)

    def forward(self, titles, title_mask, texts, text_mask):
        count = 0
        titleEmbed = torch.zeros(HIDDEN_SIZE).to(device)
        textEmbed = torch.zeros(HIDDEN_SIZE).to(device)
        newsEmbed = torch.zeros(HIDDEN_SIZE).to(device)
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
