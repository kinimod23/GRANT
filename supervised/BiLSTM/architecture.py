import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F


class NetLstm(nn.Module):
    def __init__(self, vocab_size, label_size, batch_size, hidden_dim=128, use_gpu=True):
        super(NetLstm, self).__init__()
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, 200, padding_idx=0)
        self.lstm = nn.LSTM(200, self.hidden_dim, dropout=0.3, bidirectional=True, num_layers=2)
        self.hidden2label = nn.Linear(4 * self.hidden_dim, label_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(4, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(4, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(4, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(4, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, text):
        embeds = self.word_embeddings(text)
        embeds = embeds.view(text.size(0), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        max_pool = F.adaptive_max_pool1d(lstm_out.permute(1, 2, 0), 1).view(self.batch_size, -1)
        avg_pool = F.adaptive_avg_pool1d(lstm_out.permute(1, 2, 0), 1).view(self.batch_size, -1)
        self.hidden = self.init_hidden()
        outp = torch.cat([max_pool, avg_pool], dim=1)
        y = self.dropout(self.relu(outp))
        y = self.hidden2label(y)
        return y
