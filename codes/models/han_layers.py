import torch
import torch.nn as nn


class WordLevelRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        word_num_hidden = config.word_num_hidden
        words_dim = config.words_dim
        words_num = config.words_num
        embeddings = config.embeddings
        self.mode = config.mode
        if self.mode == 'rand':
            rand_embed_init = torch.Tensor(
                words_num, words_dim).uniform(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(
                rand_embed_init, freeze=False)
        elif self.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(
                embeddings, freeze=True)
        elif self.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(
                embeddings, freeze=False)
        else:
            print("Unsupported order")
            exit()
        self.word_context_weights = nn.Parameter(
            torch.rand(2 * word_num_hidden, 1))
        self.GRU = nn.GRU(words_dim, word_num_hidden, bidirectional=True)
        self.linear = nn.Linear(2 * word_num_hidden,
                                2 * word_num_hidden, bias=True)
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_word = nn.Softmax()

    def forward(self, x):
        # x expected to be of dimensions--> (num_words, batch_size)
        if self.mode == 'rand':
            x = self.embed(x)
        elif self.mode == 'static':
            x = self.static_embed(x)
        elif self.mode == 'non-static':
            x = self.non_static_embed(x)
        else:
            print("Unsupported mode")
            exit()
        h, _ = self.GRU(x)
        x = torch.tanh(self.linear(h))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_word(x.transpose(1, 0))
        x = torch.mul(h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        return x


class SentLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        sentence_num_hidden = config.sentence_num_hidden
        word_num_hidden = config.word_num_hidden
        num_labels = config.num_labels
        self.sentence_context_weights = nn.Parameter(
            torch.rand(2 * sentence_num_hidden, 1))
        self.sentence_context_weights.data.uniform_(-0.1, 0.1)
        self.sentence_gru = nn.GRU(
            2 * word_num_hidden, sentence_num_hidden, bidirectional=True)
        self.sentence_linear = nn.Linear(
            2 * sentence_num_hidden, 2 * sentence_num_hidden, bias=True)
        self.fc = nn.Linear(2 * sentence_num_hidden, num_labels)
        self.soft_sent = nn.Softmax()

    def forward(self, x):
        sentence_h, _ = self.sentence_gru(x)
        x = torch.tanh(self.sentence_linear(sentence_h))
        x = torch.matmul(x, self.sentence_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_sent(x.transpose(1, 0))
        x = torch.mul(sentence_h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        x = self.fc(x.squeeze(0))
        return x


class HAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.word_attention_rnn = WordLevelRNN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)

    def forward(self, x,  **kwargs):
        x = x.permute(1, 2, 0)  # Expected : # sentences, # words, batch size
        num_sentences = x.size(0)
        word_attentions = None
        for i in range(num_sentences):
            word_attn = self.word_attention_rnn(x[i, :, :])
            if word_attentions is None:
                word_attentions = word_attn
            else:
                word_attentions = torch.cat((word_attentions, word_attn), 0)
        return self.sentence_attention_rnn(word_attentions)
