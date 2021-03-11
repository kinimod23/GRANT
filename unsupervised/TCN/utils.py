import os
import torch
from torch.autograd import Variable
import pickle
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.metrics import log_loss
import pandas as pd

def data_generator(args, vocabulary):
    if os.path.exists(args.data + "/corpus") and not args.corpus:
        corpus = pickle.load(open(args.data + '/corpus', 'rb'))
    else:
        corpus = Corpus(args.data, vocabulary)
        pickle.dump(corpus, open(args.data + '/corpus', 'wb'))
    return corpus


def generate_vocabulary(data, vocabulary_size):
    with open(data + '/train.txt', 'r') as f:
        words = f.read().split()
        counter = Counter(words)
        most_common = counter.most_common(vocabulary_size)
        vocabulary = set([word for word, count in most_common])
        print(vocabulary)
        return vocabulary


class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<unk>':0}
        self.idx2word = ['<unk>']

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, vocabulary):
        self.dictionary = Dictionary()
        self.vocabulary = vocabulary
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'valid.txt'))



    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split()
                tokens += len(words)
                for word in words:
                    if word in self.vocabulary:
                        self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split()
                for word in words:
                    if word in self.vocabulary:
                        ids[token] = self.dictionary.word2idx[word]
                    else:
                        ids[token] = self.dictionary.word2idx['<unk>']
                    token += 1
        return ids

class CorpusTest(object):
    def __init__(self, text, vocabulary, dictionary):
        self.dictionary = dictionary
        self.vocabulary = vocabulary
        self.test = self.tokenize(text)



    def tokenize(self, text):
        sents = sent_tokenize(text)
        tokens = 0
        for sent in sents:
            """Tokenizes a text file."""
            words = word_tokenize(sent)
            tokens += len(words)
        ids = torch.LongTensor(tokens)
        token = 0
        for sent in sents:
            words = word_tokenize(sent)
            for word in words:
                if word in self.vocabulary and word in self.dictionary.word2idx:
                    ids[token] = self.dictionary.word2idx[word]
                else:
                    ids[token] = self.dictionary.word2idx['<unk>']
                token += 1
        return ids


def batchify(data, batch_size, args):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1)
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.seq_len, source.size(1) - 1 - i)
    data = Variable(source[:, i:i+seq_len])
    target = Variable(source[:, i+1:i+1+seq_len])     # CAUTION: This is un-flattened!
    return data, target


def beam_search_decoder(data, target, k):
    score = 1.0
    for i, row in enumerate(data):
        score = score * float(row[target[i]])
    print('target score: ', score)
    sequences = [[list(), 1.0]]
    for row in data:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * float(row[j])]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse=True)
        sequences = ordered[:k]
        print(sequences)
        print(len(sequences))
    return sequences


def calculate_average(data, target):
    score = 0
    length = int(data.size()[0])
    for i, row in enumerate(data):
        score = score + float(row[target[i]])
    return score / length


def return_scores(data, target, dictionary):
    scores = []
    for i, row in enumerate(data):
        y_true = np.zeros(row.size(0))
        y_true[int(target[i])] = 1.0
        nextWordLogLoss = log_loss(y_true, row.detach().cpu().numpy())
        word = dictionary.idx2word[int(target[i])]
        score = nextWordLogLoss
        scores.append((word, score))
    return scores


def ecdf(data):
    raw_data = np.array(data)
    # create a sorted series of unique data
    cdfx = np.sort(data.unique())
    # x-data for the ECDF: evenly spaced sequence of the uniques
    x_values = np.linspace(start=min(cdfx), stop=max(cdfx), num=len(cdfx))

    # size of the x_values
    size_data = raw_data.size
    # y-data for the ECDF:
    y_values = []
    for i in x_values:
        # all the values in raw data less than the ith value in x_values
        temp = raw_data[raw_data <= i]
        # fraction of that value with respect to the size of the x_values
        value = temp.size / size_data
        # pushing the value in the y_values
        y_values.append(value)
    # return both x and y values
    return x_values, y_values


def count_tokens(input_file):
    with open(input_file, encoding='utf') as f:
        word_counter = 0
        for line in f.readlines():
            word_counter += len(line.split())

        print("Num tokens: ", word_counter)


def corpus_to_sents(input_file, output_file):
    data = pd.read_csv(input_file, encoding="utf-8", delimiter='\t')
    output_file_train = output_file + '_train.txt'
    output_file_test = output_file + '_test.txt'
    if os.path.exists(output_file_train):
        os.remove(output_file_train)
    if os.path.exists(output_file_test):
        os.remove(output_file_test)

    output_train = open(output_file_train, 'a', encoding="utf8")
    output_test = open(output_file_test, 'a', encoding="utf8")

    test_idx = int(data.shape[0] * 0.9)

    for i, row in data.iterrows():
        text = row['text']
        sents = sent_tokenize(text)
        for sent in sents:
            if i < test_idx:
                output_train.write(sent.strip() + "\n")
            else:
                output_test.write(sent.strip() + "\n")
    output_train.close()
    output_test.close()





