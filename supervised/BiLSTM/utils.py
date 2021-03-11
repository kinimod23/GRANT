# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from architecture import NetLstm
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from nltk import PunktSentenceTokenizer
from nltk import WordPunctTokenizer
from collections import defaultdict
import os

# Set the random seed manually for reproducibility.
torch.manual_seed(2019)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(2019)
if torch.cuda.is_available():
    print(torch.cuda.is_available())

sentTokenizer = PunktSentenceTokenizer()
wordTokenizer = WordPunctTokenizer()


def generate_vocabulary(data):
    all_data = " ".join(data)
    dd = defaultdict(int)
    words = [word.lower() for sent in sentTokenizer.tokenize(all_data) for word in wordTokenizer.tokenize(sent)]
    for word in words:
        dd[word] += 1

    print("Len vocab initial: ", len(dd))
    vocabulary = []
    for word, freq in dd.items():
        if freq >= 4:
            vocabulary.append(word)
    print("Len vocab final: ", len(vocabulary))
    return set(vocabulary)


class Dictionary(object):
    def __init__(self, words=True):
        if words:
            #unknown is one, zero is reserved for padding
            self.word2idx = {'<pad>':0,'<unk>':1}
            self.idx2word = ['<pad>', '<unk>']
        else:
            self.word2idx = {}
            self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, vocabulary, train_x, train_y, test_x, test_y, batch_size, max_length=2048, words=True, vocab_only=False):
        self.dictionary = Dictionary(words)
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.max_length = max_length
        self.words = words
        self.vocab_only = vocab_only

        self.train_x, self.train_y = self.tokenize(train_x, train_y, words, vocab_only)
        self.test_x, self.test_y = self.tokenize(test_x, test_y, words, vocab_only, dict_exist=True)

    def tokenize(self, data_x, data_y, words, vocab_only, dict_exist=False):

        # Add words to the dictionary
        all_data = []
        for i, text in enumerate(data_x):
            if words:
                tokens = [word.lower() for sent in sentTokenizer.tokenize(text) for word in wordTokenizer.tokenize(sent)]
            else:
                tokens = [c for c in text]
            if not dict_exist:
                for token in tokens:
                    if words:
                        if token in self.vocabulary:
                            self.dictionary.add_word(token)
                    else:
                        self.dictionary.add_word(token)

            if vocab_only:
                tokens = [token for token in tokens if token in self.vocabulary]
            if not dict_exist and not self.max_length:
                self.max_length = len(tokens)
            all_data.append((len(tokens), tokens, data_y[i]))

        print(len(self.dictionary.idx2word), len(all_data))
        all_data = sorted(all_data, key=lambda x: x[0])
        cut_idx = len(all_data) - (len(all_data) % self.batch_size)
        all_data = all_data[:cut_idx]

        data_x = [x[1] for x in all_data]
        data_y = [x[2] for x in all_data]

        batchified_data_x = []
        batchified_data_y = []

        for i in range(0, len(all_data), self.batch_size):
            batchified_data_x.append(data_x[i: i + self.batch_size])
            batchified_data_y.append(torch.as_tensor(np.array(data_y[i: i + self.batch_size])))

        normalized_batchified_data_x = []
        print("Max length: ", self.max_length)
        for batch in batchified_data_x:
            if self.max_length:
                max_length = min(len(batch[-1]), self.max_length)
            else:
                max_length = len(batch[-1])
            np_data = np.zeros((self.batch_size, max_length), dtype='int32')
            for i, tokens in enumerate(batch):
                for j in range(max_length):
                    try:
                       token = tokens[j]
                    except:
                        continue
                    if words:
                        if token in self.vocabulary and token in self.dictionary.word2idx:
                            np_data[i,j] = self.dictionary.word2idx[token]
                        else:
                            np_data[i,j] = self.dictionary.word2idx['<unk>']
                    else:
                        np_data[i, j] = self.dictionary.word2idx[token]
            normalized_batchified_data_x.append(torch.LongTensor(np_data))
        return normalized_batchified_data_x, batchified_data_y


def batchify(data_x, data_y, batch_size):
    doc_length = data_x.size(-1)
    nbatch = data_x.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data_x = data_x.narrow(0, 0, nbatch * batch_size)
    data_y = data_y.narrow(0, 0, nbatch * batch_size)

    # Evenly divide the data across the batch_size batches.
    data_x = data_x.view(-1, batch_size, doc_length)
    data_x = data_x
    data_y = data_y.view(-1, batch_size)
    return data_x, data_y


def get_batch(source_x, source_y, i):
    data = Variable(source_x[i]).cuda()
    target = Variable(source_y[i]).cuda()
    return data, target



def train_and_test(xtrain, ytrain, xval, yval, xtest, ytest, num_epoch, test_set_name, saved_path):
    batch_size = 8

    print("Fold shape: ", len(xtrain), len(xval))

    vocabulary = generate_vocabulary(xtrain)
    print("Vocab size: ", len(vocabulary))
    corpus = Corpus(vocabulary, xtrain, ytrain, xval, yval, batch_size)


    num_classes = len(set(ytrain))
    print("Num classes: ", num_classes)

    print("corpus generated")
    vocab_size = len(corpus.dictionary) + 1
    print("vocabulary_size: ", vocab_size)
    print('batchification done')

    _, best_loss = train(corpus.train_x, corpus.train_y, corpus.test_x, corpus.test_y, vocab_size, batch_size, num_classes, num_epoch, test_set_name, saved_path)
    corpus = Corpus(vocabulary, xtrain, ytrain, xtest, ytest, batch_size)
    test_x, test_y = corpus.tokenize(xtest, ytest, True, False, dict_exist=True)

    model = torch.load(os.path.join(saved_path, test_set_name + "_model.pt"))
    return predict(model,  test_x, test_y)



def train(data_x, data_y, test_x, test_y, vocab_size, batch_size, num_classes, num_epoch, test_set_name, saved_path):
    print('Starting training')

    model = NetLstm(vocab_size, num_classes, batch_size)
    model.cuda()


    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_loss = 999999
    best_accuracy = 0

    # Step 3. Run our forward pass.
    total_pred = []
    total_true = []
    for epoch in range(num_epoch):
        print()
        print("Epoch: ", epoch + 1)
        print()
        model.train()

        for i in range(len(data_x)):
            batch_x, batch_y = get_batch(data_x, data_y, i)
            optimizer.zero_grad()
            model.hidden = model.init_hidden()
            predicted_batch = model(batch_x.t())
            maxes = []
            for prediction in predicted_batch:
                values, idx = prediction.max(0)
                maxes.append(idx.item())

            true_y = list(batch_y.cpu().numpy())

            loss = loss_function(predicted_batch, batch_y)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_pred.extend(maxes)
            total_true.extend(true_y)
        accuracy = accuracy_score(total_pred, total_true)
        print("Train accuracy: ", accuracy)

        #testing
        total_pred = []
        total_true = []
        total_loss = []
        model.eval()
        with torch.no_grad():
            for i in range(len(test_x)):
                batch_x, batch_y = get_batch(test_x, test_y, i)

                model.hidden = model.init_hidden()
                predicted_batch = model(batch_x.t())
                loss = loss_function(predicted_batch, batch_y)
                maxes = []

                for prediction in predicted_batch:
                    values, idx = prediction.max(0)
                    maxes.append(idx.item())

                true_y = list(batch_y.cpu().numpy())
                total_loss.append(loss.item())

                total_pred.extend(maxes)
                total_true.extend(true_y)
        accuracy = accuracy_score(total_pred, total_true)
        all_loss = sum(total_loss)/len(total_loss)

        if accuracy > best_accuracy:
            with open(os.path.join(saved_path, test_set_name + "_model.pt"), 'wb') as f:
                print('Saving model')
                torch.save(model, f)
            best_loss = all_loss
            best_accuracy = accuracy
        print("Val accuracy: ", accuracy, "Best acc: ", best_accuracy)
    return model, best_loss

def predict(model, test_x, test_y):
    model.cuda()
    model.eval()

    print("Testing")
    total_pred = []
    total_true = []
    with torch.no_grad():
        for i in range(len(test_x)):
            batch_x, batch_y = get_batch(test_x, test_y, i)

            model.hidden = model.init_hidden()
            predicted_batch = model(batch_x.t())
            maxes = []
            for prediction in predicted_batch:
                values, idx = prediction.max(0)
                maxes.append(idx.item())
            total_pred.extend(maxes)

            true_y = list(batch_y.cpu().numpy())
            total_true.extend(true_y)


    accuracy = accuracy_score(total_true, total_pred)
    precison = precision_score(total_true, total_pred, average='weighted')
    recall = recall_score(total_true, total_pred, average='weighted')
    f1 = f1_score(total_true, total_pred, average='weighted')
    cm = confusion_matrix(total_true, total_pred)
    qwk = cohen_kappa_score(total_true, total_pred, weights="quadratic")
    print("Test accuracy: ", accuracy)
    print("Test precison: ", precison)
    print("Test recall: ", recall)
    print("Test f1: ", f1)
    print("Test cm: ", cm)
    print("Test qwk: ", qwk)
    print("------------------------------------------------------")
    print()
    return total_true, total_pred, accuracy, precison, recall, f1, qwk













