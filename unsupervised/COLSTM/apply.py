import numpy as np
import re
import argparse
import pickle
from LSTMCNN import LSTMCNN
from math import exp
from sklearn.metrics import log_loss
import pandas as pd
import nltk
from statsmodels.distributions.empirical_distribution import ECDF
import math

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


def build_dataset(input_file, num_sents=10):
    data = [['text']]
    with open(input_file, encoding='utf') as f:
        counter = 0
        text = []
        for line in f.readlines():
            if counter == num_sents:
                data.append([" ".join(text)])
                counter = 0
                text = []
            else:
                counter += 1
                text.append(line)
    header = data.pop(0)
    return pd.DataFrame(data, columns=header)



def vocab_unpack(vocab, pos_tags=False):
    if pos_tags:
        return vocab['idx2word'], vocab['word2idx'][()], vocab['idx2char'], vocab['char2idx'][()], vocab['idx2pos'], vocab['pos2idx'][()]
    else:
        return vocab['idx2word'], vocab['word2idx'][()], vocab['idx2char'], vocab['char2idx'][()]


def calculate_score(l):
    perp_l = [x[1] for x in l]
    length = len(l)

    perp_l = sorted(perp_l)
    perp_l = [math.sqrt(i + 1) * prob if l[i][0] != '|' else 2 * math.sqrt(i + 1) * prob for i,prob in enumerate(perp_l)]
    return sum(perp_l)/length


def sort_by_sent_length(l):
    return l


class Vocabulary:
    def __init__(self, tokens, vocab_file, max_word_l=65):
        self.tokens = tokens
        self.max_word_l = max_word_l
        self.prog = re.compile('\s+')

        print('loading vocabulary file...')

        vocab_mapping = np.load(vocab_file)
        self.idx2word, self.word2idx, self.idx2char, self.char2idx = vocab_unpack(vocab_mapping)
        self.vocab_size = len(self.idx2word)

        print('Word vocab size: %d, Char vocab size: %d' % (len(self.idx2word), len(self.idx2char)))

        self.word_vocab_size = len(self.idx2word)
        self.char_vocab_size = len(self.idx2char)

    def index(self, word):
        if len(word.strip()) == 0:
            word = self.tokens.UNK
        if word[0] == self.tokens.UNK and len(word) > 1: # unk token with character info available
            word = word[2:]
            w = self.word2idx[self.tokens.UNK]
        else:
            w = self.word2idx[word] if word in self.word2idx else self.word2idx[self.tokens.UNK]

        c = np.zeros(self.max_word_l, dtype='int32')
        chars = [self.char2idx[self.tokens.START]] # start-of-word symbol
        chars += [self.char2idx[char] for char in word if char in self.char2idx]
        chars.append(self.char2idx[self.tokens.END]) # end-of-word symbol
        if len(chars) >= self.max_word_l:
            chars[self.max_word_l-1] = self.char2idx[self.tokens.END]
            c = chars[:self.max_word_l]
        else:
            c[:len(chars)] = chars
        return w, c


    def get_input(self, line):
        output_words = []
        output_chars = []
        line = line.replace('<unk>', self.tokens.UNK)  # replace unk with a single character
        line = line.replace(self.tokens.START, '')  # start-of-word token is reserved
        line = line.replace(self.tokens.END, '')  # end-of-word token is reserved
        words = self.prog.split(line)

        for rword in filter(None, words):
            w, c = self.index(rword)
            output_words.append(w)
            output_chars.append(c)
        if self.tokens.EOS != '':   # PTB does not have <eos> so we add a character for <eos> tokens
            w, c = self.index(self.tokens.EOS)   # other datasets don't need this
            output_words.append(w)
            output_chars.append(c)
        words = np.array(output_words[-1:] + output_words[:-1], dtype='int32')

        chars = np.array(output_chars[-1:] + output_chars[:-1], dtype='int32')[:, np.newaxis, :]
        output = np.array(output_words, dtype='int32')[:, np.newaxis, np.newaxis]
        #print('words: ',words, 'y: ', output)
        eval_output = ({'word':words, 'chars':chars}, output)

        words = np.array(output_words[:], dtype='int32')
        chars = np.array(output_chars[:], dtype='int32')[:, np.newaxis, :]
        predict_output = ({'word': words, 'chars': chars}, [])

        return eval_output, predict_output



class evaluator:
    def __init__(self, name, vocabulary, settings):
        self.opt = pickle.load(open(settings, "rb"))
        self.opt.batch_size = 1
        self.opt.seq_length = 1
        self.reader = Vocabulary(self.opt.tokens, vocabulary, max_word_l=self.opt.max_word_l)
        self.model = LSTMCNN(self.opt)
        self.model.load_weights(name)

    def logprob(self, line):
        eval, predict = self.reader.get_input(line)
        x_eval, y_eval = eval
        x_predict, y_predict = predict
        nwords = len(y_eval)
        return self.model.evaluate(x_eval, y_eval, batch_size=1, verbose=2), nwords, x_predict


def main(input_file, output, model_name, vocabulary, settings, calc_rank=False):
    results = []
    print('Generating predictions: ', output)
    ev = evaluator(model_name, vocabulary, settings)
    vocab_mapping = np.load(vocabulary)
    idx2word, word2idx = vocab_unpack(vocab_mapping)[0:2]
    opt = pickle.load(open(settings, "rb"))
    opt.batch_size = 1
    opt.seq_length = 1
    model = LSTMCNN(opt)
    model.load_weights(model_name)
    data_iterator = pd.read_csv(input_file, encoding="utf-8", delimiter='\t', chunksize=1000)
    df_data = pd.DataFrame()
    for sub_data in data_iterator:
        df_data = pd.concat([df_data, sub_data], axis=0)

    print("Data shape before preprocessing:", df_data.shape)

    if calc_rank:
        test_data = list(pd.read_csv('valid.csv', encoding="utf-8", delimiter='\t')['entropy'].values)

    document_counter = 0

    for index, row in df_data.iterrows():
        document_counter += 1

        document_name = str(document_counter) + '_' + str(row['readability'])
        document_class = row['readability']
        document = row['text'].lower()

        model.reset_states()
        file_perplexities = []
        file_word_perplexities = []

        lp = 0
        nw = 0
        sum_all = 0
        sent_counter = 0
        lp_perp = 0
        nw_perp = 0

        for sent in nltk.sent_tokenize(document):

            lprob, nwords, words = ev.logprob(sent)
            sent_counter += 1
            sent_perp = []
            lp_perp += lprob * nwords
            nw_perp += nwords
            perplexity = exp(lp_perp / nw_perp)
            sum_all += perplexity

            for i in range(words['word'].shape[0]):
                char_dimension = words['chars'].shape[2]
                y = np.array([words['word'][i]], dtype='int32')[:, np.newaxis, np.newaxis]


                predictions = model.predict({'word': np.array([words['word'][i - 1]]),
                                            'chars': words['chars'][i - 1].reshape(1, 1, char_dimension)}, batch_size=1, verbose=0)
                y_predicted = predictions.flatten()

                model.reset_states()

                #calculate score
                y_true = np.zeros(y_predicted.shape[0])
                y_true[y[0][0]] = 1.0
                nextWordLogLoss = log_loss(y_true, y_predicted)

                indices_max = np.argpartition(y_predicted, -1)[-1:]
                all_max = []
                for idx in indices_max:
                    y_max = np.zeros(y_predicted.shape[0])
                    y_max[idx] = 1.0
                    maxWordLogLoss = log_loss(y_max, y_predicted)
                    all_max.append(maxWordLogLoss)

                nw += 1
                lp += nextWordLogLoss
                sent_perp.append((idx2word[words['word'][i]], nextWordLogLoss, all_max))

            file_word_perplexities.append([calculate_score(sent_perp)," ".join([word + '/' + str(round(perp,1)) for word, perp, max in sent_perp])])
            file_perplexities.append([calculate_score(sent_perp), sent.strip()])

        score = sum([perp for perp, sent in file_perplexities]) / len(file_perplexities)
        final_perplexity = sum_all / sent_counter

        if not calc_rank:
            results.append([document, score, final_perplexity, document_class])
        else:
            ranked_data = np.array(test_data + [score])
            ecdf = ECDF(ranked_data)
            idx = list(ecdf.x).index(score)
            score = ecdf.y[idx]
            results.append([document, score, document_class])

        print(document_name + " score: " + str(score) + " perplexity: " + str(final_perplexity))

    results = pd.DataFrame(results, columns=['text', 'score', 'perplexity', 'class'])
    results.to_csv(output, encoding="utf8", index=False, sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="unsupervised/LSTMCNN/cv/simple-wiki-50000.h5", help="path to trained model in .h5 format")
    parser.add_argument('--settings', type=str, default="unsupervised/LSTMCNN/cv/simple-wiki-50000.pkl", help="path to model's setting file in .pkl format")
    parser.add_argument('--vocabulary', type=str, default="data/simple-wiki/vocab.npz", help="path to vocab file")
    parser.add_argument('--data', type=str, default="data/WeeBit/weebit_reextracted.csv", help="Path to readability corpus")
    parser.add_argument('--output', type=str, default="unsupervised/LSTMCNN/results/simple_weebit.csv", help="path to output results file")
    args = parser.parse_args()

    main(args.data, args.output, args.model, args.vocabulary, args.settings, calc_rank=False)




