
# Modified from https://github.com/karpathy/char-rnn
# This version is for cases where one has already segmented train/val/test splits
import codecs
import numpy as np
from os import path
import gc
import re
from collections import Counter, OrderedDict, namedtuple
import pickle

encoding='utf8'
# encoding='iso-8859-1'

Tokens = namedtuple('Tokens', ['EOS', 'UNK', 'START', 'END', 'ZEROPAD'])

def vocab_unpack(vocab, pos_tags=False):
    if pos_tags:
        return vocab['idx2word'], vocab['word2idx'][()], vocab['idx2char'], vocab['char2idx'][()], vocab['idx2pos'], vocab['pos2idx'][()]
    else:
        return vocab['idx2word'], vocab['word2idx'][()], vocab['idx2char'], vocab['char2idx'][()]

class BatchLoaderUnk:
    def __init__(self, tokens, data_dir, batch_size, seq_length, max_word_l, n_words, n_chars, pos_tags, settings, retrain=False):
        self.n_words = n_words
        self.n_chars = n_chars
        self.pos_tags = pos_tags
        self.settings =  path.join(data_dir, settings)
        train_file = path.join(data_dir, 'train.txt')
        valid_file = path.join(data_dir, 'valid.txt')
        input_files = [train_file, valid_file]
        print(input_files)
        vocab_file = path.join(data_dir, 'vocab.npz')
        tensor_file = path.join(data_dir, 'data')
        char_file = path.join(data_dir, 'data_char')
        if self.pos_tags:
            pos_file = path.join(data_dir, 'data_pos')

        # construct a tensor with all the data
        if not (path.exists(vocab_file) or path.exists(tensor_file) or path.exists(char_file)) or retrain:
            print('one-time setup: preprocessing input train/valid/test files in dir: ', data_dir)
            if self.pos_tags:
                self.text_to_tensor(tokens, input_files, vocab_file, tensor_file, char_file, max_word_l, pos_file, retrain)
            else:
                self.text_to_tensor(tokens, input_files, vocab_file, tensor_file, char_file, max_word_l, out_posfile="_", retrain=retrain)


        print('loading data files...')
        all_data = []
        all_data_char = []
        all_data_pos = []
        for split in range(2):
            all_data.append(np.load("{}_{}.npy".format(tensor_file, split), allow_pickle=True))  # train, valid, test tensors
            all_data_char.append(np.load("{}_{}.npy".format(char_file, split), allow_pickle=True))  # train, valid, test character indices
            if self.pos_tags:
                all_data_pos.append(np.load("{}_{}.npy".format(pos_file, split), allow_pickle=True))  # train, valid, test pos indices
        vocab_mapping = np.load(vocab_file, allow_pickle=True)
        if self.pos_tags:
            self.idx2word, self.word2idx, self.idx2char, self.char2idx, self.idx2pos, self.pos2idx = vocab_unpack(vocab_mapping, self.pos_tags)
        else:
            self.idx2word, self.word2idx, self.idx2char, self.char2idx = vocab_unpack(vocab_mapping)
        self.vocab_size = len(self.idx2word)
        if self.pos_tags:
            print('Word vocab size: %d, Char vocab size: %d, POS vocab size: %d' % (len(self.idx2word), len(self.idx2char), len(self.idx2pos)))
        else:
            print('Word vocab size: %d, Char vocab size: %d' % (len(self.idx2word), len(self.idx2char)))
        # create word-char mappings
        self.max_word_l = all_data_char[0].shape[1]
        # cut off the end for train/valid sets so that it divides evenly
        # test set is not cut off
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.data_sizes = []
        self.split_sizes = []
        self.all_batches = []
        print('reshaping tensors...')
        for split, data in enumerate(all_data):
            data_len = data.shape[0]
            self.data_sizes.append(data_len)
            if split < 2 and data_len % (batch_size * seq_length) != 0:
                data = data[:batch_size * seq_length * (data_len // (batch_size * seq_length))]
            ydata = data.copy()
            ydata[0:-1] = data[1:]
            ydata[-1] = data[0]
            data_char = all_data_char[split][:len(data)]
            if self.pos_tags:
                data_pos = all_data_pos[split][:len(data)]
            if split < 2:
                rdata = data.reshape((batch_size, -1))
                rydata = ydata.reshape((batch_size, -1))
                if self.pos_tags:
                    rdata_pos = data_pos.reshape((batch_size, -1))
                rdata_char = data_char.reshape((batch_size, -1, self.max_word_l))
            else: # for test we repeat dimensions to batch size (easier but inefficient evaluation)
                nseq = (data_len + (seq_length - 1)) // seq_length
                rdata = data.copy()
                rdata.resize((1, nseq*seq_length))
                rdata = np.tile(rdata, (batch_size, 1))
                rydata = ydata.copy()
                rydata.resize((1, nseq*seq_length))
                rydata = np.tile(rydata, (batch_size, 1))
                if self.pos_tags:
                    rdata_pos = data_pos.copy()
                    rdata_pos.resize((1, nseq * seq_length))
                    rdata_pos = np.tile(rdata_pos, (batch_size, 1))
                rdata_char = data_char.copy()
                rdata_char.resize((1, nseq*seq_length, rdata_char.shape[1]))
                rdata_char = np.tile(rdata_char, (batch_size, 1, 1))
            # split in batches
            x_batches = np.split(rdata, rdata.shape[1]/seq_length, axis=1)
            y_batches = np.split(rydata, rydata.shape[1]/seq_length, axis=1)
            x_char_batches = np.split(rdata_char, rdata_char.shape[1]/seq_length, axis=1)
            if self.pos_tags:
                x_pos_batches = np.split(rdata_pos, rdata_pos.shape[1] / seq_length, axis=1)
            nbatches = len(x_batches)
            self.split_sizes.append(nbatches)
            assert len(x_batches) == len(y_batches)
            assert len(x_batches) == len(x_char_batches)
            if self.pos_tags:
                assert len(x_batches) == len(x_pos_batches)
                self.all_batches.append((x_batches, y_batches, x_char_batches, x_pos_batches))
            else:
                self.all_batches.append((x_batches, y_batches, x_char_batches))
            gc.collect()


        self.batch_idx = [0, 0, 0]
        self.word_vocab_size = len(self.idx2word)
        print('data load done. Number of batches in train: %d, val: %d' \
              % (self.split_sizes[0], self.split_sizes[1]))
        gc.collect()

    def reset_batch_pointer(self, split_idx, batch_idx=0):
        self.batch_idx[split_idx] = batch_idx

    def next_batch(self, split_idx):
        while True:
            # split_idx is integer: 0 = train, 1 = val, 2 = test
            self.batch_idx[split_idx] += 1
            if self.batch_idx[split_idx] >= self.split_sizes[split_idx]:
                self.batch_idx[split_idx] = 0 # cycle around to beginning

            # pull out the correct next batch
            idx = self.batch_idx[split_idx]
            word = self.all_batches[split_idx][0][idx]
            sparse_ydata = self.all_batches[split_idx][1][idx]
            chars = self.all_batches[split_idx][2][idx]
            if self.pos_tags:
                pos = self.all_batches[split_idx][3][idx]

            # expand dims for sparse_cross_entropy optimization
            ydata = np.expand_dims(sparse_ydata, axis=2)

            if self.pos_tags:
                yield ({'word':word, 'chars':chars, 'pos':pos}, ydata)
            else:
                yield ({'word': word, 'chars': chars}, ydata)

    def text_to_tensor(self, tokens, input_files, out_vocabfile, out_tensorfile, out_charfile, max_word_l, out_posfile, retrain=False):
        split_counts = []
        if not retrain:
            print('Processing text into tensors...')
            max_word_l_tmp = 0 # max word length of the corpus
            idx2word = [tokens.UNK] # unknown word token
            word2idx = OrderedDict()
            word2idx[tokens.UNK] = 0
            idx2char = [tokens.ZEROPAD, tokens.START, tokens.END, tokens.UNK] # zero-pad, start-of-word, end-of-word tokens
            char2idx = OrderedDict()
            char2idx[tokens.ZEROPAD] = 0
            char2idx[tokens.START] = 1
            char2idx[tokens.END] = 2
            char2idx[tokens.UNK] = 3

            if self.pos_tags:
                idx2pos = [tokens.UNK]  # unknown word token
                pos2idx = OrderedDict()
                pos2idx[tokens.UNK] = 0
        else:
            print('retraining trained model')
            options = pickle.load(open('{}.pkl'.format(self.settings), "rb"))
            print('loading vocabulary file...')
            vocab_file = path.join(options.data_dir, 'vocab.npz')
            vocab_mapping = np.load(vocab_file)
            if self.pos_tags:
                idx2word, word2idx, idx2char, char2idx, idx2pos, pos2idx = vocab_unpack(vocab_mapping, self.pos_tags)
            else:
                idx2word, word2idx, idx2char, char2idx = vocab_unpack(vocab_mapping)

            if self.pos_tags:
                print('Word vocab size: %d, Char vocab size: %d, POS vocab size: %d' % (
                len(idx2word), len(idx2char), len(idx2pos)))
            else:
                print('Word vocab size: %d, Char vocab size: %d' % (len(idx2word), len(idx2char)))

        # first go through train/valid/test to get max word length
        # if actual max word length is smaller than specified
        # we use that instead. this is inefficient, but only a one-off thing so should be fine
        # also counts the number of tokens
        prog = re.compile('\s+')
        wordcount = Counter()
        charcount = Counter()
        if self.pos_tags:
            poscount = Counter()
        for	split in range(2): # split = 0 (train), 1 (val), or 2 (test)
            def update(word_pos):
                if self.pos_tags:
                    try:
                        word = word_pos.rsplit('/', 1)[0]
                        pos_tag = word_pos.rsplit('/', 1)[1][0]
                        poscount.update(pos_tag)
                    except:
                        word = word_pos
                        pos_tag = tokens.UNK
                        poscount.update(pos_tag)
                else:
                    word = word_pos

                if len(word.strip()) == 0:
                    word = tokens.UNK
                if word[0] == tokens.UNK:
                    if len(word) > 1: # unk token with character info available
                        word = word[2:]
                else:
                    wordcount.update([word])
                word = word.replace(tokens.UNK, '')
                charcount.update(word)

            f = codecs.open(input_files[split], 'r', encoding)
            counts = 0
            for line in f:
                line = line.replace('<unk>', tokens.UNK)  # replace unk with a single character
                line = line.replace(tokens.START, '')  # start-of-word token is reserved
                line = line.replace(tokens.END, '')  # end-of-word token is reserved
                words = prog.split(line)

                for word in filter(None, words):
                    update(word)
                    if self.pos_tags:
                        try:
                            word = word.rsplit('/', 1)[0]
                        except:
                            pass
                    if not retrain:
                        max_word_l_tmp = max(max_word_l_tmp, len(word) + 2) # add 2 for start/end chars
                    counts += 1
                if tokens.EOS != '':
                    update(tokens.EOS)
                    counts += 1 # PTB uses \n for <eos>, so need to add one more token at the end

            f.close()
            split_counts.append(counts)

        if not retrain:

            print('Most frequent words:', len(wordcount))
            for ii, ww in enumerate(wordcount.most_common(self.n_words - 1)):
                word = ww[0]
                word2idx[word] = ii + 1
                idx2word.append(word)
                if ii < 5: print(word)

            print('Most frequent chars:', len(charcount))
            for ii, cc in enumerate(charcount.most_common(self.n_chars - 4)):
                char = cc[0]
                char2idx[char] = ii + 4
                idx2char.append(char)
                if ii < 5: print(char)

            if self.pos_tags:
                print('Most frequent POS tags:', len(poscount))
                for ii, pp in enumerate(poscount.most_common(self.n_chars - 1)):
                    pos = pp[0]
                    pos2idx[pos] = ii + 1
                    idx2pos.append(pos)
                    if ii < 5: print(pos)

                print('POS counts:')
                for ii, cc in enumerate(poscount.most_common()):
                    print(ii, cc[0].encode(encoding), cc[1])

            print('Char counts:')
            for ii, cc in enumerate(charcount.most_common()):
                print(ii, cc[0].encode(encoding), cc[1])

            print('After first pass of data, max word length is: ', max_word_l_tmp)
            print('Token count: train %d, val %d' % (split_counts[0], split_counts[1]))

        # if actual max word length is less than the limit, use that
        if retrain:
            max_word_l = options.max_word_l
        else:
            max_word_l = min(max_word_l_tmp, max_word_l)

        for split in range(2):  # split = 0 (train), 1 (val), or 2 (test)
            # Preallocate the tensors we will need.
            # Watch out the second one needs a lot of RAM.
            output_tensor = np.empty(split_counts[split], dtype='int32')
            output_chars = np.zeros((split_counts[split], max_word_l), dtype='int32')
            if self.pos_tags:
                output_pos = np.empty(split_counts[split], dtype='int32')

            def append(word_pos, word_num):
                if self.pos_tags:
                    try:
                        word = word_pos.rsplit('/', 1)[0]
                        pos_tag = word_pos.rsplit('/', 1)[1][0]

                    except:
                        word = word_pos
                        pos_tag = tokens.UNK
                else:
                    word = word_pos
                chars = [char2idx[tokens.START]] # start-of-word symbol
                if len(word.strip()) == 0:
                    word = tokens.UNK
                if word[0] == tokens.UNK and len(word) > 1: # unk token with character info available
                    word = word[2:]
                    output_tensor[word_num] = word2idx[tokens.UNK]
                else:
                    output_tensor[word_num] = word2idx[word] if word in word2idx else word2idx[tokens.UNK]
                if self.pos_tags:
                    output_pos[word_num] = pos2idx[pos_tag] if pos_tag in pos2idx else pos2idx[tokens.UNK]
                chars += [char2idx[char] for char in word if char in char2idx]
                chars.append(char2idx[tokens.END]) # end-of-word symbol
                if len(chars) >= max_word_l:
                    chars[max_word_l-1] = char2idx[tokens.END]
                    output_chars[word_num] = chars[:max_word_l]
                else:
                    output_chars[word_num, :len(chars)] = chars
                return word_num + 1

            f = codecs.open(input_files[split], 'r', encoding)
            word_num = 0
            for line in f:
                line = line.replace('<unk>', tokens.UNK)  # replace unk with a single character
                line = line.replace(tokens.START, '')  # start-of-word token is reserved
                line = line.replace(tokens.END, '')  # end-of-word token is reserved
                words = prog.split(line)
                for rword in filter(None, words):
                    word_num = append(rword, word_num)
                if tokens.EOS != '':   # PTB does not have <eos> so we add a character for <eos> tokens
                    word_num = append(tokens.EOS, word_num)   # other datasets don't need this
            f.close()
            tensorfile_split = "{}_{}.npy".format(out_tensorfile, split)
            print('saving ', tensorfile_split)
            np.save(tensorfile_split, output_tensor)
            charfile_split = "{}_{}.npy".format(out_charfile, split)
            print('saving ', charfile_split)
            np.save(charfile_split, output_chars)
            if self.pos_tags:
                posfile_split = "{}_{}.npy".format(out_posfile, split)
                print('saving ', posfile_split)
                np.save(posfile_split, output_pos)


        # save output preprocessed files
        print ('saving ', out_vocabfile)
        if self.pos_tags:
            np.savez(out_vocabfile, idx2word=idx2word, word2idx=word2idx, idx2char=idx2char, char2idx=char2idx, idx2pos=idx2pos, pos2idx=pos2idx)
        else:
            np.savez(out_vocabfile, idx2word=idx2word, word2idx=word2idx, idx2char=idx2char, char2idx=char2idx)
