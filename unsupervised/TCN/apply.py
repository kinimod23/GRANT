import pandas as pd
from utils import *
import math
import argparse
from statsmodels.distributions.empirical_distribution import ECDF
from torch import nn

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  

def lm_data_generator(args, vocabulary):
    if os.path.exists(args.lm_data + "/corpus") and not args.corpus:
        corpus = pickle.load(open(args.lm_data + '/corpus', 'rb'))
    else:
        corpus = Corpus(args.lm_data, vocabulary)
        pickle.dump(corpus, open(args.lm_data + '/corpus', 'wb'))
    return corpus

def calculate_sent_score(scores, sents):
    counter = 0
    score = 0
    num_empty_sents = 0
    for sent in sents:
        new_sent = []
        words = []
        for _ in sent:
            word, word_score = scores[counter]
            new_sent.append(word_score)
            words.append(word)
            counter += 1

        perp_l = sorted(new_sent)
        unk_weight = 2
        perp_l = [math.sqrt(i + 1) * prob if words[i]!= '<unk>' else math.sqrt(i + 1) * prob * unk_weight  for i, prob in enumerate(perp_l)]
        try:
            sent_score = sum(perp_l) / len(new_sent)
            score += sent_score
        except:
            print(sent)
            num_empty_sents += 1
    try:
        return score/(len(sents) - num_empty_sents)
    except:
        return None

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


def tokenize(document, validseqlength):
    sents = sent_tokenize(document)
    tokens = 0
    all_sents = []

    for sent in sents:
        """Tokenizes a text file."""
        words = word_tokenize(sent)
        for i, word in enumerate(words):
            if i == 0:
                sent_start_index = tokens
            tokens += 1
        all_sents.append((sent_start_index, words))
    filtered_sents = []
    if all_sents[-1][0] > validseqlength:
        for idx, data in enumerate(all_sents):
            start, sent = data
            if start > validseqlength:
                filtered_sents = all_sents[idx:]
                filtered_sents = [x[1] for x in filtered_sents]
                validseqlength = start - 1
                break
    else:
        if len(all_sents) > 3:
            filtered_sents = all_sents[2:]
            validseqlength = all_sents[2][0] - 1
            filtered_sents = [x[1] for x in filtered_sents]
        elif len(all_sents) > 1:
            filtered_sents = all_sents[1:]
            validseqlength = len(all_sents[0][1]) - 1
            filtered_sents = [x[1] for x in filtered_sents]
        else:
            validseqlength = 2
            filtered_sents = [all_sents[0][1][3:]]
            print('only one sentence')
    return filtered_sents, validseqlength


def calculate_perplexity(input_file, dictionary, test_set=False, results_path="results/results", calc_rank=False):
    n_words = len(dictionary)
    if not test_set:
        data_iterator = pd.read_csv(input_file, encoding="utf-8", delimiter='\t', chunksize=1000)
        df_data = pd.DataFrame()
        for sub_data in data_iterator:
            df_data = pd.concat([df_data, sub_data], axis=0)
    else:
        df_data = build_dataset(input_file)
    print("Data shape before preprocessing:", df_data.shape)
    document_counter = 0
    if not test_set:
        results = [['text', 'class', 'perplexity', 'score']]
    else:
        results = [['text', 'perplexity', 'score']]
    if calc_rank:
        test_data = list(pd.read_csv('results/valid.csv', encoding="utf-8", delimiter='\t')['score'].values)

    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=-1)

    model.eval()
    with torch.no_grad():
        for index, row in df_data.iterrows():
            document_counter += 1
            if not test_set:
                document_name = str(document_counter) + '_' + str(row['readability'])
                document_class = row['readability']
            else:
                document_name = str(document_counter)
            document = row['text'].lower()

            doc = CorpusTest(document, vocabulary, dictionary)
            eval_batch_size = 1
            t_data = batchify(doc.test, eval_batch_size, args)
            total_loss = 0
            processed_data_size = 0

            sents, eff_history = tokenize(document, args.validseqlen)
            all_scores = []
            for i in range(0, t_data.size(1) - 1, args.validseqlen):
                if i != 0:
                    eff_history = args.seq_len - args.validseqlen

                data, targets = get_batch(t_data, i, args, evaluation=True)
                output = model(data)
                final_output = output[:, eff_history:].contiguous().view(-1, n_words)
                final_target = targets[:, eff_history:].contiguous().view(-1)

                softmax_output = softmax(final_output)

                scores = return_scores(softmax_output, final_target, dictionary)
                all_scores.extend(scores)
                loss = criterion(final_output, final_target)

                if not torch.isnan(loss.data).item():
                    total_loss += (data.size(1) - eff_history) * loss.data
                    processed_data_size += data.size(1) - eff_history

            score = calculate_sent_score(all_scores, sents)
            if not score or np.isnan(score) or np.isnan(total_loss.item()):
                print('No score')
                print(sents)
                continue
            if total_loss:

                test_loss =  total_loss.item() / processed_data_size
                if not test_set:
                    perplexity = math.exp(test_loss)
                    if not calc_rank:
                        results.append([document, document_class, math.exp(test_loss), score])
                        print('=' * 89)
                        print('| Document name {} of class {} | test loss {:5.2f} | test score {:8.8f} | perplexity {:8.2f} | length {:8.2f}'.format(document_name, document_class, test_loss, score, perplexity, len(document.split())))
                        print('=' * 89)
                    else:
                        ranked_data = np.array(test_data + [score])
                        ecdf = ECDF(ranked_data)
                        idx = list(ecdf.x).index(score)
                        score = ecdf.y[idx]
                        results.append([document, document_class, math.exp(test_loss), score])
                        print('=' * 89)
                        print('| Document name {} of class {} | test loss {:5.2f} | test score {:8.8f} | perplexity {:8.2f} | length {:8.2f}'.format(document_name, document_class, test_loss, score, perplexity, len(document.split())))
                        print('=' * 89)
                else:
                    results.append([document, math.exp(test_loss), score])
                    print('=' * 89)
                    print('| Document name {} | test loss {:5.2f} | test score {:8.8f} |perplexity {:8.2f}'.format(document_name, test_loss, score, math.exp(test_loss)))
                    print('=' * 89)
            else:
                'Document to short'

    headers = results.pop(0)
    df_results = pd.DataFrame(results, columns=headers)
    df_results.to_csv(results_path, encoding='utf8', sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling - Word-level Language Modeling')
    parser.add_argument('--data', type=str,
                        default='data/WeeBit/weebit_reextracted.csv',
                        help='location of the readability corpus')
    parser.add_argument('--lm_data', type=str,
                        default='data/wiki-simple-wiki',
                        help='location of the data corpus used for language model training')
    parser.add_argument('--lm_model_path', type=str,
                        default="unsupervised/TCN/cv/model_simple_en_50000_test_perplexity_45.34.pt",
                        help='Trained language model path')
    parser.add_argument('--results_path', type=str,
                        default='unsupervised/TCN/results/simple_weebit.csv',
                        help='Where to save results')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA (default: True)')
    parser.add_argument('--tied', action='store_false',
                        help='tie the encoder-decoder weights (default: True)')
    parser.add_argument('--optim', type=str, default='SGD',
                        help='optimizer type (default: SGD)')
    parser.add_argument('--validseqlen', type=int, default=40,
                        help='valid sequence length (default: 40)')
    parser.add_argument('--seq_len', type=int, default=80,
                        help='total sequence length, including effective history (default: 80)')
    parser.add_argument('--corpus', action='store_true',
                        help='force re-make the corpus (default: False)')
    parser.add_argument('--vocabulary_size', type=int, default=50000,
                        help='Vocabulary size')
    args = parser.parse_args()

    vocabulary = generate_vocabulary(args.lm_data, args.vocabulary_size)
    corpus = lm_data_generator(args, vocabulary)

    with open(args.lm_model_path, 'rb') as f:
        model = torch.load(f)
        if args.cuda:
            model = model.cuda()

    calculate_perplexity(args.data, corpus.dictionary, False, args.results_path, calc_rank=False)
