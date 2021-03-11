import torch
from transformers import BertTokenizer, BertForMaskedLM
import pandas as pd
import nltk
import torch.nn as nn
import numpy as np
from sklearn.metrics import  log_loss
import math
import argparse


def calculate_score(sent_scores):
    perp_l = sorted(sent_scores, key=lambda x: x[0])
    words = [word for score, word in perp_l]
    scores = [score for score, word in perp_l]

    unk_weight = 2
    perp_l = [math.sqrt(i + 1) * prob if not words[i].startswith('##') else math.sqrt(i + 1) * prob * unk_weight for i, prob in enumerate(scores)]
    return sum(perp_l)/len(words)



def calculate_perplexity(input_file, results_path, lang):
    # Load pre-trained model tokenizer (vocabulary)
    if lang == 'en':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if lang == 'de':
        tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')        
    else:
        tokenizer = BertTokenizer.from_pretrained('EMBEDDIA/crosloengual-bert')
    softmax = nn.Softmax(dim=-1)
    criterion = nn.CrossEntropyLoss()

    # Load pre-trained model (weights)
    if lang == 'en':
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    if lang == 'de':
        model = BertForMaskedLM.from_pretrained('bert-base-german-cased')        
    else:
        model = BertForMaskedLM.from_pretrained('EMBEDDIA/crosloengual-bert')
    model.cuda()
    model.eval()

    df_data = pd.read_csv(input_file, encoding="utf-8", delimiter='\t')

    print("Data shape before preprocessing:", df_data.shape)
    document_counter = 0
    results = [['text', 'class', 'score', 'perplexity']]


    for index, row in df_data.iterrows():
        document_counter += 1
        document_name = str(document_counter) + '_' + str(row['readability'])
        document_class = row['readability']
        document = row['text']
        all_scores = []
        total_loss = 0
        sequence_length = 0
        for sent in nltk.sent_tokenize(document):
            tokenized_text = tokenizer.tokenize(sent)
            length = len(tokenized_text)
            if length > 200:
                tokenized_text = tokenized_text[:200]
                length = len(tokenized_text)
            sequence_length += length
            sent_scores = []
            for i in range(length):
                masked_index = i
                tokenized_text = tokenizer.tokenize(sent)
                if len(tokenized_text) > 200:
                    tokenized_text = tokenized_text[:200]
                true_word = tokenized_text[i]
                true_idx = tokenizer.convert_tokens_to_ids(tokenized_text)[masked_index]
                tokenized_text[masked_index] = '[MASK]'

                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                segments_ids = [0] * length

                tokens_tensor = torch.tensor([indexed_tokens]).cuda()
                segments_tensors = torch.tensor([segments_ids]).cuda()

                predictions = model(tokens_tensor, segments_tensors)[0]
                loss = criterion(predictions[0, masked_index].unsqueeze(0), torch.tensor([true_idx]).cuda())
                total_loss += loss.item()


                predictions = softmax(predictions[0, masked_index])

                y_true = np.zeros(predictions.size(0))
                y_true[true_idx] = 1.0
                nextWordLogLoss = log_loss(y_true, predictions.detach().cpu().numpy())
                sent_scores.append((nextWordLogLoss, true_word))
            score = calculate_score(sent_scores)
            all_scores.append(score)

        final_score = sum(all_scores)/len(all_scores)

        test_loss = total_loss / sequence_length
        perplexity = math.exp(test_loss)

        results.append([document, document_class, final_score, perplexity])
        print('=' * 89)
        print('| Document name {} of class {} | test score {:8.8f} | length {:8.2f} | perplexity {:8.8f}'.format(document_name, document_class, final_score, len(document.split()), perplexity))
        print('=' * 89)
    headers = results.pop(0)
    df_results = pd.DataFrame(results, columns=headers)
    df_results.to_csv(results_path, encoding='utf8', sep='\t')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Calculate perplexity and RSRS scores of the BERT model')
    argparser.add_argument('--input', type=str,
                           default='data/WeeBit/weebit_reextracted.csv',
                           help='Choose dataset input csv file produced by the language model')
    argparser.add_argument('--output', type=str,
                           default="unsupervised/BERT/results/results_weebit.csv",
                           help='Choose path of output file with results')
    argparser.add_argument('--lang', type=str,
                           default="en",
                           help="Choose language of the input dataset, can be either 'en' for English or 'slo' for Slovenian")
    args = argparser.parse_args()
    calculate_perplexity(args.input, results_path=args.output, lang=args.lang)
