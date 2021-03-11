# -*- coding: utf-8 -*-

import argparse
import time
from utils import train_and_test
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


if __name__ == '__main__':
    start_time = time.time()
    argparser = argparse.ArgumentParser(description='Bilstm for readability classification')
    argparser.add_argument("--input_path", type=str,
                           default='data/WeeBit/weebit_reextracted.csv',
                           help='Choose input trainset')
    argparser.add_argument("--input_path_test",
                        default="data/WeeBit/weebit_reextracted_test.csv",
                        type=str,
                        help="Path to test input dataset.")
    argparser.add_argument("--input_path_valid",
                        default="data/WeeBit/weebit_reextracted_valid.csv",
                        type=str,
                        help="Path to validation input dataset.")                             
    argparser.add_argument("--task_name", type=str,
                           default='weebit',
                           help="The name of the task to train, should be 'onestopenglish', 'weebit', 'newsela' or 'ucbeniki")
    argparser.add_argument("--saved_path", type=str, default="supervised/pooled_BiLSTM/trained_models")
    argparser.add_argument("--num_epoch", type=int,
                           default=100,
                           help='Choose number of epochs')
    args = argparser.parse_args()
    #input = args.input_path
    output = args.task_name

    #df_data = pd.read_csv(input, encoding='utf8', sep='\t')
    #df_data = df_data.sample(frac=1, random_state=2019)
    #y = df_data.readability.values
    #kf = model_selection.StratifiedKFold(n_splits=5)

    predicted_all_folds = []
    true_all_folds = []
    counter = 0
    accuracies_all_folds = []
    precision_all_folds = []
    recall_all_folds = []
    f1_all_folds = []
    qwk_all_folds = []
    fold = 0


        
    df_train = pd.read_csv(args.input_path, encoding='utf8', sep='\t')
    df_test = pd.read_csv(args.input_path_test, encoding='utf8', sep='\t')
    df_valid = pd.read_csv(args.input_path_valid, encoding='utf8', sep='\t')




    train_texts, train_labels = df_train.text.values.tolist(), df_train.readability.values.tolist()
    train_texts = [text.lower() for text in train_texts]

    valid_texts, valid_labels = df_valid.text.values.tolist(), df_valid.readability.values.tolist()
    valid_texts = [text.lower() for text in valid_texts]

    test_texts, test_labels = df_test.text.values.tolist(), df_test.readability.values.tolist()
    test_texts = [text.lower() for text in test_texts]


    if output in ['weebit']:
        train_labels = [label - 2 for label in train_labels]
        valid_labels = [label - 2 for label in valid_labels]
        test_labels = [label - 2 for label in test_labels]
    else:
        train_labels = [label for label in train_labels]
        valid_labels = [label for label in valid_labels]
        test_labels = [label for label in test_labels]

    print('Labels: ', set(train_labels))

    total_true, total_pred, accuracy, precison, recall, f1, qwk = train_and_test(train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels, args.num_epoch, output, args.saved_path)
    true_all_folds.extend(total_true)
    predicted_all_folds.extend(total_pred)
    accuracies_all_folds.append(accuracy)
    precision_all_folds.append(precison)
    recall_all_folds.append(recall)
    f1_all_folds.append(f1)
    qwk_all_folds.append(qwk)

    #if output in ['newsela', 'merlin', 'capito', 'apa']:
     #   break

    print()
    print("Accuracy: ", accuracy_score(true_all_folds, predicted_all_folds))
    print("Precison: ", precision_score(true_all_folds, predicted_all_folds, average="weighted"))
    print("Recall: ", recall_score(true_all_folds, predicted_all_folds, average="weighted"))
    print("F1: ", f1_score(true_all_folds, predicted_all_folds, average="weighted"))
    print('Confusion matrix: ', confusion_matrix(true_all_folds, predicted_all_folds))
    print('QWK: ', cohen_kappa_score(true_all_folds, predicted_all_folds, weights="quadratic"))
    print('All folds accuracy: ', accuracies_all_folds)
    print('All folds precision: ', precision_all_folds)
    print('All folds recall: ', recall_all_folds)
    print('All folds f1: ', f1_all_folds)
    print('All folds QWK: ', qwk_all_folds)


    print("--- Model creation in minutes ---", round(((time.time() - start_time) / 60), 2))
    print("--- Training & Testing in minutes ---", round(((time.time() - start_time) / 60), 2))









