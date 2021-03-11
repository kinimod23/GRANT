import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_max_lengths, get_evaluation
from src.dataset import MyDataset
from src.hierarchical_att_model import HierAttNet
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, precision_score, recall_score, f1_score
import argparse
import numpy as np
import pandas as pd
from collections import Counter


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--task_name", type=str, default="weebit", help="The name of the task to train, should be 'onestopenglish', 'weebit', 'newsela' or 'ucbeniki'")
    parser.add_argument("--input_path", type=str, default="data/WeeBit/weebit_reextracted.csv")
    parser.add_argument("--input_path_test", type=str, default="data/WeeBit/weebit_reextracted_test.csv")   
    parser.add_argument("--input_path_valid", type=str, default="data/WeeBit/weebit_reextracted_valid.csv")   
    parser.add_argument("--vocab_path", type=str, default="supervised/HAN/vocab/weebit_vocab.pk")
    parser.add_argument("--saved_path", type=str, default="supervised/HAN/trained_models")

    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epoches", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=200)
    parser.add_argument("--sent_hidden_size", type=int, default=100)
    parser.add_argument("--es_min_delta", type=float, default=0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=20,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")

    args = parser.parse_args()
    return args


def train(opt, train_data_path, test_data_path, valid_data_path):
    task = opt.task_name
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2019)
    else:
        torch.manual_seed(2019)
    output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": False,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}

    max_word_length, max_sent_length = get_max_lengths(opt.input_path)
    print("Max words: ", max_word_length, "Max sents: ", max_sent_length)

    #df_data = pd.read_csv(data_path, encoding='utf8', sep='\t')
    #df_data = df_data.sample(frac=1, random_state=2019)
    #print(df_data.shape)
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


    #if os.path.exists(opt.vocab_path):
     #   os.remove(opt.vocab_path)

    df_train = pd.read_csv(train_data_path, encoding='utf8', sep='\t')
    df_test = pd.read_csv(test_data_path, encoding='utf8', sep='\t')
    df_valid= pd.read_csv(valid_data_path, encoding='utf8', sep='\t')



    training_set = MyDataset(df_train, opt.vocab_path, task, max_sent_length, max_word_length)
    training_generator = DataLoader(training_set, **training_params)
    
    test_set = MyDataset(df_test, opt.vocab_path, task, max_sent_length, max_word_length)
    test_generator = DataLoader(test_set, **test_params)

    valid_set = MyDataset(df_valid, opt.vocab_path, task, max_sent_length, max_word_length)
    valid_generator = DataLoader(valid_set, **test_params)

    model = HierAttNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size, training_set.num_classes, opt.vocab_path, max_sent_length, max_word_length)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    best_loss = 1e5
    best_epoch = 0
    num_iter_per_epoch = len(training_generator)

    for epoch in range(opt.num_epoches):
        model.train()
        for iter, (feature, label) in enumerate(training_generator):
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            model._init_hidden_state()
            predictions = model(feature)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()

            #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))

        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature, te_label in valid_generator:
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_feature = te_feature.cuda()
                    te_label = te_label.cuda()
                with torch.no_grad():
                    model._init_hidden_state(num_sample)
                    te_predictions = model(te_feature)
                te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.append(te_predictions.clone().cpu())
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix", "qwk"])


            print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"]))
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                print('Saving model')
                torch.save(model, opt.saved_path + os.sep + "whole_model_han")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, best_loss))
                break

    print()
    print('Evaluation: ')
    print()

    model.eval()
    model = torch.load(opt.saved_path + os.sep + "whole_model_han")
    loss_ls = []
    te_label_ls = []
    te_pred_ls = []
    for te_feature, te_label in test_generator:
        num_sample = len(te_label)
        if torch.cuda.is_available():
            te_feature = te_feature.cuda()
            te_label = te_label.cuda()
        with torch.no_grad():
            model._init_hidden_state(num_sample)
            te_predictions = model(te_feature)
        te_loss = criterion(te_predictions, te_label)
        loss_ls.append(te_loss * num_sample)
        te_label_ls.extend(te_label.clone().cpu())
        te_pred_ls.append(te_predictions.clone().cpu())
    te_pred = torch.cat(te_pred_ls, 0)
    te_label = np.array(te_label_ls)
    test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "precision", "recall", "f1", "confusion_matrix", 'qwk'])

    true = te_label
    preds = np.argmax(te_pred.numpy(), -1)
    predicted_all_folds.extend(preds)
    true_all_folds.extend(true)

    print("Test set accuracy: {}".format(test_metrics["accuracy"]))
    print("Test set precision: {}".format(test_metrics["precision"]))
    print("Test set recall: {}".format(test_metrics["recall"]))
    print("Test set f1: {}".format(test_metrics["f1"]))
    print("Test set cm: {}".format(test_metrics["confusion_matrix"]))
    print("Test set qwk: {}".format(test_metrics["qwk"]))

    accuracies_all_folds.append(test_metrics["accuracy"])
    precision_all_folds.append(test_metrics["precision"])
    recall_all_folds.append(test_metrics["recall"])
    f1_all_folds.append(test_metrics["f1"])
    qwk_all_folds.append(test_metrics["qwk"])
    print()

    #if task in ['newsela', 'merlin', 'capito', 'apa']:
     #   break

    print()
    print("Task: ", task)
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



if __name__ == "__main__":
    opt = get_args()
    train(opt, opt.input_path, opt.input_path_test, opt.input_path_valid)
