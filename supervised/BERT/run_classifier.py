# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

import pandas as pd
from sklearn import model_selection
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
from collections import Counter

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class ReadabilityDataset(Dataset):

  def __init__(self, docs, targets, tokenizer, max_len):
    self.docs = docs
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.docs)

  def __getitem__(self, item):
    doc = str(self.docs[item])
    target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      doc,
      add_special_tokens=True,
      truncation = True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'doc': doc,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def create_data_loader(docs, labels, tokenizer, max_len, batch_size):
  ds = ReadabilityDataset(
      docs=np.array(docs),
      targets=np.array(labels),
      tokenizer=tokenizer,
      max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0
  )

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class ClegProcessor(DataProcessor):

    def get_train_examples(self, data):
        """See base class."""
        return self._create_examples(data, "train")

    def get_dev_examples(self, data):
        """See base class."""
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        docs = []
        labels = []
        for idx, row in lines.iterrows():
            guid = "%s-%s" % (set_type, idx)
            text = row['text']
            label = int(row['readability'])
            docs.append(text)
            labels.append(label)
        return guid, docs, labels
    
        
class FalkoProcessor(DataProcessor):

    def get_train_examples(self, data):
        """See base class."""
        return self._create_examples(data, "train")

    def get_dev_examples(self, data):
        """See base class."""
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        docs = []
        labels = []
        for idx, row in lines.iterrows():
            guid = "%s-%s" % (set_type, idx)
            text = row['text']
            label = int(row['readability'])
            docs.append(text)
            labels.append(label)
        return guid, docs, labels


class CapitoProcessor(DataProcessor):

    def get_train_examples(self, data):
        """See base class."""
        return self._create_examples(data, "train")

    def get_dev_examples(self, data):
        """See base class."""
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        docs = []
        labels = []
        for idx, row in lines.iterrows():
            guid = "%s-%s" % (set_type, idx)
            text = row['text']
            label = int(row['readability'])
            docs.append(text)
            labels.append(label)
        return guid, docs, labels


class NewselaProcessor(DataProcessor):

    def get_train_examples(self, data):
        """See base class."""
        return self._create_examples(data, "train")

    def get_dev_examples(self, data):
        """See base class."""
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        docs = []
        labels = []
        for idx, row in lines.iterrows():
            guid = "%s-%s" % (set_type, idx)
            text = row['text']
            label = int(row['readability']) - 2
            docs.append(text)
            labels.append(label)
        return guid, docs, labels


class ApaProcessor(DataProcessor):

    def get_train_examples(self, data):
        """See base class."""
        return self._create_examples(data, "train")

    def get_dev_examples(self, data):
        """See base class."""
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        docs = []
        labels = []
        for idx, row in lines.iterrows():
            guid = "%s-%s" % (set_type, idx)
            text = row['text']
            label = int(row['readability'])
            docs.append(text)
            labels.append(label)
        return guid, docs, labels


class MerlinProcessor(DataProcessor):

    def get_train_examples(self, data):
        """See base class."""
        return self._create_examples(data, "train")

    def get_dev_examples(self, data):
        """See base class."""
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        docs = []
        labels = []
        for idx, row in lines.iterrows():
            guid = "%s-%s" % (set_type, idx)
            text = row['text']
            label = int(row['readability'])
            docs.append(text)
            labels.append(label)
        return guid, docs, labels


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='weebit',
                        type=str,
                        help="The name of the task to train, should be 'onestopenglish', 'weebit', 'newsela' or 'ucbeniki'")
    parser.add_argument("--input_path",
                        default="data/WeeBit/weebit_reextracted.csv",
                        type=str,
                        help="Path to training input dataset.")
    parser.add_argument("--input_path_valid",
                        default="data/WeeBit/weebit_reextracted_valid.csv",
                        type=str,
                        help="Path to validation input dataset.")
    parser.add_argument("--input_path_test",
                        default="data/WeeBit/weebit_reextracted_test.csv",
                        type=str,
                        help="Path to test input dataset.")

    parser.add_argument("--output_dir",
                        default='supervised/BERT/results/weebit',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--same_test",
                        action='store_true',
                        default=True,
                        help="Whether to run training.")

    args = parser.parse_args()

    processors = {
        "newsela": NewselaProcessor,
        "capito": CapitoProcessor,
        "apa": ApaProcessor,
        "merlin": MerlinProcessor,
        "falko": FalkoProcessor,
        "cleg": ClegProcessor,      
    }

    num_labels_task = {
        "newsela": 11,
        "capito": 4,
        "apa": 3,
        "merlin": 5,
        "falko": 2,
        "cleg": 3,
    }

    input_paths = {
        "capito": args.input_path,
        "newsela": args.input_path,
        "apa": args.input_path,
        "merlin": args.input_path,
        "falko": args.input_path,
        "cleg": args.input_path,
    }
    

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    #print("Devicex is: " +str(device))
    #print("currx is: " +torch.cuda.current_device())

    logger.info("device: {} n_gpu: {}".format(
        device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    #input_path = input_paths[task_name]

    #df_data = pd.read_csv(input_path, encoding='utf8', sep='\t')
    #df_data = df_data.sample(frac=1, random_state=123)
    #y = df_data.readability.values

    predicted_all_folds = []
    true_all_folds = []

    accuracies_all_folds = []
    precision_all_folds = []
    recall_all_folds = []
    f1_all_folds = []
    qwk_all_folds = []

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    num_train_steps = None


    df_train = pd.read_csv(args.input_path, encoding='utf8', sep='\t')
    df_test = pd.read_csv(args.input_path_test, encoding='utf8', sep='\t')


    if args.do_train:
        train_guid, train_docs, train_labels = processor.get_train_examples(df_train)
        num_train_steps = int((len(train_docs) / args.train_batch_size) * args.num_train_epochs)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels = num_labels, output_attentions=False, output_hidden_states=False)
    model.to(device)

    t_total = num_train_steps

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_docs))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        train_dataloader = create_data_loader(train_docs, train_labels, tokenizer, args.max_seq_length, args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)
                loss = model(input_ids, attention_mask=attention_mask, labels=targets)[0]
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if args.do_train:
        torch.save(model_to_save.state_dict(), output_model_file)

    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(output_model_file)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict, output_attentions=False, output_hidden_states=False, num_labels=num_labels)
    model.to(device)

    if args.do_eval:
        eval_guid, eval_docs, eval_labels = processor.get_dev_examples(df_test)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_docs))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_dataloader = create_data_loader(eval_docs, eval_labels, tokenizer, args.max_seq_length, args.train_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0

        predicted_all = []
        true_all = []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, attention_mask=attention_mask, labels=targets)[0]
                logits = model(input_ids, attention_mask=attention_mask)[0]

            logits = logits.detach().cpu().numpy()
            targets = targets.to('cpu').numpy()

            preds = np.argmax(logits, axis=1).tolist()
            true = targets.tolist()
            predicted_all.extend(preds)
            true_all.extend(true)
            predicted_all_folds.extend(preds)
            true_all_folds.extend(true)

            eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        print("Task: ", task_name)
        accuracy_fold = accuracy_score(true_all, predicted_all)
        precision = precision_score(true_all, predicted_all, average='weighted')
        recall = recall_score(true_all, predicted_all, average='weighted')
        f1 = f1_score(true_all, predicted_all, average='weighted')
        qwk = cohen_kappa_score(true_all, predicted_all, weights="quadratic")
        accuracies_all_folds.append(accuracy_fold)
        precision_all_folds.append(precision)
        recall_all_folds.append(recall)
        f1_all_folds.append(f1)
        qwk_all_folds.append(qwk)

        np.savetxt("predicted_docs.csv", predicted_all, fmt='%d')
        np.savetxt("true_docs.csv", true_all, fmt='%d')  

        print("Accuracy fold: ", accuracy_score(true_all, predicted_all))
        print("Precision fold: ", precision_score(true_all, predicted_all, average='weighted'))
        print("Recall fold: ", recall_score(true_all, predicted_all, average='weighted'))
        print("F1 fold: ",f1_score(true_all, predicted_all, average='weighted'))
        print('Confusion matrix fold: ', confusion_matrix(true_all, predicted_all))
        print('Weighted kappa matrix: ', cohen_kappa_score(true_all, predicted_all, weights="quadratic"))

            #if task_name in ['newsela', 'capito', 'apa', 'merlin']:
             #   break

    del loss
    del tr_loss
    del tmp_eval_loss
    del eval_loss
    del eval_dataloader
    del optimizer
    del logits
    torch.cuda.empty_cache()


    print("Task: ", task_name)
    print("Accuracy: ", accuracy_score(true_all_folds, predicted_all_folds))
    print("Precision: ", precision_score(true_all_folds, predicted_all_folds, average='weighted'))
    print("Recall: ", recall_score(true_all_folds, predicted_all_folds, average='weighted'))
    print("F1: ", f1_score(true_all_folds, predicted_all_folds, average='weighted'))
    print('Confusion matrix: ', confusion_matrix(true_all_folds, predicted_all_folds))
    print('QWK: ', cohen_kappa_score(true_all_folds, predicted_all_folds, weights="quadratic"))

    print('All folds accuracy: ', accuracies_all_folds)
    print('All folds precision: ', precision_all_folds)
    print('All folds recall: ', recall_all_folds)
    print('All folds f1: ', f1_all_folds)
    print('All folds QWK: ', qwk_all_folds)

if __name__ == "__main__":
    main()




