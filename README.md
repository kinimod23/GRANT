# GRANT - A German Readability Assessment Neural network Tool

* based on the approach of Martinc et al. https://arxiv.org/pdf/1907.11779.pdf
* Newsela Dataset https://www.aclweb.org/anthology/Q15-1021.pdf
* Falko Dataset https://www.linguistik.hu-berlin.de/de/institut/professuren/korpuslinguistik/forschung/falko/Falko-Handbuchv2.0.pdf
* CLEG Dataset https://korpling.german.hu-berlin.de/public/CLEG13/CLEG13_documentation.pdf
* WebCorpus Dataset https://www.aclweb.org/anthology/2020.lrec-1.404/
* APA Dataset https://www.capito.eu
* Capito Dataset https://www.capito.eu
* MERLIN Dataset https://www.aclweb.org/anthology/L14-1488/
* Evaluation https://www.derdiedaf.com/c-191


----------------------------------------------------------------------------------------------

## Unsupervised Approach
(training on ~ 1.078 million simple sentences from German Simple Wikipedia)

### COLSTM 

Train:

    python unsupervised/COLSTM/train.py --data_dir data/GPWKP-simple --checkpoint_dir unsupervised/COLSTM/checkpoints --savefile 'gpwkp-simple-1.078M'

Apply:

    python unsupervised/COLSTM/apply.py --data data/MERLIN/merlin_martinc_all4uns.csv --output unsupervised/COLSTM/results/results_uns_colstm_merlin.csv --model unsupervised/COLSTM/checkpoints/gpwkp-simple-1.078M.h5 --settings unsupervised/COLSTM/checkpoints/gpwkp-simple-1.078M.pkl --vocabulary data/GPWKP-simple/vocab.npz

Evaluate:

    python unsupervised/evaluate.py  --input unsupervised/COLSTM/results/results_uns_colstm_merlin.csv


### TCN

Train:

    python unsupervised/TCN/train.py --data data/GPWKP-simple --model_path unsupervised/TCN/trained_models/gpwkp-simple-1.078M.pt

Apply:

    python3 unsupervised/TCN/apply.py --data data/MERLIN/merlin_martinc_all4uns.csv --lm_model_path unsupervised/TCN/trained_models/tcn_gpwkp-simple-1.078M_rattle.pt --lm_data data/GPWKP-simple --results_path unsupervised/TCN/results/results_uns_tcn_merlin.csv

Evaluate:

    python unsupervised/evaluate.py  --input unsupervised/TCN/results/results_uns_tcn_merlin.csv


### BERT

no extra training!

pre-trained model: *bert-base-german-cased* <sup>1</sup> is used 

Apply:

    python3 unsupervised/BERT/apply.py --input data/MERLIN/merlin_martinc_all4uns.csv --output unsupervised/BERT/results/results_uns_bert_merlin.csv 

Evaluate:

    python unsupervised/evaluate.py  --input unsupervised/BERT/results/results_uns_bert_merlin.csv


<sup>1</sup>https://huggingface.co/transformers/pretrained_models.html

----------------------------------------------------------------------------------------------

## Supervised Approach

### HAN training + evaluation

Documents:

    python supervised/HAN/train_test.py --task_name merlin --input_path data/MERLIN/merlin_martinc_docs_train.csv --input_path_valid data/MERLIN/merlin_martinc_docs_valid.csv --input_path_test data/MERLIN/merlin_martinc_docs_test.csv --saved_path supervised/HAN/trained_models --vocab_path supervised/HAN/vocab/merlin_docs_vocab.pk

Sentences:

    python supervised/HAN/train_test.py --task_name merlin --input_path data/MERLIN/merlin_martinc_sents_train.csv --input_path_valid data/MERLIN/merlin_martinc_sents_valid.csv --input_path_test data/MERLIN/merlin_martinc_sents_test.csv --saved_path supervised/HAN/trained_models --vocab_path supervised/HAN/vocab/merlin_sents_vocab.pk


### BiLSTM training + evaluation

Documents:

    python supervised/BiLSTM/train_test.py --task_name merlin --input_path data/MERLIN/merlin_martinc_docs_train.csv --input_path_valid data/MERLIN/merlin_martinc_docs_valid.csv --input_path_test data/MERLIN/merlin_martinc_docs_test.csv --saved_path supervised/BiLSTM/trained_models

Sentences:

    python supervised/BiLSTM/train_test.py --task_name merlin --input_path data/MERLIN/merlin_martinc_sents_train.csv --input_path_valid data/MERLIN/merlin_martinc_sents_valid.csv --input_path_test data/MERLIN/merlin_martinc_sents_test.csv --saved_path supervised/BiLSTM/trained_models


### BERT training + evaluation

Documents:

    python supervised/BERT/run_classifier.py --task_name merlin --input_path data/MERLIN/merlin_martinc_docs_trainbert.csv --input_path_test data/MERLIN/merlin_martinc_docs_test.csv --output_dir supervised/BERT/trained_models

Sentences:

    python supervised/BERT/run_classifier.py --task_name merlin --input_path data/MERLIN/merlin_martinc_sents_trainbert.csv --input_path_test data/MERLIN/merlin_martinc_sents_test.csv --output_dir supervised/BERT/trained_models



---------------------------------------------------------------------------------------------------