import pandas as pd
from scipy.stats.stats import pearsonr
import argparse


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Evaluate perplexity and RSRS scores of the BERT model')
    argparser.add_argument('--input', dest='input', type=str,
                           default='unsupervised/BERT/results/results_weebit.csv',
                           help='Choose input csv file produced by the apply.py language model script')
    args = argparser.parse_args()

    data = pd.read_csv(args.input, encoding="utf-8", delimiter="\t")
    data = data.dropna(0)
    print("Pearson perplexity results: ",pearsonr(data['class'], data['perplexity']))
    print("Pearson RSRS results: ", pearsonr(data['class'], data['score']))
