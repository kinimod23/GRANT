import argparse
import time
import math
import torch.optim as optim
import sys
sys.path.append("../../")
from utils import *
from model import *

def evaluate(data_source):
    print('Evaluating')
    model.eval()
    total_loss = 0
    processed_data_size = 0
    for i in range(0, data_source.size(1) - 1, args.validseqlen):
        #print(data_source.size())
        if i + args.seq_len - args.validseqlen >= data_source.size(1) - 1:
            continue
        data, targets = get_batch(data_source, i, args, evaluation=True)
        #print(data, targets)
        output = model(data)
        #print(output.size())

        # Discard the effective history, just like in training
        eff_history = args.seq_len - args.validseqlen
        final_output = output[:, eff_history:].contiguous().view(-1, n_words)
        final_target = targets[:, eff_history:].contiguous().view(-1)

        loss = criterion(final_output, final_target)

        # Note that we don't add TAR loss here
        total_loss += (data.size(1) - eff_history) * loss.item()
        processed_data_size += data.size(1) - eff_history

    return total_loss / processed_data_size

def train():
    print('Training')
    # Turn on training mode which enables dropout.
    global train_data
    model.train()
    total_loss = 0
    start_time = time.time()
    for batch_idx, i in enumerate(range(0, train_data.size(1) - 1, args.validseqlen)):
        if i + args.seq_len - args.validseqlen >= train_data.size(1) - 1:
            continue
        data, targets = get_batch(train_data, i, args)
        #print(data.size())
        optimizer.zero_grad()
        output = model(data)

        # Discard the effective history part
        eff_history = args.seq_len - args.validseqlen
        if eff_history < 0:
            raise ValueError("Valid sequence length must be smaller than sequence length!")
        final_target = targets[:, eff_history:].contiguous().view(-1)
        final_output = output[:, eff_history:].contiguous().view(-1, n_words)
        loss = criterion(final_output, final_target)

        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            #print(total_loss.cpu().numpy())
            cur_loss = total_loss.cpu().numpy() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_idx, train_data.size(1) // args.validseqlen, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling - Word-level Language Modeling')
    parser.add_argument('--data', type=str, default='data/simple-wiki',
                        help='location of the data corpus used for language model training. Should contain train.txt/valid.txt with input data')
    parser.add_argument('--model_path', type=str, default="unsupervised/TCN/cv/model_normal_wiki_50000.pt",
                        help='Where to save the trained language model')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size (default: 16)')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA (default: True)')
    parser.add_argument('--dropout', type=float, default=0.45,
                        help='dropout applied to layers (default: 0.45)')
    parser.add_argument('--emb_dropout', type=float, default=0.25,
                        help='dropout applied to the embedded layer (default: 0.25)')
    parser.add_argument('--clip', type=float, default=0.35,
                        help='gradient clip, -1 means no clip (default: 0.35)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit (default: 100)')
    parser.add_argument('--ksize', type=int, default=3,
                        help='kernel size (default: 3)')
    parser.add_argument('--emsize', type=int, default=600,
                        help='size of word embeddings (default: 600)')
    parser.add_argument('--levels', type=int, default=4,
                        help='# of levels (default: 4)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval (default: 100)')
    parser.add_argument('--lr', type=float, default=4,
                        help='initial learning rate (default: 4)')
    parser.add_argument('--nhid', type=int, default=600,
                        help='number of hidden units per layer (default: 600)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1111)')
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

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        print(torch.cuda.is_available())
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    print(args)
    vocabulary = generate_vocabulary(args.data, args.vocabulary_size)
    corpus = data_generator(args, vocabulary)
    print('Dict created')
    eval_batch_size = 1
    train_data = batchify(corpus.train, args.batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, eval_batch_size, args)

    with open(args.data + '/valid.txt', 'r') as f:
        document = f.read()
    test_corpus = CorpusTest(document, vocabulary, corpus.dictionary)
    test_data = batchify(test_corpus.test, eval_batch_size, args)

    n_words = len(corpus.dictionary)
    print("Vocabulary: ", n_words)

    num_chans = [args.nhid] * (args.levels - 1) + [args.emsize]
    k_size = args.ksize
    dropout = args.dropout
    emb_dropout = args.emb_dropout
    tied = args.tied
    model = TCN(args.emsize, n_words, num_chans, dropout=dropout, emb_dropout=emb_dropout, kernel_size=k_size,
                tied_weights=tied)

    if args.cuda:
        model.cuda()

    # May use adaptive softmax to speed up training
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=-1)

    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

    best_vloss = 1e8

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        all_vloss = []
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)
            test_loss = evaluate(test_data)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                  'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            test_loss, math.exp(test_loss)))
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_vloss:
                with open(args.model_path, 'wb') as f:
                    print('Save model!\n')
                    torch.save(model, f)
                best_vloss = val_loss

            # Anneal the learning rate if the validation loss plateaus
            if epoch > 5 and val_loss >= max(all_vloss[-5:]):
                lr = lr / 2.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            all_vloss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
