import argparse
import random
import numpy as np
import time
import torch
from torch import optim
from lf_evaluator import *
from models import *
from data import *
from utils import *
from train import *
from sentiment_data import *

def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')

    # General system running and configuration options
    parser.add_argument('--do_nearest_neighbor', dest='do_nearest_neighbor', default=False, action='store_true', help='run the nearest neighbor model')

    parser.add_argument('--train_path', type=str, default='data/geo_train.tsv', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/geo_dev.tsv', help='path to dev data')
    parser.add_argument('--test_path', type=str, default='data/geo_test.tsv', help='path to blind test data')
    parser.add_argument('--test_output_path', type=str, default='geo_test_output.tsv', help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo', help='domain (geo for geoquery)')

    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=1, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')
    parser.add_argument('--input_dim', type=int, default=100, help='input vector dimensionality')
    parser.add_argument('--output_dim', type=int, default=100, help='output vector dimensionality')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden state dimensionality')
    parser.add_argument('--emb_dropout', type=int, default=0.2, help='Dropout for embedding layer')
    parser.add_argument('--rnn_dropout', type=int, default=0, help='Dropout for LSTM')
    parser.add_argument('--bidirectional', dest='bidirectional', default=True, action='store_true', help='run the nearest neighbor model')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0)
    parser.add_argument('--attn', type=str, default='N')
    parser.add_argument('--beam_size', type=int, default=3, help='beam size for beam search')
    parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.300d-relativized.txt', help='path to word vectors file')
    args = parser.parse_args()
    return args


# Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
# returns the associated logical form.
class NearestNeighborSemanticParser(object):
    # Take any arguments necessary for parsing
    def __init__(self, training_data):
        self.training_data = training_data

    # decode should return a list of k-best lists of Derivations. A Derivation consists of the underlying Example,
    # a probability, and a tokenized output string. If you're just doing one-best decoding of example ex and you
    # produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
    def decode(self, test_data):
        # Find the highest word overlap with the test data
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            for train_ex in self.training_data:
                # Compute word overlap
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # N.B. a list!
            print('best_train_ex.y_tok: ', best_train_ex.y_tok)
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data

    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev, test, args.decoder_len_limit)
    word_vectors = read_word_embeddings(args.word_vecs_path)
    print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    print("Input indexer: %s" % input_indexer)
    print("Output indexer: %s" % output_indexer)
    print("Here are some examples post tokenization and indexing:")
    for i in range(0, min(len(train_data_indexed), 10)):
        print(train_data_indexed[i])
    print(type(train_data_indexed[2]))
    print(output_indexer.get_object(3))
    if args.do_nearest_neighbor:
        print("Doing nearest neighbor")
        decoder = NearestNeighborSemanticParser(train_data_indexed)
        test = decoder.decode(test_data_indexed[0:2])
        print(test)
        evaluate(dev_data_indexed, decoder)
    else:
        beam_length = 1
        out = len(output_indexer)
        #train_data_indexed = train_data_indexed[0:50]
        #test_data_indexed  = test_data_indexed[0:1]
        #dev_data_indexed = dev_data_indexed[0:10]
        decoder = train_iters(train_data_indexed, args.epochs, input_indexer, output_indexer, args, beam_length, out, word_vectors)
        #test = decoder.decode_beam(dev_data_indexed)
        # Next line tests copy act
        #decoder = train_iters(train_data_indexed, args.epochs, input_indexer, input_indexer, args, beam_length, out)

        #pred = decoder.decode_beam(dev_data_indexed)
        #pred_original = decoder.decode(dev_data_indexed)
        #print("pred: ", pred_original)
        print("BEGIN EVALUATION")
        evaluate(dev_data_indexed, decoder)
        #test = [(' '.join(d.y_toks)) for x in pred for d in x]
        #for x in pred:
            #for d in x:
                #print("y_toks: ", d.y_toks)
        #for i in test_data_indexed:
            #print(i.y_tok)
    #print("=======FINAL EVALUATION=======")
    #evaluate(test_data_indexed, decoder, outfile="geo_test_output.tsv")
