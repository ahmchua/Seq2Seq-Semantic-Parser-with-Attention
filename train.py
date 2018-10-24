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

class Seq2SeqSemanticParser(object):
    def __init__(self, model_input_emb, model_enc, model_output_emb, model_dec, args, output_indexer):
        self.model_input_emb = model_input_emb
        self.model_enc = model_enc
        self.model_output_emb = model_output_emb
        self.model_dec = model_dec
        self.args = args
        self.output_indexer = output_indexer
        # Add any args you need here

    def decode(self, test_data):
        self.model_input_emb.eval()
        self.model_enc.eval()
        self.model_output_emb.eval()
        self.model_dec.eval()

        self.model_enc.zero_grad()
        self.model_dec.zero_grad()

        SOS_token = 1
        EOS_token = self.output_indexer.index_of('<EOS>')
        SOS_label = self.output_indexer.get_object(SOS_token)
        beam_length = 1
        derivations = []
        print("EOS_token: ", EOS_token)

        for ex in test_data:
            count = 0
            y_toks =[]
            self.model_enc.zero_grad()
            self.model_dec.zero_grad()

            x_tensor = torch.as_tensor([ex.x_indexed])
            y_tensor = torch.as_tensor(ex.y_indexed)
            inp_lens_tensor = torch.as_tensor([ex.x_len()])

            enc_output, enc_context_mask, enc_hidden = encode_input_for_decoder(x_tensor, inp_lens_tensor, self.model_input_emb, self.model_enc)
            dec_hidden = enc_hidden
            dec_input = torch.as_tensor([[SOS_token]])

            while (dec_input.item() != EOS_token) and count <= self.args.decoder_len_limit:
                dec_output, dec_input, dec_input_val, dec_hidden = decode(dec_input, dec_hidden, self.model_output_emb, self.model_dec, beam_length)
                y_label = self.output_indexer.get_object(dec_input.item())
                if dec_input.item() != EOS_token:
                    y_toks.append(y_label)
                count = count + 1
                print("dec_input: ", dec_input)
                print("dec_input.item(): ", dec_input.item())
            derivations.append([Derivation(ex, 1.0 , y_toks)])
            #print("prediction: ", y_toks)
        return derivations


# Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
# Optionally reverses them.
def make_padded_input_tensor(exs, input_indexer, max_len, reverse_input):
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

# Analogous to make_padded_input_tensor, but without the option to reverse input
def make_padded_output_tensor(exs, output_indexer, max_len):
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])


# Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
# inp_lens_tensor lengths.
# x_tensor: batch size x sent len tensor of input token indices
# inp_lens: batch size length vector containing the length of each sentence in the batch
# model_input_emb: EmbeddingLayer
# model_enc: RNNEncoder
# Returns the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting

# E.g., calling this with x_tensor (0 is pad token):
# [[12, 25, 0, 0],
#  [1, 2, 3, 0],
#  [2, 0, 0, 0]]
# inp_lens = [2, 3, 1]
# will return outputs with the following shape:
# enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
# enc_final_states = 3 x dim
def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc):
    input_emb = model_input_emb.forward(x_tensor)
    #print("input_emb: ", input_emb)
    (enc_output_each_word, enc_context_mask, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    #enc_final_states_reshaped = (enc_final_states[0].squeeze(), enc_final_states[1].squeeze())
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


def decode(y_index, hidden, model_output_emb, model_dec, beam_length):
    #print("y_ex: ", y_index)
    output_emb = model_output_emb.forward(y_index)
    #print("output_emb: ", output_emb)
    output, hidden = model_dec.forward(output_emb, hidden)
    top_val, top_ind = output.topk(beam_length)
    dec_input = top_ind.detach()
    dec_input_prob = top_val.detach()
    return output, dec_input, dec_input_prob, hidden

def train_model_encdec(ex, model_input_emb, model_enc, model_output_emb, model_dec, enc_optimizer, dec_optimizer, beam_length, teacher_forcing_ratio):
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    #train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    #test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    #input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    #all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, args.reverse_input)
    #all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, args.reverse_input)

    #output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    #all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    #all_test_output_data = make_padded_output_tensor(test_data, output_indexer, output_max_len)

    #print("Train length: %i" % input_max_len)
    #print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    #print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # Initialize loss
    loss = 0
    SOS_token = 1
    EOS_token = 2
    criterion = torch.nn.NLLLoss()

    # Loop over epochs, loop over examples, given some indexed words, call encode_input_for_decoder, then call your
    # decoder, accumulate losses, update parameters

    model_enc.zero_grad()
    model_dec.zero_grad()

    x_tensor = torch.as_tensor([ex.x_indexed])
    #y_tensor = torch.as_tensor(ex.x_indexed)
    y_tensor = torch.as_tensor(ex.y_indexed)
    inp_lens_tensor = torch.as_tensor([ex.x_len()])
    enc_output, enc_context_mask, enc_hidden = encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc)
    dec_hidden = enc_hidden

    #print("enc_hidden shape", enc_hidden)

    dec_input = torch.as_tensor([[SOS_token]])
    y_temp = []

    teacher_forcing = True if random.random() <= teacher_forcing_ratio else False

    for y_ex in range(len(y_tensor)):
        dec_output, dec_input, dec_input_val, dec_hidden = decode(dec_input, dec_hidden, model_output_emb, model_dec, beam_length)
        loss += criterion(dec_output, y_tensor[y_ex].unsqueeze(0))
        y_temp.append(int(dec_input.squeeze()))
        #if dec_input ==EOS_token:
            #break
        if teacher_forcing:
            dec_input = y_tensor[y_ex].unsqueeze(0).unsqueeze(0)

    #print("predicted: ", y_temp)
    #print("actual: ", ex.x_indexed, '\n')

    loss.backward()
    enc_optimizer.step()
    dec_optimizer.step()

    return loss

def train_iters(train_data, epochs, input_indexer, output_indexer, args, beam_length, out):

    # Create encoder, decoder and embedding layers
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    model_output_emb = EmbeddingLayer(args.input_dim, len(output_indexer), args.emb_dropout)
    model_dec = RNNDecoder(args.input_dim, args.hidden_size, out, dropout = 0 )

    # Set all models to training mode
    model_input_emb.train()
    model_enc.train()
    model_output_emb.train()
    model_dec.train()

    #Initialize optimizers
    enc_optimizer = optim.Adam(model_enc.parameters(), lr=args.lr)
    dec_optimizer = optim.Adam(model_dec.parameters(), lr=args.lr)

    count = 0.0
    for i in range(epochs):
        loss = 0.0
        for ex in train_data:
            inc_loss = train_model_encdec(ex, model_input_emb, model_enc, model_output_emb, model_dec, enc_optimizer, dec_optimizer, beam_length, args.teacher_forcing_ratio)
            if count %25 ==0:
                print ("loss: ", inc_loss)
            count = count + 1

        print("epoch: ", i)
        print("loss: ", loss)

    return Seq2SeqSemanticParser(model_input_emb, model_enc, model_output_emb, model_dec, args, output_indexer)


# Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
# every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
# executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
# example with a valid denotation (if you've provided more than one).
def evaluate(test_data, decoder, example_freq=50, outfile=None):
    e = GeoqueryDomain()
    pred_derivations = decoder.decode(test_data)
    #print("\n pred_derivations: ", pred_derivations, '\n')
    selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations)
    num_exact_match = 0
    num_tokens_correct = 0
    num_denotation_match = 0
    total_tokens = 0
    #print("selected_derivs_len: ", len(selected_derivs))
    for i, ex in enumerate(test_data):
        #print("i: ", i)
        #print(selected_derivs[i])
        if i % example_freq == 0:
            print('Example %d' % i)
            print('  x      = "%s"' % ex.x)
            print('  y_tok  = "%s"' % ex.y_tok)
            print('  y_pred = "%s"' % selected_derivs[i].y_toks)
        # Compute accuracy metrics
        y_pred = ' '.join(selected_derivs[i].y_toks)
        # Check exact match
        if y_pred == ' '.join(ex.y_tok):
            num_exact_match += 1
        # Check position-by-position token correctness
        num_tokens_correct += sum(a == b for a, b in zip(selected_derivs[i].y_toks, ex.y_tok))
        total_tokens += len(ex.y_tok)
        # Check correctness of the denotation
        if denotation_correct[i]:
            num_denotation_match += 1
    print("Exact logical form matches: %s" % (render_ratio(num_exact_match, len(test_data))))
    print("Token-level accuracy: %s" % (render_ratio(num_tokens_correct, total_tokens)))
    print("Denotation matches: %s" % (render_ratio(num_denotation_match, len(test_data))))
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()


def render_ratio(numer, denom):
    return "%i / %i = %.3f" % (numer, denom, float(numer)/denom)
