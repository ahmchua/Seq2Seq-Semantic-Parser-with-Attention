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

            enc_output, enc_context_mask, enc_hidden, enc_bi_hidden = encode_input_for_decoder(x_tensor, inp_lens_tensor, self.model_input_emb, self.model_enc)
            dec_hidden = enc_hidden
            dec_input = torch.as_tensor([[SOS_token]])

            while (dec_input.item() != EOS_token) and count <= self.args.decoder_len_limit:
                dec_output, dec_input, dec_input_val, dec_hidden = decode(dec_input, dec_hidden, self.model_output_emb, self.model_dec, beam_length)
                y_label = self.output_indexer.get_object(dec_input.item())
                if dec_input.item() != EOS_token:
                    y_toks.append(y_label)
                count = count + 1
                #print("dec_input: ", dec_input)
                #print("dec_input.item(): ", dec_input.item())
            derivations.append([Derivation(ex, 1.0 , y_toks)])
            #print("prediction: ", y_toks)
        return derivations


class Seq2SeqSemanticParserAttn(object):
    def __init__(self, model_input_emb, model_enc, model_output_emb, model_dec, args, output_indexer):
        self.model_input_emb = model_input_emb
        self.model_enc = model_enc
        self.model_output_emb = model_output_emb
        self.model_dec = model_dec
        self.args = args
        self.output_indexer = output_indexer
        self.beam_size = args.beam_size
        # Add any args you need here

    def decode(self, test_data):
        self.model_input_emb.eval()
        self.model_enc.eval()
        self.model_output_emb.eval()
        self.model_dec.eval()

        self.model_input_emb.zero_grad()
        self.model_output_emb.zero_grad()
        self.model_enc.zero_grad()
        self.model_dec.zero_grad()

        SOS_token = 1
        #EOS_token = 8
        EOS_token = self.output_indexer.index_of('<EOS>')
        SOS_label = self.output_indexer.get_object(SOS_token)
        beam_length = 1
        derivations = []
        print("EOS_token: ", EOS_token)

        for ex in test_data:
            count = 0
            y_toks =[]
            self.model_input_emb.zero_grad()
            self.model_output_emb.zero_grad()
            self.model_enc.zero_grad()
            self.model_dec.zero_grad()

            x_tensor = torch.as_tensor([ex.x_indexed])
            y_tensor = torch.as_tensor(ex.y_indexed)
            inp_lens_tensor = torch.as_tensor([ex.x_len()])

            enc_output, enc_context_mask, enc_hidden, enc_bi_hidden = encode_input_for_decoder(x_tensor, inp_lens_tensor, self.model_input_emb, self.model_enc)
            dec_hidden = enc_hidden
            context_vec = enc_bi_hidden[0]
            dec_input = torch.as_tensor([[SOS_token]])
            y_temp = []

            while (dec_input.item() != EOS_token) and count <= self.args.decoder_len_limit:
                dec_output, dec_input, dec_input_val, dec_hidden, context_vec = decode_attn(dec_input, dec_hidden, self.model_output_emb, self.model_dec, context_vec, enc_output, beam_length)
                y_label = self.output_indexer.get_object(dec_input.item())
                if dec_input.item() != EOS_token:
                    y_toks.append(y_label)
                    y_temp.append(dec_input.item())
                count = count + 1
                #print("dec_input: ", dec_input)
                #print("dec_input.item(): ", dec_input.item())
            derivations.append([Derivation(ex, 1.0 , y_toks)])
            #print("true path: ", ex.x_indexed)
            #print("prediction: ", y_temp)
        return derivations

    def decode_beam(self, test_data):
        print("DECODING BEAM")
        self.model_input_emb.eval()
        self.model_enc.eval()
        self.model_output_emb.eval()
        self.model_dec.eval()

        self.model_input_emb.zero_grad()
        self.model_output_emb.zero_grad()
        self.model_enc.zero_grad()
        self.model_dec.zero_grad()

        SOS_token = 1
        EOS_token = self.output_indexer.index_of('<EOS>')
        SOS_label = self.output_indexer.get_object(SOS_token)

        beam = Beam(self.beam_size)
        #beam_temp = Beam(self.beam_size)

        derivations = []
        ex_count = 0
        #print("EOS_token: ", EOS_token)

        for ex in test_data:
            ex_derivs = []
            if ex_count %25 ==0:
                print("Current train idx: ", ex_count)
            count = 0
            y_toks =[]
            self.model_enc.zero_grad()
            self.model_dec.zero_grad()

            x_tensor = torch.as_tensor([ex.x_indexed])
            y_tensor = torch.as_tensor(ex.y_indexed)
            inp_lens_tensor = torch.as_tensor([ex.x_len()])

            enc_output, enc_context_mask, enc_hidden, enc_bi_hidden = encode_input_for_decoder(x_tensor, inp_lens_tensor, self.model_input_emb, self.model_enc)
            dec_hidden = enc_hidden
            context_vec = enc_bi_hidden[0]
            dec_input = torch.as_tensor([[SOS_token]])
            beam.add(dec_input, 0.0, dec_hidden, [])

            while count <= self.args.decoder_len_limit:
                beam_temp = Beam(self.beam_size)
                # Check if all dec_input items are EOS, if so, break.
                if beam.all_EOS(EOS_token):
                    break
                for dec_input, dec_prob, dec_hidden, path in beam.get_elts_and_scores():
                    #print("dec_input: ", dec_input)
                    if dec_input.item() == EOS_token:
                        beam_temp.add(dec_input, dec_prob, dec_hidden, path)
                        #print ("FOUND EOS TOKEN")
                    if dec_input.item() != EOS_token:
                        dec_output, dec_input, dec_input_val, dec_hidden, context_vec = decode_attn(dec_input, dec_hidden, self.model_output_emb, self.model_dec, context_vec, enc_output, self.beam_size)
                        dec_input = dec_input.squeeze()
                        dec_input_val = dec_input_val.squeeze()
                        #print("dec_input after decode: ", dec_input)
                        #print("dec_input: ", dec_input)
                        #print("dec_input_val", dec_input_val)
                        for i in range(len(dec_input)):
                            y_label = self.output_indexer.get_object(dec_input[i].item())
                            #path.append(y_label)
                            #path.append(dec_input[i].item())
                            #path_temp = path + [y_label]
                            dec_prob_temp = (dec_prob + dec_input_val[i].item())/(len(path)+1)
                            #print("Score of new path: ", dec_prob_temp)
                            beam_temp.add(dec_input[i].unsqueeze(0).unsqueeze(0), dec_prob_temp, dec_hidden, path+[y_label])
                        #print("current beam elements: ", beam_temp.elts)
                        #print("current beam score: ", beam_temp.scores)
                        #print("current beam path: ", beam_temp.path)
                beam = beam_temp
                count = count + 1
            ex_count = ex_count + 1

            print("True path: ", ex.x_indexed)
            for dec_input, dec_prob, dec_hidden, path in beam.get_elts_and_scores():
                ex_derivs.append(Derivation(ex, dec_prob, path))
                print("Predicted path: ", path)

            derivations.append(ex_derivs)
        return derivations

class Seq2SeqSemanticParserAttnCopy(object):
    def __init__(self, model_input_emb, model_enc, model_output_emb, model_dec, args, indexers):
        self.model_input_emb = model_input_emb
        self.model_enc = model_enc
        self.model_output_emb = model_output_emb
        self.model_dec = model_dec
        self.args = args
        self.input_indexer, self.output_indexer = indexers
        self.beam_size = args.beam_size
        # Add any args you need here

    def decode(self, test_data):
        self.model_input_emb.eval()
        self.model_enc.eval()
        self.model_output_emb.eval()
        self.model_dec.eval()

        self.model_input_emb.zero_grad()
        self.model_output_emb.zero_grad()
        self.model_enc.zero_grad()
        self.model_dec.zero_grad()

        SOS_token = 1
        #EOS_token = 8
        EOS_token = self.output_indexer.index_of('<EOS>')
        SOS_label = self.output_indexer.get_object(SOS_token)
        beam_length = 1
        derivations = []
        print("EOS_token: ", EOS_token)

        for ex in test_data:
            count = 0
            y_toks =[]
            self.model_input_emb.zero_grad()
            self.model_output_emb.zero_grad()
            self.model_enc.zero_grad()
            self.model_dec.zero_grad()

            x_tensor = torch.as_tensor([ex.x_indexed])
            y_tensor = torch.as_tensor(ex.y_indexed)
            inp_lens_tensor = torch.as_tensor([ex.x_len()])

            enc_output, enc_context_mask, enc_hidden, enc_bi_hidden = encode_input_for_decoder(x_tensor, inp_lens_tensor, self.model_input_emb, self.model_enc)
            dec_hidden = enc_hidden
            context_vec = enc_bi_hidden[0]
            dec_input = torch.as_tensor([[SOS_token]])
            input_output_map = create_input_output_map(ex.x_indexed, self.input_indexer, self.output_indexer)

            y_temp = []

            while (dec_input.item() != EOS_token) and count <= self.args.decoder_len_limit:
                dec_output, dec_input, dec_input_val, dec_hidden, context_vec, for_inference = decode_copy_attn(dec_input, dec_hidden, self.model_output_emb, self.model_dec, context_vec, enc_output, beam_length, input_output_map)
                top_val, top_ind = for_inference.topk(1)
                top_ind = top_ind.detach()
                if top_ind <len(self.output_indexer):
                    y_label = self.output_indexer.get_object(dec_input.item())
                else:
                    source_word_idx = top_ind.item() - len(self.output_indexer)
                    copy_word_idx = ex.x_indexed[source_word_idx]
                    y_label = self.input_indexer.get_object(copy_word_idx)
                    #dec_input = torch.as_tensor([[self.output_indexer.index_of(UNK_SYMBOL)]])
                if dec_input.item() != EOS_token:
                    y_toks.append(y_label)
                count = count + 1
                #print("dec_input: ", dec_input)
                #print("dec_input.item(): ", dec_input.item())
            derivations.append([Derivation(ex, 1.0 , y_toks)])
            #print("true path: ", ex.x_indexed)
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

def create_input_output_map(input_seq, input_indexer, output_indexer):
    map = torch.zeros(len(input_seq), len(output_indexer))
    unk_index = output_indexer.index_of(UNK_SYMBOL)
    for i, idx in enumerate(input_seq):
        word = input_indexer.get_object(idx)
        out_index = output_indexer.index_of(word)
        if out_index == -1:
            map[i][unk_index] =  1.0
        else:
            map[i][out_index] =  1.0
    return map

def create_in_out_dict(input_indexer, output_indexer):
    in_out = {}
    unk_index = output_indexer.index_of(UNK_SYMBOL)
    for k, v in input_indexer.ints_to_objs.items():
        out_index = output_indexer.index(v)
        if out_index == -1:
            in_out[k] = unk_index
        else:
            in_out[k] = out_index
    return in_out


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
    (enc_output_each_word, enc_context_mask, enc_final_states, enc_bi_hidden) = model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    #enc_final_states_reshaped = (enc_final_states[0].squeeze(), enc_final_states[1].squeeze())
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped, enc_bi_hidden)


def decode(y_index, hidden, model_output_emb, model_dec, beam_length):
    #print("y_ex: ", y_index)
    output_emb = model_output_emb.forward(y_index)
    output, hidden = model_dec.forward(output_emb, hidden)
    top_val, top_ind = output.topk(beam_length)
    dec_input = top_ind.detach()
    dec_input_prob = top_val.detach()
    return output, dec_input, dec_input_prob, hidden

def decode_attn(y_index, hidden, model_output_emb, model_dec, context_vec, encoder_output, beam_length):
    output_emb = model_output_emb.forward(y_index)
    #print("output_emb: ", output_emb)
    #print("output_emb size: ", output_emb.squeeze(1).size())
    #print("context_vec_0 size: ", context_vec.size())
    output_emb= torch.cat((output_emb.squeeze(1), context_vec), dim = 1) #CHANGE
    output_emb = output_emb.unsqueeze(0)
    #print("Concatenated value size: ", test.size())
    output, hidden, context_vec = model_dec.forward(output_emb, hidden, encoder_output)
    top_val, top_ind = output.topk(beam_length)
    dec_input = top_ind.detach()
    dec_input_prob = top_val.detach()
    return output, dec_input, dec_input_prob, hidden, context_vec

def decode_copy_attn(y_index, hidden, model_output_emb, model_dec, context_vec, encoder_output, beam_length, input_output_map):
    output_emb = model_output_emb.forward(y_index)
    #print("output_emb: ", output_emb)
    #print("output_emb size: ", output_emb.squeeze(1).size())
    #print("context_vec_0 size: ", context_vec.size())
    output_emb= torch.cat((output_emb.squeeze(1), context_vec), dim = 1) #CHANGE
    output_emb = output_emb.unsqueeze(0)
    #print("Concatenated value size: ", test.size())
    output, hidden, context_vec, for_inference = model_dec.forward(output_emb, hidden, encoder_output, input_output_map)
    top_val, top_ind = output.topk(beam_length)
    dec_input = top_ind.detach()
    dec_input_prob = top_val.detach()
    return output, dec_input, dec_input_prob, hidden, context_vec, for_inference

def train_model_encdec(ex, model_input_emb, model_enc, model_output_emb, model_dec, enc_optimizer, dec_optimizer, input_emb_optimizer, output_emb_optimizer,beam_length, teacher_forcing_ratio):


    # Set all models to training mode
    model_input_emb.train()
    model_enc.train()
    model_output_emb.train()
    model_dec.train()

    # Initialize loss
    loss = 0
    SOS_token = 1
    EOS_token = 2
    criterion = torch.nn.NLLLoss()

    # Loop over epochs, loop over examples, given some indexed words, call encode_input_for_decoder, then call your
    # decoder, accumulate losses, update parameters

    model_input_emb.zero_grad()
    model_output_emb.zero_grad()
    model_enc.zero_grad()
    model_dec.zero_grad()

    x_tensor = torch.as_tensor([ex.x_indexed])
    #y_tensor = torch.as_tensor(ex.x_indexed)
    y_tensor = torch.as_tensor(ex.y_indexed)
    inp_lens_tensor = torch.as_tensor([ex.x_len()])
    enc_output, enc_context_mask, enc_hidden, enc_bi_hidden = encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc)
    dec_hidden = enc_hidden

    #print("enc_hidden shape", enc_hidden)

    dec_input = torch.as_tensor([[SOS_token]])
    y_temp = []

    teacher_forcing = True if random.random() <= teacher_forcing_ratio else False

    for y_ex in range(len(y_tensor)):
        dec_output, dec_input, dec_input_val, dec_hidden = decode(dec_input, dec_hidden, model_output_emb, model_dec, beam_length)
        loss += criterion(dec_output, y_tensor[y_ex].unsqueeze(0))
        y_temp.append(int(dec_input.squeeze()))
        if dec_input ==EOS_token:
            break
        if teacher_forcing:
            dec_input = y_tensor[y_ex].unsqueeze(0).unsqueeze(0)

    loss.backward()
    enc_optimizer.step()
    dec_optimizer.step()
    input_emb_optimizer.step()
    output_emb_optimizer.step()


    return loss

def train_attn(ex, model_input_emb, model_enc, model_output_emb, model_dec, enc_optimizer, dec_optimizer, input_emb_optimizer, output_emb_optimizer, beam_length, teacher_forcing_ratio):
    loss = 0
    SOS_token = 1
    EOS_token = 2
    criterion = torch.nn.NLLLoss()

    # Set all models to training mode
    model_input_emb.train()
    model_enc.train()
    model_output_emb.train()
    model_dec.train()

    model_input_emb.zero_grad()
    model_output_emb.zero_grad()
    model_enc.zero_grad()
    model_dec.zero_grad()

    x_tensor = torch.as_tensor([ex.x_indexed])
    #y_tensor = torch.as_tensor(ex.x_indexed)
    y_tensor = torch.as_tensor(ex.y_indexed)
    inp_lens_tensor = torch.as_tensor([ex.x_len()])

    (enc_output, enc_context_mask, enc_hidden, enc_bi_hidden) = encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc)
    dec_hidden = enc_hidden
    context_vec = enc_bi_hidden[0]

    dec_input = torch.as_tensor([[SOS_token]])
    y_temp = []

    # Initialize loss
    loss = 0
    SOS_token = 1
    EOS_token = 2
    criterion = torch.nn.NLLLoss()

    teacher_forcing = True if random.random() <= teacher_forcing_ratio else False

    for y_ex in range(len(y_tensor)):
        dec_output, dec_input, dec_input_val, dec_hidden, context_vec = decode_attn(dec_input, dec_hidden, model_output_emb, model_dec, context_vec, enc_output, beam_length)
        loss += criterion(dec_output, y_tensor[y_ex].unsqueeze(0))
        y_temp.append(dec_input.item())
        if dec_input == EOS_token:
            break
        if teacher_forcing:
            dec_input = y_tensor[y_ex].unsqueeze(0).unsqueeze(0)

    loss.backward()
    input_emb_optimizer.step()
    output_emb_optimizer.step()
    enc_optimizer.step()
    dec_optimizer.step()

    return loss

def train_copy_attn(ex, model_input_emb, model_enc, model_output_emb, model_dec, enc_optimizer, dec_optimizer, input_emb_optimizer, output_emb_optimizer, beam_length, teacher_forcing_ratio, indexers):
    # Unpack packed items
    #print("TRAINING ATTENTION WITH COPY")
    input_indexer, output_indexer = indexers

    # Set all models to training mode
    model_input_emb.train()
    model_enc.train()
    model_output_emb.train()
    model_dec.train()

    loss = 0
    SOS_token = 1
    EOS_token = 2
    criterion = torch.nn.CrossEntropyLoss()

    model_input_emb.zero_grad()
    model_output_emb.zero_grad()
    model_enc.zero_grad()
    model_dec.zero_grad()

    x_tensor = torch.as_tensor([ex.x_indexed])
    #y_tensor = torch.as_tensor(ex.x_indexed)
    y_tensor = torch.as_tensor(ex.y_indexed)
    inp_lens_tensor = torch.as_tensor([ex.x_len()])
    input_output_map = create_input_output_map(ex.x_indexed, input_indexer, output_indexer)

    (enc_output, enc_context_mask, enc_hidden, enc_bi_hidden) = encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc)
    dec_hidden = enc_hidden
    context_vec = enc_bi_hidden[0]

    dec_input = torch.as_tensor([[SOS_token]])
    y_temp = []

    # Initialize loss
    loss = 0
    SOS_token = 1
    EOS_token = 2
    criterion = torch.nn.NLLLoss()

    teacher_forcing = True if random.random() <= teacher_forcing_ratio else False

    for y_ex in range(len(y_tensor)):
        dec_output, dec_input, dec_input_val, dec_hidden, context_vec, for_inference = decode_copy_attn(dec_input, dec_hidden, model_output_emb, model_dec, context_vec, enc_output, beam_length, input_output_map)
        loss += criterion(dec_output, y_tensor[y_ex].unsqueeze(0))
        y_temp.append(dec_input.item())
        #teacher_forcing = True if random.random() <= teacher_forcing_ratio else False
        if dec_input == EOS_token:
            break
        if teacher_forcing:
            dec_input = y_tensor[y_ex].unsqueeze(0).unsqueeze(0)

    loss.backward()
    input_emb_optimizer.step()
    output_emb_optimizer.step()
    enc_optimizer.step()
    dec_optimizer.step()


    return loss




def train_iters(train_data, dev_data, test_data, epochs, input_indexer, output_indexer, args, beam_length, out, word_vectors_in, word_vectors_out):
    # Write results to file for readability
    f_name = args.results_path+".txt"
    f = open(f_name, 'w')

    # Create encoder, decoder and embedding layers
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    model_output_emb = EmbeddingLayer(args.input_dim, len(output_indexer), args.emb_dropout)

    # Convert word_vectors to x_tensor
    word_vectors_in = torch.FloatTensor(word_vectors_in)
    word_vectors_out = torch.FloatTensor(word_vectors_out)

    # Create tuples
    indexers = (input_indexer, output_indexer)

    # Initialize embedding layer with glove read_word_embeddings
    model_input_emb.add_pretrained(word_vectors_in)
    model_output_emb.add_pretrained(word_vectors_out)

    # Select which decoder model based on attention vs. non-attention
    if args.copy == 'Y':
        print("Model with copy")
        model_dec = CopyAttnRNNDecoder(args.input_dim + args.hidden_size*2, args.hidden_size * 2, args.hidden_size, out, args.rnn_dropout)
        #model_dec = CopyMLPAttnRNNDecoder(args.input_dim + args.hidden_size*2, args.hidden_size * 2, args.hidden_size, out, args.rnn_dropout) #CHANGE
        train_model = train_copy_attn
    elif args.attn == 'N' and args.copy == 'N':
        print("Model with no attention")
        model_dec = RNNDecoder(args.input_dim, args.hidden_size, out, dropout = 0 )
        train_model = train_model_encdec
    elif args.attn == 'Y' and args.copy == 'N':
        print("Model with attention")
        model_dec = AttnRNNDecoder(args.input_dim + args.hidden_size*2, args.hidden_size * 2, args.hidden_size, out, args.rnn_dropout)  #CHANGE
        #model_dec = AttnRNNDecoder(args.input_dim, args.hidden_size * 2, args.hidden_size, out, args.rnn_dropout)
        train_model = train_attn



    #Initialize optimizers
    input_emb_optimizer = optim.Adam(model_input_emb.parameters(), lr=args.lr)
    enc_optimizer = optim.Adam(model_enc.parameters(), lr=args.lr)
    output_emb_optimizer = optim.Adam(model_output_emb.parameters(), lr=args.lr)
    dec_optimizer = optim.Adam(model_dec.parameters(), lr=args.lr)

    teacher_forcing_ratio_orig = args.teacher_forcing_ratio

    count = 0.0
    count_ss = 0.0
    teacher_forcing_ratio = 1.0
    for i in range(epochs):
        print("epoch: ", i)
        teacher_forcing_ratio = teacher_forcing_ratio_orig**i
        for ex in train_data:        
            if args.copy == 'Y':
                inc_loss = train_model(ex, model_input_emb, model_enc, model_output_emb, model_dec, enc_optimizer, dec_optimizer, input_emb_optimizer, output_emb_optimizer, beam_length, teacher_forcing_ratio, indexers)
            else:
                inc_loss = train_model(ex, model_input_emb, model_enc, model_output_emb, model_dec, enc_optimizer, dec_optimizer, input_emb_optimizer, output_emb_optimizer, beam_length, args.teacher_forcing_ratio)

            if count %25 ==0:
                print ("loss: ", inc_loss)
            count = count + 1

        if args.copy == 'Y':
            decoder = Seq2SeqSemanticParserAttnCopy(model_input_emb, model_enc, model_output_emb, model_dec, args, indexers)
        elif args.attn == 'N':
            decoder = Seq2SeqSemanticParser(model_input_emb, model_enc, model_output_emb, model_dec, args, output_indexer)
        elif args.attn == 'Y':
            decoder = Seq2SeqSemanticParserAttn(model_input_emb, model_enc, model_output_emb, model_dec, args, output_indexer)

        print("Begin Evaluation for epoch: ", i)
        denotation_acc = evaluate(dev_data, decoder)
        f.write(str(denotation_acc)[0:6] + '\n')
        if args.copy == 'Y' and denotation_acc >=0.74:
            print("RETURNING DECODER WITH ACCURACY: ", denotation_acc)
            torch.save(model_input_emb.state_dict(), args.results_path+'_'+'copy_input_emb_'+ str(denotation_acc)[0:6]+ '.pth.tar')
            torch.save(model_output_emb.state_dict(), args.results_path+'_'+'copy_output_emb_'+ str(denotation_acc)[0:6]+'.pth.tar')
            torch.save(model_enc.state_dict(), args.results_path+'_'+'copy_enc_'+ str(denotation_acc)[0:6]+'.pth.tar')
            torch.save(model_dec.state_dict(), args.results_path+'_'+'copy_dec_'+ str(denotation_acc)[0:6]+'.pth.tar')
            out_name = "geo_test_output_"+str(args.teacher_forcing_ratio)[0:6]+'_'+str(denotation_acc)[0:6]+".tsv"
            evaluate(test_data, decoder, outfile=out_name)
        elif args.attn == 'Y' and args.copy == 'N' and denotation_acc >= 0.61:
            torch.save(model_input_emb.state_dict(), args.results_path+'_'+'attn_input_emb_'+ str(denotation_acc)[0:6]+'.pth.tar')
            torch.save(model_output_emb.state_dict(), args.results_path+'_'+'attn_output_emb_'+ str(denotation_acc)[0:6]+'.pth.tar')
            torch.save(model_enc.state_dict(), args.results_path+'_'+'attn_enc_'+ str(denotation_acc)[0:6]+'.pth.tar')
            torch.save(model_dec.state_dict(), args.results_path+'_'+'attn_dec_'+ str(denotation_acc)[0:6]+'.pth.tar')
        elif args.attn == 'N' and args.copy == 'N' and denotation_acc >= 0.15:
            torch.save(model_input_emb.state_dict(), args.results_path+'_'+'base_input_emb_'+ str(denotation_acc)[0:6]+'.pth.tar')
            torch.save(model_output_emb.state_dict(), args.results_path+'_'+'base_output_emb_'+ str(denotation_acc)[0:6]+'.pth.tar')
            torch.save(model_enc.state_dict(), args.results_path+'_'+'base_enc_'+ str(denotation_acc)[0:6]+'.pth.tar')
            torch.save(model_dec.state_dict(), args.results_path+'_'+'base_dec_'+ str(denotation_acc)[0:6]+'.pth.tar')


    return decoder

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
        #if i % example_freq == 0:
        #    print('Example %d' % i)
        #    print('  x      = "%s"' % ex.x)
        #    print('  y_tok  = "%s"' % ex.y_tok)
        #    print('  y_pred = "%s"' % selected_derivs[i].y_toks)
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
    print("Denotation abby: %.3f"% (denotation_acc(num_denotation_match, len(test_data))))
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()
    return denotation_acc(num_denotation_match, len(test_data))


def render_ratio(numer, denom):
    return "%i / %i = %.3f" % (numer, denom, float(numer)/denom)

def denotation_acc(numer, denom):
    return float(numer)/denom
