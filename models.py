import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var

import numpy as np


# Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
# Works for both non-batched and batched inputs
class EmbeddingLayer(nn.Module):
    # Parameters: dimension of the word embeddings, number of words, and the dropout rate to apply
    # (0.2 is often a reasonable value)
    def __init__(self, input_dim, full_dict_size, embedding_dropout_rate):
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    def add_pretrained(self, word_vectors):
        self.word_embedding = self.word_embedding.from_pretrained(word_vectors, freeze = False)

    # Takes either a non-batched input [sent len x input_dim] or a batched input
    # [batch size x sent len x input dim]
    def forward(self, input):
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


# One-layer RNN encoder for batched inputs -- handles multiple sentences at once. You're free to call it with a
# leading dimension of 1 (batch size 1) but it does expect this dimension.
class RNNEncoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN,
    # and a boolean flag for whether or not we're using a bidirectional encoder
    def __init__(self, input_size, hidden_size, dropout, bidirect):
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                               dropout=dropout, bidirectional=self.bidirect)
        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    # embedded_words should be a [batch size x sent len x input dim] tensor
    # input_lens is a tensor containing the length of each input sentence
    # Returns output (each word's representation), context_mask (a mask of 0s and 1s
    # reflecting where the model's output should be considered), and h_t, a *tuple* containing
    # the final states h and c from the encoder for each sentence.
    def forward(self, embedded_words, input_lens):
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        max_length = input_lens.data[0].item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            bi_ht = (h_, c_)
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        return (output, context_mask, h_t, bi_ht)

# One-layer RNN decoder for batched inputs -- handles multiple sentences at once. You're free to call it with a
# leading dimension of 1 (batch size 1) but it does expect this dimension.
class RNNDecoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN,
    # and a boolean flag for whether or not we're using a bidirectional encoder
    def __init__(self, input_size, hidden_size, out, dropout):
        super(RNNDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out = out
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,dropout=dropout)
        self.ff = nn.Linear(hidden_size, out)
        self.softmax = nn.LogSoftmax(dim = 1)

        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        nn.init.xavier_uniform_(self.ff.weight)


    # Like the encoder, the embedding layer is outside the decoder. So, it is assumed here
    # that the input is a word embedding
    def forward(self, input, hidden):
        #print("input: ", input)
        output, (h,c) = self.rnn(input, hidden)
        output = self.ff(h[0])
        return self.softmax(output), (h,c)

# One-layer RNN decoder for batched inputs -- handles multiple sentences at once. You're free to call it with a
# leading dimension of 1 (batch size 1) but it does expect this dimension.
class AttnRNNDecoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN,
    # and a boolean flag for whether or not we're using a bidirectional encoder
    def __init__(self, input_size, hidden_size_enc, hidden_size_dec, out, dropout):
        super(AttnRNNDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_dec = hidden_size_dec
        self.out = out

        self.rnn = nn.LSTM(input_size, hidden_size_dec, num_layers=1, batch_first=True,dropout=dropout)
        self.attn = nn.Linear(hidden_size_enc, hidden_size_dec)
        self.attn_hid = nn.Linear(hidden_size_dec + hidden_size_enc, hidden_size_dec)
        self.ff = nn.Linear(hidden_size_dec, out)
        self.softmax = nn.LogSoftmax(dim = 1)

        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        nn.init.xavier_uniform_(self.attn.weight)
        nn.init.xavier_uniform_(self.attn_hid.weight)
        nn.init.xavier_uniform_(self.ff.weight)


    # Like the encoder, the embedding layer is outside the decoder. So, it is assumed here
    # that the input is a word embedding
    # Input is [embed(e_{t-1}; c_{t-1}), h_{t-1}]
    def forward(self, input, hidden, encoder_outputs):
        # Show the example to the decoder and get hidden state
        output, (h,c) = self.rnn(input, hidden)
        h_bar = h[0]

        encoder_outputs = encoder_outputs.squeeze()
        # Calculate eij
        attn_weight = self.attn(encoder_outputs).squeeze()
        attn_weight = torch.transpose(attn_weight, 0, 1)
        attn_energy = torch.matmul(h_bar, attn_weight)
        attn_score = F.softmax(attn_energy, dim = 1)

        # Calculate the context vector, ci
        context = torch.matmul(attn_score, encoder_outputs)

        # Concatenate the context vector ci and the hidden state hbar
        attn_hid_combined = torch.cat((context, h_bar), 1)
        attn_hid_transformed = self.attn_hid(attn_hid_combined)
        out = self.ff(attn_hid_transformed)
        return self.softmax(out), (h,c), context

# One-layer RNN decoder for batched inputs -- handles multiple sentences at once. You're free to call it with a
# leading dimension of 1 (batch size 1) but it does expect this dimension.
class CopyAttnRNNDecoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN,
    # and a boolean flag for whether or not we're using a bidirectional encoder
    def __init__(self, input_size, hidden_size_enc, hidden_size_dec, out, dropout):
        super(CopyAttnRNNDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_dec = hidden_size_dec
        self.out = out

        self.rnn = nn.LSTM(input_size, hidden_size_dec, num_layers=1, batch_first=True,dropout=dropout)

        # To calculate Pvocab
        self.attn = nn.Linear(hidden_size_enc, hidden_size_dec)
        self.attn_hid = nn.Linear(hidden_size_dec + hidden_size_enc, hidden_size_dec)
        self.ff = nn.Linear(hidden_size_dec, out)
        self.softmax = nn.Softmax(dim = 1)

        # To calculate pgen
        self.Wh = nn.Linear(hidden_size_enc, 1)
        self.Ws = nn.Linear(hidden_size_dec, 1)
        self.Wx = nn.Linear(input_size,1)
        self.bptr = nn.Linear(1,1)
        self.sigmoid = nn.Sigmoid()


        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        nn.init.xavier_uniform_(self.attn.weight)
        nn.init.xavier_uniform_(self.attn_hid.weight)
        nn.init.xavier_uniform_(self.ff.weight)
        nn.init.xavier_uniform_(self.Wh.weight)
        nn.init.xavier_uniform_(self.Ws.weight)
        nn.init.xavier_uniform_(self.Wx.weight)
        nn.init.xavier_uniform_(self.bptr.weight)


    # Like the encoder, the embedding layer is outside the decoder. So, it is assumed here
    # that the input is a word embedding
    # Input is [embed(e_{t-1}; c_{t-1}), h_{t-1}]
    def forward(self, input, hidden, encoder_outputs, input_output_map):
        # Show the example to the decoder and get hidden state
        output, (h,c) = self.rnn(input, hidden)
        h_bar = h[0]

        encoder_outputs = encoder_outputs.squeeze()
        # Calculate eij
        attn_weight = self.attn(encoder_outputs).squeeze()
        attn_weight = torch.transpose(attn_weight, 0, 1)
        attn_energy = torch.matmul(h_bar, attn_weight)
        attn_score = F.softmax(attn_energy, dim = 1)

        # Calculate the context vector, ci
        context = torch.matmul(attn_score, encoder_outputs)
        #print("attn_score: ", attn_score.size())

        # Concatenate the context vector ci and the hidden state hbar
        # Calculate p_vocab with two hidden layers
        attn_hid_combined = torch.cat((context, h_bar), 1)
        attn_hid_transformed = self.attn_hid(attn_hid_combined)
        out = self.ff(attn_hid_transformed)
        p_vocab = self.softmax(out)
        #print("context: ", context)

        # Calculate p_gen
        #constant = torch.as_tensor([1.0])
        #p_gen = self.sigmoid(self.Wh(context) + self.Ws(h_bar) + self.Wx(input)) # CHANGE
        p_gen = self.sigmoid(self.Wh(context) + self.Ws(h_bar))
        p_gen = p_gen.squeeze(0)

        # Calculate P(W)
        ai = torch.matmul(attn_score, input_output_map)
        #print("size of ai: ", ai.size())
        #print("size of p_vocab: ", p_vocab.size())
        Pw = p_gen * p_vocab + (1-p_gen) * ai
        Pw = torch.log(Pw)
        #print("Pw: ", Pw)
        #print("size of Pw: ", Pw.size())
        for_inference = torch.cat((p_gen*p_vocab, (1-p_gen)*attn_score), 1)
        #print("for_inference: ", for_inference)
        return Pw, (h,c), context, for_inference
