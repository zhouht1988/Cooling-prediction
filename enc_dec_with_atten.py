# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:16:28 2023

@author: HAIZHO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import random


class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, dropout, model_type='GRU'):
    
        super().__init__()
        
        if model_type == 'GRU':
            self.rnn = nn.GRU(input_dim, enc_hid_dim, bidirectional = False, batch_first=True)  
        else:
            self.rnn = nn.LSTM(input_dim, enc_hid_dim, bidirectional = False, batch_first=True) 
        
        #self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        outputs, hidden = self.rnn(src)
        
        #initial decoder hidden is final hidden state of the forwards (and backwards)
        #  encoder RNNs fed through a linear layer
        # if bidirectional == True, then
        # hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        hidden = torch.tanh(self.fc(hidden[-1,:,:]))
        
        return outputs, hidden
    
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)
    
class Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, dropout, attention, model_type='GRU'):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        if model_type == 'GRU':
            self.rnn = nn.GRU((enc_hid_dim) + output_dim, dec_hid_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM((enc_hid_dim) + output_dim, dec_hid_dim, batch_first=True)  
        
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        
        input = input.unsqueeze(1)
        
        #input = [1, batch size]
        
        a = self.attention(hidden, encoder_outputs)
                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        rnn_input = torch.cat((input, weighted), dim = 2)
                    
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #this also means that output == hidden
        assert (output == hidden.permute(1, 0, 2)).all()
        
        ###connected with hidden states from encoder
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        output =  self.dropout(output)
        input = input.squeeze(1)
        prediction = self.fc_out(torch.cat((output, weighted, input), dim = 1))
        
        return prediction, hidden.squeeze(0)
    
## Seq2Seq model

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size)
        
        #encoder_outputs is all hidden states of the input sequence, back (and forwards)
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        batch_size = trg.shape[0]
        
        input = src[:,-1,:]        
        
        for t in range(0, trg_len):
            
            #insert input, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            #input = trg[:, t, :]
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            input = trg[:, t, :] if teacher_force else output
            
        return outputs
    
def cooling_model(input_dim, output_dim, device, enc_hid_dim=20, dec_hid_dim=20,
                  dropout=0.1, model_type='GRU'):
    """

    Parameters
    ----------
    input_dim : int
        The number of expected features in the input `x`.
    output_dim : int
        The number of expected features in the output `y`.
    device : torch.device
        'Cuda' or 'CPU'
    enc_hid_dim : int, optional
        The number of features in the hidden state of the encoder. The default is 20.
    dec_hid_dim : int, optional
        The number of features in the hidden state of the decoder. The default is 20.
    dropout : float, optional
        Dropout rate in the dropout layer. The default is 0.1.
    model_type : 'string', optional
        Unit type. The default is 'GRU'.

    Returns
    -------
    model : torch.nn
        cooling model.

    """
    attn = Attention(enc_hid_dim, dec_hid_dim)
    enc = Encoder(input_dim, enc_hid_dim, dec_hid_dim, dropout, model_type=model_type)
    dec = Decoder(output_dim, enc_hid_dim, dec_hid_dim, dropout, attn, model_type=model_type)
    model = Seq2Seq(enc, dec, device)
    return model