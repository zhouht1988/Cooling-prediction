# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 19:25:10 2023

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
            self.rnn = nn.GRU(input_dim, enc_hid_dim, bidirectional = False, batch_first=True)  # 改为单向
        else:
            self.rnn = nn.LSTM(input_dim, enc_hid_dim, bidirectional = False, batch_first=True)  # 改为单向
        
        self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        outputs, hidden = self.rnn(src)
                
        hidden = torch.tanh(self.fc(hidden[-1,:,:]))
        
        return outputs, hidden
    
class Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, dropout, model_type='GRU'):
        super().__init__()

        self.output_dim = output_dim
        
        
        if model_type == 'GRU':
            self.rnn = nn.GRU(output_dim, dec_hid_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(output_dim, dec_hid_dim, batch_first=True)
        
        self.fc_out = nn.Linear(dec_hid_dim + output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        
        input = input.unsqueeze(1)
    
            
        output, hidden = self.rnn(input, hidden.unsqueeze(0))
        
        output = output.squeeze(1)
        output =  self.dropout(output)
        input = input.squeeze(1)
        prediction = self.fc_out(torch.cat((output, input), dim = 1))
        
        return prediction, hidden.squeeze(0)


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
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
                
        batch_size = trg.shape[0]
        
        input = src[:,-1,:]        
        
        for t in range(0, trg_len):
            
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            input = trg[:, t, :]
            
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
    enc = Encoder(input_dim, enc_hid_dim, dec_hid_dim, dropout, model_type=model_type)
    dec = Decoder(output_dim, enc_hid_dim, dec_hid_dim, dropout, model_type=model_type)
    model = Seq2Seq(enc, dec, device)
    return model