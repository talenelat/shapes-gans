'''
Adapted from: 
https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
'''

import argparse
import random
import os
import numpy as np
import pickle 
import nltk
import pandas as pd 

import torch  
import torch.nn as nn 
import torchvision.models as models 
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.datasets as dset 
import torchvision.transforms as transforms 
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, vocab_size, nc, ndf, image_size):
        super(EncoderCNN, self).__init__()
        
        self.nc = nc
        self.ndf = ndf
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.image_size = image_size

        # Input: (nc x image_size x image_size)
        
        self.enc_cv1 = nn.Conv2d(in_channels=self.nc, out_channels=self.ndf, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        # Output: (ndf x image_size x image_size)
        
        self.enc_cv2 = nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc_bn1 = nn.BatchNorm2d(self.ndf * 2)
        self.enc_lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        # Output: (2*ndf x image_size // 2 x image_size // 2)

        self.enc_cv3 = nn.Conv2d(in_channels=self.ndf * 2, out_channels=self.ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc_bn2 = nn.BatchNorm2d(self.ndf * 4)
        self.enc_lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        # Output: (4*ndf x image_size // 4 x image_size // 4)
        
        self.enc_lin = nn.Linear(4*self.ndf * self.image_size // 4 * self.image_size // 4, out_features=self.embed_size, bias=False)
        self.enc_bn3 = nn.BatchNorm1d(embed_size, momentum=0.01)
        # Output: (embed_size)            

    def forward(self, image):
        image = image.reshape(-1, self.nc, self.image_size, self.image_size)
        e1 = self.enc_lrelu1(self.enc_cv1(image))
        e2 = self.enc_lrelu2(self.enc_bn1(self.enc_cv2(e1)))
        e3 = self.enc_lrelu3(self.enc_bn2(self.enc_cv3(e2)))
        e3 = e3.reshape(e3.size(0), -1)
        e4 = self.enc_bn3(self.enc_lin(e3))
        return e4


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=14):
        super(DecoderRNN, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length

        self.dec_emb = nn.Embedding(self.vocab_size, self.embed_size)
        self.dec_lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.dec_lin = nn.Linear(self.hidden_size, self.vocab_size)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.dec_emb(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.dec_lstm(packed)
        outputs = self.dec_lin(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        caption_emb = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.dec_lstm(inputs, states)
            outputs = self.dec_lin(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            caption_emb.append(predicted)
            inputs = self.dec_emb(predicted)
            inputs = inputs.unsqueeze(1)
        caption_emb = torch.stack(caption_emb, 1)                
        # Output: (batch_size, max_seq_length)
        return caption_emb