

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer

#from ..utils import visualize_masked_embedding


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
    

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return x
    
class BIOTEncoder(nn.Module):
    def __init__(
        self,
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=23,
        n_fft=200,
        hop_length=100,
        mask_ratio=0.15,
        **kwargs
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mask_ratio = mask_ratio

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,  # dropout right after self-attention layer
            attn_dropout=0.2,  # dropout post-attention
        )
        self.positional_encoding = PositionalEncoding(emb_size)

        # channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(n_channels, 256)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)
    
    def random_masking(self, x):
        """
        Azzeramento casuale globale di valori in un tensore [B, C, T],
        secondo una percentuale `mask_ratio` dei valori totali.
        """
        # Crea una maschera booleana con valori True dove si vuole azzerare
        mask = torch.rand_like(x) < self.mask_ratio  # stessa shape di x
        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask
    
    def set_mask_ratio(self, mask_ratio):
        """
        Imposta il rapporto di mascheramento per la maschera casuale.
        """
        self.mask_ratio = mask_ratio



    def forward(self, x, n_channel_offset=0,  verbose=False):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
     
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
   
           
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
    
            batch_size, ts, _ = channel_spec_emb.shape
            # (batch_size, ts, emb)
            
            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )

   
            # (batch_size, ts, emb)

            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)
            #print(f" emb after positional_encoding -> {channel_emb.shape}")
            
            emb_seq.append(channel_emb)
            
        
      
        emb = torch.cat(emb_seq, dim=1) # (batch_size, n_channels * ts, emb)

             
        # random masking
        masked_emb = emb.clone() 
        masked_emb, mask = self.random_masking(masked_emb)

        out_biot = self.transformer(masked_emb) # (batch_size, n_channels * ts, emb)
        
        
        

        return emb, masked_emb, out_biot,  mask