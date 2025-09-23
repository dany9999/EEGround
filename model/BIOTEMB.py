

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
        self.dropout = nn.Dropout(p=0.1)

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
        return self.dropout(x)

class BIOTEncoder(nn.Module):
    def __init__(
        self,
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=18,
        n_fft=250,
        hop_length=125,
        mask_ratio=0.15,
        pretraining=False,
        **kwargs
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mask_ratio = mask_ratio
        self.pretraining = pretraining


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
        window = torch.hann_window(self.n_fft, device=sample.device)
        spectral = torch.stft(
            input=sample.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)

    # def stft(self, sample):
    #     spectral = torch.stft( 
    #         input = sample.squeeze(1),
    #         n_fft = self.n_fft,
    #         hop_length = self.hop_length,
    #         center = False,
    #         onesided = True,
    #         return_complex = True,
    #     )
    #     return torch.abs(spectral)
    
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

    # def random_masking(self, x):
    #     """
    #     Maschera il `mask_ratio` delle righe nel tensore `x` di shape (B, Seq, D),
    #     azzerando interamente le embedding di alcune righe.
    #     """
    #     B, Seq, D = x.shape
    #     num_mask = int(Seq * self.mask_ratio)

    #     # Stessa maschera per ogni elemento nel batch
    #     mask_indices = torch.randperm(Seq)[:num_mask]
    #     mask = torch.zeros(B, Seq, dtype=torch.bool, device=x.device)
    #     mask[:, mask_indices] = True

    #     x_masked = x.clone()
    #     x_masked[mask] = 0.0

    #     return x_masked, mask
    
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
        
        if self.pretraining:
            masked_emb = emb.clone() 
            masked_emb, mask = self.random_masking(emb)
            out_biot = self.transformer(masked_emb) # (batch_size, n_channels * ts, emb)
        else:
            masked_emb = emb
            mask = None
            out_biot = self.transformer(masked_emb).mean(dim=1) # (batch_size, n_channels * ts, emb)


        
        return emb, masked_emb, out_biot,  mask