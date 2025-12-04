

# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from linear_attention_transformer import LinearAttentionTransformer

# #from ..utils import visualize_masked_embedding


# class PatchFrequencyEmbedding(nn.Module):
#     def __init__(self, emb_size=256, n_freq=101):
#         super().__init__()
#         self.projection = nn.Linear(n_freq, emb_size)

#     def forward(self, x):
#         """
#         x: (batch, freq, time)
#         out: (batch, time, emb_size)
#         """
#         x = x.permute(0, 2, 1)
#         x = self.projection(x)
#         return x


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, max_len: int = 1000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=0.1)

#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer("pe", pe)

#     def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
#         """
#         Args:
#             x: `embeddings`, shape (batch, max_len, d_model)
#         Returns:
#             `encoder input`, shape (batch, max_len, d_model)
#         """
#         x = x + self.pe[:, : x.size(1)]
#         return self.dropout(x)

# class BIOTEncoder(nn.Module):
#     def __init__(
#         self,
#         emb_size=256,
#         heads=8,
#         depth=4,
#         n_channels=18,
#         n_fft=250,
#         hop_length=125,
#         mask_ratio=0.15,
#         pretraining=False,
#         **kwargs
#     ):
#         super().__init__()

#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.mask_ratio = mask_ratio
#         self.pretraining = pretraining


#         self.patch_embedding = PatchFrequencyEmbedding(
#             emb_size=emb_size, n_freq=self.n_fft // 2 + 1
#         )
#         self.transformer = LinearAttentionTransformer(
#             dim=emb_size,
#             heads=heads,
#             depth=depth,
#             max_seq_len=1024,
#             attn_layer_dropout=0.2,  # dropout right after self-attention layer
#             attn_dropout=0.2,  # dropout post-attention
#         )
#         self.positional_encoding = PositionalEncoding(emb_size)

#         # channel token, N_channels >= your actual channels
#         self.channel_tokens = nn.Embedding(n_channels, 256)
#         self.index = nn.Parameter(
#             torch.LongTensor(range(n_channels)), requires_grad=False
#         )

#     def stft(self, sample):
#         window = torch.hann_window(self.n_fft, device=sample.device)
#         spectral = torch.stft(
#             input=sample.squeeze(1),
#             n_fft=self.n_fft,
#             hop_length=self.hop_length,
#             window=window,
#             center=False,
#             onesided=True,
#             return_complex=True,
#         )
#         return torch.abs(spectral)

#     # def stft(self, sample):
#     #     spectral = torch.stft( 
#     #         input = sample.squeeze(1),
#     #         n_fft = self.n_fft,
#     #         hop_length = self.hop_length,
#     #         center = False,
#     #         onesided = True,
#     #         return_complex = True,
#     #     )
#     #     return torch.abs(spectral)
    
#     def random_masking(self, x):
#         """
#         Azzeramento casuale globale di valori in un tensore [B, C, T],
#         secondo una percentuale `mask_ratio` dei valori totali.
#         """
#         # Crea una maschera booleana con valori True dove si vuole azzerare
#         mask = torch.rand_like(x) < self.mask_ratio  # stessa shape di x
#         x_masked = x.clone()
#         x_masked[mask] = 0.0
#         return x_masked, mask

#     # def random_masking(self, x):
#     #     """
#     #     Maschera il `mask_ratio` delle righe nel tensore `x` di shape (B, Seq, D),
#     #     azzerando interamente le embedding di alcune righe.
#     #     """
#     #     B, Seq, D = x.shape
#     #     num_mask = int(Seq * self.mask_ratio)

#     #     # Stessa maschera per ogni elemento nel batch
#     #     mask_indices = torch.randperm(Seq)[:num_mask]
#     #     mask = torch.zeros(B, Seq, dtype=torch.bool, device=x.device)
#     #     mask[:, mask_indices] = True

#     #     x_masked = x.clone()
#     #     x_masked[mask] = 0.0

#     #     return x_masked, mask
    
#     def set_mask_ratio(self, mask_ratio):
#         """
#         Imposta il rapporto di mascheramento per la maschera casuale.
#         """
#         self.mask_ratio = mask_ratio



#     def forward(self, x, n_channel_offset=0,  verbose=False):
#         """
#         x: [batch_size, channel, ts]
#         output: [batch_size, emb_size]
#         """
     
#         emb_seq = []
#         for i in range(x.shape[1]):
#             channel_spec_emb = self.stft(x[:, i : i + 1, :])
   
           
#             channel_spec_emb = self.patch_embedding(channel_spec_emb)
    
#             batch_size, ts, _ = channel_spec_emb.shape
#             # (batch_size, ts, emb)
            
#             channel_token_emb = (
#                 self.channel_tokens(self.index[i + n_channel_offset])
#                 .unsqueeze(0)
#                 .unsqueeze(0)
#                 .repeat(batch_size, ts, 1)
#             )

   
#             # (batch_size, ts, emb)

#             channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)
#             #print(f" emb after positional_encoding -> {channel_emb.shape}")
            
#             emb_seq.append(channel_emb)
            
        
      
#         emb = torch.cat(emb_seq, dim=1) # (batch_size, n_channels * ts, emb)

             
#         # random masking
        
#         if self.pretraining:
#             masked_emb = emb.clone() 
#             masked_emb, mask = self.random_masking(emb)
#             out_biot = self.transformer(masked_emb) # (batch_size, n_channels * ts, emb)
#         else:
#             masked_emb = emb
#             mask = None
#             out_biot = self.transformer(masked_emb) # (batch_size, n_channels * ts, emb)


        
#         return emb, masked_emb, out_biot,  mask

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from linear_attention_transformer import LinearAttentionTransformer


# class PatchFrequencyEmbedding(nn.Module):
#     def __init__(self, emb_size=256, n_freq=101):
#         super().__init__()
#         self.projection = nn.Linear(n_freq, emb_size)

#     def forward(self, x):
#         """
#         x: (batch, freq, time)
#         out: (batch, time, emb_size)
#         """
#         x = x.permute(0, 2, 1)   # → (B, T, F)
#         x = self.projection(x)   # → (B, T, D)
#         return x


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, max_len: int = 1000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=0.1)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         x = x + self.pe[:, : x.size(1)]
#         return self.dropout(x)



# class BIOTEncoder(nn.Module):
#     def __init__(
#         self,
#         emb_size=256,
#         heads=8,
#         depth=4,
#         n_channels=18,
#         n_fft=250,
#         hop_length=125,
#         mask_ratio=0.15,
#         pretraining=False,
#         **kwargs
#     ):
#         super().__init__()

#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.mask_ratio = mask_ratio
#         self.pretraining = pretraining

#         self.patch_embedding = PatchFrequencyEmbedding(
#             emb_size=emb_size, n_freq=self.n_fft // 2 + 1
#         )

#         self.transformer = LinearAttentionTransformer(
#             dim=emb_size,
#             heads=heads,
#             depth=depth,
#             max_seq_len=1024,
#             attn_layer_dropout=0.2,
#             attn_dropout=0.2,
#         )

#         self.positional_encoding = PositionalEncoding(emb_size)

#         # Embedding dei canali
#         self.channel_tokens = nn.Embedding(n_channels, emb_size)
#         self.index = nn.Parameter(
#             torch.LongTensor(range(n_channels)), requires_grad=False
#         )

#     def stft(self, sample):
#         window = torch.hann_window(self.n_fft, device=sample.device)
#         spectral = torch.stft(
#             input=sample.squeeze(1),
#             n_fft=self.n_fft,
#             hop_length=self.hop_length,
#             window=window,
#             center=False,
#             onesided=True,
#             return_complex=True,
#         )
#         return torch.abs(spectral)   # (B, F, T)


#     # ---------------------------------------------------------
#     #  MAE-style TOKEN MASKING (mask intere patch, non valori)
#     # ---------------------------------------------------------
#     def random_masking(self, x):
#         """
#         MAE token-wise masking.
#         x: [B, Seq, D]
#         Ritorna:
#             masked_x: [B, Seq, D]
#             mask: [B, Seq] boolean
#         """
#         B, Seq, D = x.shape
#         num_mask = int(self.mask_ratio * Seq)

#         # stessa maschera per tutto il batch (MAE originale)
#         idx = torch.randperm(Seq, device=x.device)[:num_mask]

#         mask = torch.zeros(B, Seq, dtype=torch.bool, device=x.device)
#         mask[:, idx] = True

#         x_masked = x.clone()
#         x_masked[mask] = 0.0

#         return x_masked, mask


#     def set_mask_ratio(self, mask_ratio):
#         self.mask_ratio = mask_ratio


#     def forward(self, x, n_channel_offset=0, verbose=False):
#         """
#         x: [B, C, T]
#         Returns:
#             emb:       [B, Seq, D]
#             masked:    [B, Seq, D]
#             out_biot:  [B, Seq, D]
#             mask:      [B, Seq]
#         """

#         emb_seq = []

#         print(f" Input x -> {x.shape}")

#         # ---------------------------------------------------------
#         # Per ogni canale → STFT → PatchEmbedding → Positional+Channel Token
#         # ---------------------------------------------------------
#         for i in range(x.shape[1]):
            

#             spec = self.stft(x[:, i : i + 1, :])      # (B, F, T)
#             print(f" STFT channel {i} -> {spec.shape}")
#             patch = self.patch_embedding(spec)        # (B, T, D)
#             print(f" Patch Embedding channel {i} -> {patch.shape}")
#             B, T, D = patch.shape

#             channel_token = (
#                 self.channel_tokens(self.index[i + n_channel_offset])
#                 .unsqueeze(0).unsqueeze(0)
#                 .repeat(B, T, 1)
#             )
#             print(f" Channel Token channel {i} -> {channel_token.shape}")

#             emb = patch + channel_token
#             emb = self.positional_encoding(emb)
#             print(f" Positional Encoding channel {i} -> {emb.shape}")

#             emb_seq.append(emb)


#         # ---------------------------------------------------------
#         # Cat di tutti i canali: Seq = C × T
#         # ---------------------------------------------------------
#         emb = torch.cat(emb_seq, dim=1)     # (B, Seq, D)
#         print(f" Concatenated Embedding -> {emb.shape}")

#         # ---------------------------------------------------------
#         #  MASKING MAE
#         # ---------------------------------------------------------
#         if self.pretraining:
#             masked_emb, mask = self.random_masking(emb)
#             out_biot = self.transformer(masked_emb)
#             print(f" Transformer output -> {out_biot.shape}")
#         else:
#             masked_emb = emb
#             mask = None
#             out_biot = self.transformer(emb)
           

#         return emb, masked_emb, out_biot, mask
    
# import matplotlib.pyplot as plt

# if __name__ == "__main__":
    
#         # -----------------------------
#     # PARAMETRI
#     # -----------------------------
#     B = 1        # un solo sample per visualizzazione
#     C = 18
#     T = 1000
#     mask_ratio = 0.75

#     # -----------------------------
#     # SEGNALE RANDOM
#     # -----------------------------
#     x = torch.randn(B, C, T)

#     # -----------------------------
#     # MODELLO
#     # -----------------------------
#     model = BIOTEncoder(
#         emb_size=256,
#         heads=8,
#         depth=4,
#         n_channels=C,
#         n_fft=250,
#         hop_length=125,
#         mask_ratio=mask_ratio,
#         pretraining=True
#     )

#     model.eval()

#     # -----------------------------
#     # FORWARD
#     # -----------------------------
#     with torch.no_grad():
#         emb, masked_emb, out_biot, mask = model(x)

#     # -----------------------------
#     # PRENDIAMO SOLO IL PRIMO SAMPLE
#     # -----------------------------
#     emb_np = emb[0].cpu().numpy()           # [Seq, D]
#     masked_np = masked_emb[0].cpu().numpy()
#     mask_np = mask[0].cpu().numpy()         # [Seq]

#     print("Shape embedding:", emb_np.shape)
#     print("Token mascherati:", mask_np.sum(), "/", mask_np.shape[0])

#     # -----------------------------
#     # CREIAMO UNA MATRICE COLORATA
#     # valori originali + blu sui mascherati
#     # -----------------------------
#     masked_vis = masked_np.copy()

#     # Forziamo i token mascherati a zero per visualizzazione
#     masked_vis[mask_np] = 0.0

#     # -----------------------------
#     # PLOT
#     # -----------------------------
#     plt.figure(figsize=(16, 6))

#     plt.subplot(1, 2, 1)
#     plt.title("Embedding PRIMA del masking")
#     plt.imshow(emb_np, aspect="auto")
#     plt.colorbar()
#     plt.xlabel("Token (Seq)")
#     plt.ylabel("Feature (D)")

#     plt.subplot(1, 2, 2)
#     plt.title("Embedding DOPO il masking (blu = 0)")
#     plt.imshow(masked_vis, aspect="auto")
#     plt.colorbar()
#     plt.xlabel("Token (Seq)")
#     plt.ylabel("Feature (D)")

#     plt.tight_layout()
#     plt.show()



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)   # → (B, T, F)
        x = self.projection(x)   # → (B, T, D)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
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
        mask_ratio=0.3,
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
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )

        self.positional_encoding = PositionalEncoding(emb_size)

        # Embedding dei canali
        self.channel_tokens = nn.Embedding(n_channels, emb_size)
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
        return torch.abs(spectral)   # (B, F, T)


    # ---------------------------------------------------------
    #  MAE-style TOKEN MASKING (mask intere patch, non valori)
    # ---------------------------------------------------------
    def time_masking_stft(self, spec):
        """
        spec: (B, F, T)
        Maschera colonne temporali (asse T)
        """
        B, F, T = spec.shape
        num_mask = int(self.mask_ratio * T)

        idx = torch.randperm(T, device=spec.device)[:num_mask]

        mask = torch.zeros(B, T, dtype=torch.bool, device=spec.device)
        mask[:, idx] = True

        spec_masked = spec.clone()
        spec_masked[:, :, mask[0]] = 0.0   # stesse colonne per tutto il batch

        return spec_masked, mask


    def set_mask_ratio(self, mask_ratio):
        self.mask_ratio = mask_ratio


    def forward(self, x, n_channel_offset=0):
        """
        x: [B, C, T]
        """

        emb_clean_seq = []
        emb_masked_seq = []
        time_masks = []

        for i in range(x.shape[1]):

            # -------- STFT PULITA --------
            spec_clean = self.stft(x[:, i:i+1, :])   # (B, F, T)

            # -------- STFT MASCHERATA NEL TEMPO --------
            if self.pretraining:
                spec_masked, tmask = self.time_masking_stft(spec_clean)
                time_masks.append(tmask)   # (B, T)
            else:
                spec_masked = spec_clean

            # -------- Patch embedding --------
            patch_clean = self.patch_embedding(spec_clean)     # (B, T, D)
            patch_masked = self.patch_embedding(spec_masked)

            batch_size, ts, _ = patch_clean.shape

            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )

            emb_clean = self.positional_encoding(patch_clean + channel_token_emb)
            emb_masked = self.positional_encoding(patch_masked + channel_token_emb)

            emb_clean_seq.append(emb_clean)
            emb_masked_seq.append(emb_masked)

        # -------- CONCATENAZIONE FINALE --------
        emb_clean = torch.cat(emb_clean_seq, dim=1)   # (B, 126, 256)
        emb_masked = torch.cat(emb_masked_seq, dim=1)

        # -------- TRANSFORMER --------
        out = self.transformer(emb_masked)
        print(f" BIOTEncoder output -> {out.shape}")

        if self.pretraining:
            return emb_clean, emb_masked, out, time_masks
        else:
            return out
           

if __name__ == "__main__":
    B, C, T = 1, 18, 1000
    x = torch.randn(B, C, T)

    enc = BIOTEncoder(
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=C,
        n_fft=250,
        hop_length=125,
        mask_ratio=0.30,
        pretraining=True
    )

    with torch.no_grad():
        emb_clean, emb_masked, out, time_masks = enc(x)

    print("emb_clean:",  emb_clean.shape)   # (B, 126, 256)
    print("emb_masked:", emb_masked.shape)  # (B, 126, 256)
    print("out:",        out.shape)         # (B, 126, 256)
    print("len(time_masks):", len(time_masks))  # = C
    print("time_masks[0].shape:", time_masks[0].shape)  # (B, T=7)
      