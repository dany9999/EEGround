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
    def time_masking_idx(self, T, device):
    
        num_mask = int(self.mask_ratio * T)
        idx = torch.randperm(T, device=device)[:num_mask]
        return idx

    def apply_time_mask_idx(self, spec, idx):
        """spec: (B,F,T)"""
        spec_masked = spec.clone()
        spec_masked[:, :, idx] = 0.0
        return spec_masked

    def forward(self, x, n_channel_offset=0):
        emb_clean_seq = []
        emb_masked_seq = []
        stft_clean_list = []
        
        time_masks = None

        # ---- 1) calcolo una volta T_patch e maschera condivisa
        # prendo il primo canale per conoscere T della STFT
        spec0 = self.stft(x[:, 0:1, :])         # (B,F,Tpatch)
        B, F, Tpatch = spec0.shape

        if self.pretraining:
            idx = self.time_masking_idx(Tpatch, device=spec0.device)  # (num_mask,)
            # mask booleana (B,Tpatch) solo per logging/plot
            time_masks = torch.zeros(B, Tpatch, dtype=torch.bool, device=spec0.device)
            time_masks[:, idx] = True

        # ---- 2) loop canali
        for i in range(x.shape[1]):
            spec_clean = self.stft(x[:, i:i+1, :])  # (B,F,Tpatch)
            stft_clean_list.append(spec_clean)

            patch_clean = self.patch_embedding(spec_clean)  # (B,Tpatch, D)
            batch_size, ts, D = patch_clean.shape

            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0).unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )

            emb_clean = self.positional_encoding(patch_clean + channel_token_emb)
            emb_clean_seq.append(emb_clean)

            if self.pretraining:
                spec_masked = self.apply_time_mask_idx(spec_clean, idx)
                
                patch_masked = self.patch_embedding(spec_masked)
                emb_masked = self.positional_encoding(patch_masked + channel_token_emb)
                emb_masked_seq.append(emb_masked)

        # ---- 3) stack stft
        stft_clean = torch.stack(stft_clean_list, dim=1)       # (B,C,F,Tpatch)
        stft_clean_cat = stft_clean.permute(0,2,1,3).flatten(2) # (B,F,C*Tpatch)
           

        # ---- 4) embedding + transformer
        emb_clean = torch.cat(emb_clean_seq, dim=1)  # (B, C*Tpatch, D)

        if self.pretraining:
            emb_masked = torch.cat(emb_masked_seq, dim=1)
            out = self.transformer(emb_masked)
            return out, stft_clean_cat, time_masks
        else:
            out = self.transformer(emb_clean)
            return out
           

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    torch.manual_seed(0)

    # -------------------------
    # Parametri test
    # -------------------------
    B, C, T = 1, 10, 1000
    device = "cpu"

    x = torch.randn(B, C, T, device=device)

    enc = BIOTEncoder(
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=C,
        n_fft=250,
        hop_length=125,
        mask_ratio=0.30,
        pretraining=True
    ).to(device)

    enc.eval()

    with torch.no_grad():
        out, stft_clean_cat, stft_masked_cat, time_masks = enc(x)

    # -------------------------
    # SHAPES
    # -------------------------
    B, F, CT = stft_clean_cat.shape
    Tpatch = time_masks.shape[1]

    print("STFT clean cat:", stft_clean_cat.shape)
    print("STFT masked cat:", stft_masked_cat.shape)
    print("time_masks:", time_masks.shape)
    print("Transformer out:", out.shape)
    print("C*Tpatch:", C * Tpatch)

    # -------------------------
    # GLOBAL CONCAT VISUALIZATION
    # -------------------------
    clean_cat = stft_clean_cat[0].cpu().numpy()    # (F, C*T)
    masked_cat = stft_masked_cat[0].cpu().numpy()
    diff_cat = masked_cat - clean_cat

    # costruiamo maschera estesa a C*T
    mask_t = time_masks[0].cpu().numpy()           # (Tpatch,)
    mask_cat = np.tile(mask_t, C)                  # (C*Tpatch,)

    plt.figure(figsize=(18, 10))

    plt.subplot(4, 1, 1)
    plt.title("STFT CONCAT CLEAN  (F x C*T)")
    plt.imshow(clean_cat, aspect="auto", origin="lower")
    plt.colorbar()

    plt.subplot(4, 1, 2)
    plt.title("STFT CONCAT MASKED")
    plt.imshow(masked_cat, aspect="auto", origin="lower")
    plt.colorbar()

    plt.subplot(4, 1, 3)
    vmax = max(abs(diff_cat.min()), abs(diff_cat.max()))
    plt.title("STFT CONCAT DIFF (masked - clean)")
    plt.imshow(diff_cat, aspect="auto", origin="lower",
               cmap="bwr", vmin=-vmax, vmax=vmax)
    plt.colorbar()

    plt.subplot(4, 1, 4)
    plt.title("TIME MASK (replicata su canali)")
    plt.imshow(mask_cat[None, :], aspect="auto", cmap="gray_r")
    plt.yticks([])
    plt.xlabel("C * Tpatch")

    plt.tight_layout()
    plt.show()

    # -------------------------
    # ZOOM SU UN CANALE
    # -------------------------
    ch = 0
    start = ch * Tpatch
    end = (ch + 1) * Tpatch

    clean = clean_cat[:, start:end]
    masked = masked_cat[:, start:end]
    diff = masked - clean

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.title(f"Channel {ch} CLEAN")
    plt.imshow(clean, aspect="auto", origin="lower")
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.title(f"Channel {ch} MASKED")
    plt.imshow(masked, aspect="auto", origin="lower")
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.title(f"Channel {ch} DIFF")
    vmax = max(abs(diff.min()), abs(diff.max()))
    plt.imshow(diff, aspect="auto", origin="lower",
               cmap="bwr", vmin=-vmax, vmax=vmax)
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.title("TIME MASK")
    plt.imshow(mask_t[None, :], aspect="auto", cmap="gray_r")
    plt.yticks([])

    plt.tight_layout()
    plt.show()

    print(" Visualizzazione STFT concatenata completata")


