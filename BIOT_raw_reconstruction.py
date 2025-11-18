import math
import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, Freq, Time) → (B, Time, Freq)
        return self.projection(x)   # (B, Time, emb)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000)/d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


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
    ):
        super().__init__()

        self.pretraining = pretraining
        self.mask_ratio = mask_ratio
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size, n_freq=n_fft // 2 + 1
        )

        self.positional_encoding = PositionalEncoding(emb_size)

        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=2048,
            attn_dropout=0.1,
            attn_layer_dropout=0.1,
        )

        self.channel_token = nn.Embedding(n_channels, emb_size)
        self.index = torch.arange(n_channels, requires_grad=False)


    def stft(self, x):
        window = torch.hann_window(self.n_fft, device=x.device)
        spec = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=False,
            onesided=True,
            return_complex=True
        )
        return torch.abs(spec)  # (B, Freq, Time)


    def mask_tokens(self, x):
        B, L, D = x.shape
        num_mask = int(L * self.mask_ratio)

        mask_indices = torch.randperm(L, device=x.device)[:num_mask]
        mask = torch.zeros(L, dtype=torch.bool, device=x.device)
        mask[mask_indices] = True

        mask = mask.unsqueeze(0).repeat(B, 1)  # (B, L)

        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask


    def forward(self, x):
        B, C, T = x.shape

        all_embeddings = []

        for ch in range(C):
            st = self.stft(x[:, ch:ch+1, :])    # (B, F, Time)
            emb = self.patch_embedding(st)      # (B, Time, 256)

            tok = self.channel_token(self.index[ch]).unsqueeze(0).unsqueeze(0)
            tok = tok.repeat(B, emb.size(1), 1)

            emb = self.positional_encoding(emb + tok)
            all_embeddings.append(emb)

        emb = torch.cat(all_embeddings, dim=1)   # (B, C*T, 256)

        if self.pretraining:
            masked_emb, mask = self.mask_tokens(emb)
            out = self.transformer(masked_emb)
            return emb, masked_emb, out, mask

        # finetuning mode
        out = self.transformer(emb)
        return out