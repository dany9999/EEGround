from BIOT_raw_reconstruction import BIOTEncoder  # il tuo file con BIOTEncoder
import torch
import torch.nn as nn




class UnsupervisedPretrain(nn.Module):
    def __init__(
        self,
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=18,
        n_fft=250,
        hop_length=125,
        mask_ratio=0.30,
    ):
        super().__init__()
        self.encoder = BIOTEncoder(
            emb_size=emb_size,
            heads=heads,
            depth=depth,
            n_channels=n_channels,
            n_fft=n_fft,
            hop_length=hop_length,
            mask_ratio=mask_ratio,
            pretraining=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, emb_size)
        )

    def forward(self, x):
        emb, masked_emb, out_enc, mask = self.encoder(x)

        pred_emb = self.decoder(out_enc)

        return emb, pred_emb, mask