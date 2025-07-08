
from model.BIOT import BIOTEncoder
import torch.nn as nn
import torch




class EEGDecoder(nn.Module):
    def __init__(self, emb_size=256, n_channels=23, n_fft=200, hop_length=100, signal_len=1000):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.signal_len = signal_len
        self.n_channels = n_channels

        # Proiezione dall'embedding allo spettrogramma complesso (real + imag)
        self.proj = nn.Linear(emb_size, (n_fft // 2 + 1) * 2)  # real + imag parts

    def forward(self, x):
        """
        x: [B, C*T', emb_size]
        output: [B, C, T]
        """
        B, L, D = x.shape
        T_prime = L // self.n_channels
        x = x.view(B, self.n_channels, T_prime, D)  # [B, C, T', D]

        x = self.proj(x)  # [B, C, T', 2*F]
        x = x.view(B, self.n_channels, T_prime, -1, 2)  # [B, C, T', F, 2]
        x_complex = torch.complex(x[..., 0], x[..., 1])  # [B, C, T', F]

        x_complex = x_complex.permute(0, 1, 3, 2)  # [B, C, F, T']

        out = []
        for c in range(self.n_channels):
            rec = torch.istft(
                x_complex[:, c],  # [B, F, T']
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=self.signal_len,
                center=False,
            )
            out.append(rec)

        out = torch.stack(out, dim=1)  # [B, C, T]
        return out


# Unsupervised pre-training module
class UnsupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_channels=23, mask_ratio=0.15, signal_len=1000, **kwargs):
        super(UnsupervisedPretrain, self).__init__()

        self.biot = BIOTEncoder(
            emb_size=emb_size,
            heads=heads,
            depth=depth,
            n_channels=n_channels,
            **kwargs
        )
        self.mask_ratio = mask_ratio

        self.decoder = EEGDecoder(
            emb_size=emb_size,
            n_channels=n_channels,
            n_fft=kwargs.get('n_fft', 200),
            hop_length=kwargs.get('hop_length', 100),
            signal_len=signal_len,
        )

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

    def forward(self, x, n_channel_offset=0):
        # Output del BIOT encoder

        if self.mask_ratio > 0.0:
            
            x, mask = self.random_masking(x)

        out = self.biot(x, n_channel_offset)

        # Decodifica dell’EEG raw ricostruito
        raw_reconstructed = self.decoder(out)

        return raw_reconstructed, mask
    




if __name__ == "__main__":
    x = torch.randn(1, 23, 2560)

    model = UnsupervisedPretrain(n_fft=200, hop_length=200, depth=4, heads=8)
    
    

    original, mask, reconstruction  = model(x)


    print(f"original shape: {original.shape}")
    print(f"mask shape: {mask.shape}")
    print(f"reconstruction shape: {reconstruction.shape}") 

    