

from model.BIOTEMB import BIOTEncoder
import torch.nn as nn
import torch


# unsupervised pre-train module
class UnsupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_channels=23, n_fft=200, hop_length=100, mask_ratio=0.3, **kwargs):
        super(UnsupervisedPretrain, self).__init__()
        self.biot = BIOTEncoder(emb_size, heads, depth, n_channels,n_fft, hop_length, mask_ratio, pretraining=True, **kwargs)
        self.prediction = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

    def forward(self, x, n_channel_offset=0):
        emb, masked_emb, out_biot,  mask = self.biot(x, n_channel_offset)
        
        pred_emb = self.prediction(out_biot)

        
        return emb, mask, masked_emb, pred_emb   
    

    


if __name__ == "__main__":
    x = torch.randn(1, 18, 1000)

    model = UnsupervisedPretrain(n_fft=128, hop_length=32, depth=4, heads=8)
    original, mask, masked, reconstruction  = model(x)
    print(f"original shape: {original.shape}")
    print(f"mask shape: {mask.shape}")
    print(f"reconstruction shape: {reconstruction.shape}") 