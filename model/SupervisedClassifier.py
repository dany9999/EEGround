import torch.nn as nn
import torch
from model.BIOTEMB import BIOTEncoder


# Attention Pooling Module
class AttentionPooling(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.attn = nn.Linear(emb_dim, 1)

    def forward(self, x):
        # x: (B, Seq, D)
        weights = torch.softmax(self.attn(x), dim=1)  # (B, Seq, 1)
        pooled = (x * weights).sum(dim=1)  # (B, D)
        return pooled


# Classification Head
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            #nn.Dropout(0.3),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        return self.clshead(x)


# Full classifier module
class BIOTClassifier(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_classes=1, n_channels=19, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(
            emb_size=emb_size,
            heads=heads,
            depth=depth,
            n_channels=n_channels,
            n_fft=128,
            hop_length=32,
            mask_ratio=0.0,
            **kwargs
        )
        self.pooling = AttentionPooling(emb_size)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x, n_channel_offset=0):
        _, _, out_biot, _ = self.biot(x, n_channel_offset)  # out_biot: (B, Seq, D)
        pooled = self.pooling(out_biot)                      # (B, D)
        logits = self.classifier(pooled)                     # (B, 1)
        return logits




# Debugging block
if __name__ == "__main__":
    x = torch.randn(4, 19, 1000)  # (batch_size, channels, samples)

    model = BIOTClassifier(emb_size=256, heads=8, depth=4, n_classes=1)
    out = model(x)
    print(f"output shape: {out.shape}")  # Expected: (4, 1)