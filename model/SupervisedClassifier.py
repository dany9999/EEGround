
import torch.nn as nn
import torch
from model.BIOT import BIOTEncoder
#from BIOT import BIOTEncoder


# supervised fine-tuning module

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

# supervised classifier module
class BIOTClassifier(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_classes=1, n_channels = 23, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, n_channels=n_channels, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x, n_channel_offset=0):
        _, _, out_biot, _ = self.biot(x, n_channel_offset)

        out_biot = out_biot.mean(dim=1)
        x = self.classifier(out_biot)
        return x
    


if __name__ == "__main__":
    x = torch.randn(1, 23, 2560)

    model = BIOTClassifier(emb_size=256, heads=8, depth=4, n_classes=2)
    out  = model(x)
    print(f"output shape: {out.shape}")  # Should be [1, n_classes]
    
