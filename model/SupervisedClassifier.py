import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.BIOTEMB import BIOTEncoder


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
    def __init__(self, emb_size=256, heads=8, depth=4, n_classes=1, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, pretraining=False, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.biot(x)
        x = x.mean(dim=1)        # (B, 256)
        x = self.classifier(x)
        return x
    


if __name__ == "__main__":
    x = torch.randn(1, 18, 1000)
    model = BIOTClassifier(n_fft=250, hop_length=125, depth=4, heads=8)
    out = model(x)
    print(out.shape)
   


