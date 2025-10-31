import torch
from BIOT_vanilla.biot import BIOTClassifier

ckpt = torch.load("BIOT_vanilla/pretrained_models/EEG-PREST-16-channels.ckpt", map_location="cpu")

# Se Lightning
if "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]

print("\nEsempio di chiavi nel checkpoint:")
for k in list(ckpt.keys())[:20]:
    print(k)

model =     model = BIOTClassifier( n_channels=18, n_fft=200,hop_length=100,)
print("\nEsempio di chiavi nel modello:")
for k in list(model.state_dict().keys())[:20]:
    print(k)