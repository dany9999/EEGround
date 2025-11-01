import torch
ckpt = torch.load("logs/pretrain/encoder_only_epoch_1.pt", map_location="cpu")
print("Totale chiavi:", len(ckpt.keys()))
for i, k in enumerate(list(ckpt.keys())[:30]):
    print(f"{i+1:02d}. {k}")