import torch

ckpt = torch.load("logs/pretrain/model_epoch_1.pt", map_location="cpu")

# a volte il checkpoint è un dict con 'model_state_dict'
if "model_state_dict" in ckpt:
    sd = ckpt["model_state_dict"]
else:
    sd = ckpt

print("\n=== Prime 40 chiavi nel checkpoint ===")
for i, k in enumerate(list(sd.keys())[:40]):
    print(f"{i+1:02d}. {k}")
print("Totale chiavi:", len(sd))
print("================================================\n")