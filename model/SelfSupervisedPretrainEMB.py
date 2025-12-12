

# from model.BIOTEMB import BIOTEncoder
# import torch.nn as nn
# import torch


# # unsupervised pre-train module
# class UnsupervisedPretrain(nn.Module):
#     def __init__(self, emb_size=256, heads=8, depth=4, n_channels=23, n_fft=200, hop_length=100, mask_ratio=0.75, **kwargs):
#         super(UnsupervisedPretrain, self).__init__()
#         self.biot = BIOTEncoder(emb_size, heads, depth, n_channels,n_fft, hop_length, mask_ratio, pretraining=True, **kwargs)
#         self.prediction = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.GELU(),
#             nn.Linear(256, 256),
#         )

#     def forward(self, x, n_channel_offset=0):
#         emb, masked_emb, out_biot,  mask = self.biot(x, n_channel_offset)
        
#         pred_emb = self.prediction(out_biot)

        
#         return emb, mask, masked_emb, pred_emb   
    

    


# if __name__ == "__main__":
#     x = torch.randn(1, 18, 1000)

#     model = UnsupervisedPretrain(n_fft=128, hop_length=32, depth=4, heads=8)
#     original, mask, masked, reconstruction  = model(x)
#     print(f"original shape: {original.shape}")
#     print(f"mask shape: {mask.shape}")
#     print(f"reconstruction shape: {reconstruction.shape}") 




from model.BIOTEMB import BIOTEncoder
import torch.nn as nn
import torch


# unsupervised pre-train module
class UnsupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_channels=23, n_fft=200, hop_length=100, mask_ratio=0.75, **kwargs):
        super(UnsupervisedPretrain, self).__init__()
        self.biot = BIOTEncoder(emb_size, heads, depth, n_channels,n_fft, hop_length, mask_ratio, pretraining=True, **kwargs)
        self.stft_decoder = nn.Linear(emb_size, n_fft // 2 + 1)

    def forward(self, x, n_channel_offset=0):
        out, stft_clean_cat, time_masks = self.biot(x)

        
        # ---- decode STFT
        pred_stft = self.stft_decoder(out)      # (B, Seq, F)
        pred_stft = pred_stft.permute(0, 2, 1)  # (B, F, Seq)

        # time_masks: (B, Tpatch)
        # Seq = C * Tpatch

        # 1) espandi la mask sui canali (token-level)
        token_mask = time_masks.repeat(1, x.shape[1])  # (B, Seq)

        # 2) espandi sulle frequenze
        token_mask = token_mask.unsqueeze(1).expand(-1, pred_stft.shape[1], -1)
        # shape finale: (B, F, Seq)
        return pred_stft, stft_clean_cat, token_mask
    

import torch
import matplotlib.pyplot as plt


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Parametri test
    # -------------------------
    B = 1
    C = 18
    T = 1000
    n_fft = 250
    hop_length = 125
    mask_ratio = 0.30

    # -------------------------
    # Input fittizio
    # -------------------------
    x = torch.randn(B, C, T).to(device)

    # -------------------------
    # Modello
    # -------------------------
    model = UnsupervisedPretrain(
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=C,
        n_fft=n_fft,
        hop_length=hop_length,
        mask_ratio=mask_ratio
    ).to(device)

    model.eval()

    # -------------------------
    # Forward
    # -------------------------
    with torch.no_grad():
        pred_stft, stft_clean_cat, token_mask = model(x)

    # -------------------------
    # Shape check
    # -------------------------
    print("\n=== SHAPES ===")
    print("Input x:", x.shape)
    print("Pred STFT:", pred_stft.shape)
    print("Clean STFT:", stft_clean_cat.shape)
    print("Token mask:", token_mask.shape)

    assert pred_stft.shape == stft_clean_cat.shape, \
        " pred_stft e stft_clean_cat NON hanno la stessa shape"

    B2, F, Seq = pred_stft.shape
    assert token_mask.shape == (B2, 1, Seq), \
        " token_mask shape errata"

    print(" Shape check OK")

    # -------------------------
    # Mask ratio check
    # -------------------------
    mask_ratio_eff = token_mask.sum().item() / token_mask.numel()
    print(f"Mask ratio atteso: {mask_ratio:.3f}")
    print(f"Mask ratio reale:  {mask_ratio_eff:.3f}")

    # -------------------------
    # Visualizzazione mask
    # -------------------------
    mask_np = token_mask[0, 0].cpu().numpy()  # (Seq,)

    plt.figure(figsize=(14, 2))
    plt.title("Token mask (1 = masked)")
    plt.imshow(mask_np[None, :], aspect="auto", cmap="gray_r")
    plt.yticks([])
    plt.xlabel("Token index (Seq = C * Tpatch)")
    plt.colorbar(label="Masked")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Visualizzazione STFT (una frequenza)
    # -------------------------
    f_idx = F // 2  # frequenza centrale

    plt.figure(figsize=(14, 4))

    plt.subplot(2, 1, 1)
    plt.title(f"STFT CLEAN – freq bin {f_idx}")
    plt.plot(stft_clean_cat[0, f_idx].cpu())
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.title(f"STFT PRED – freq bin {f_idx}")
    plt.plot(pred_stft[0, f_idx].cpu())
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n✔ Test completato correttamente")

