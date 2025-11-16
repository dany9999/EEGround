import torch
import numpy as np
from BIOT_vanilla.biot import BIOTClassifier
from pretrain_emb import UnsupervisedPretrain
import re


# --- funzione identica a quella nel tuo progetto ---
def load_pretrained_encoder_into_biot(model, ckpt_path, device="cpu"):
    print(f"\n[LOAD] Caricamento da: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    new_sd = {}
    for k, v in sd.items():
        k_clean = k[len("module."):] if k.startswith("module.") else k
        if k_clean.startswith("encoder."):
            k_clean = re.sub(r"^encoder\.", "biot.", k_clean)
        elif not k_clean.startswith("biot."):
            k_clean = "biot." + k_clean
        new_sd[k_clean] = v

    model_dict = model.state_dict()
    compatible = {k: v for k, v in new_sd.items()
                  if k in model_dict and v.shape == model_dict[k].shape}

    print(f"[LOAD] Layer caricati = {len(compatible)} / {len(model_dict)}")
    model_dict.update(compatible)
    model.load_state_dict(model_dict)

    missing = set(model_dict.keys()) - set(compatible.keys())
    print(f"[LOAD] Layer mancanti (inizializzati da zero): {len(missing)}")

    return model


def compare_models(model_random, model_loaded):
    print("\n================ CONFRONTO PESI ================\n")

    sd_r = model_random.state_dict()
    sd_l = model_loaded.state_dict()

    same = []
    diff = []

    for k in sd_r:
        if k not in sd_l:
            print(f"[MISMATCH] {k} non presente nel modello caricato!")
            continue

        if torch.allclose(sd_r[k], sd_l[k], atol=1e-6):
            same.append(k)
        else:
            diff.append(k)

    print(f"\nLayer IDENTICI: {len(same)}")
    print(f"Layer DIVERSI: {len(diff)}")

    print("\n--- Layer diversi (dovrebbero essere SOLO i pesi dell’encoder) ---")
    for k in diff:
        print("  ", k)


if __name__ == "__main__":
    device = "cpu"

    # === 1. crea modello random (niente pretrain) ===
    model_random = BIOTClassifier(n_channels=18, n_fft=250, hop_length=125)
    model_random.to(device)

    # === 2. crea stesso modello ma con pesi pretrainati ===
    model_loaded = BIOTClassifier(n_channels=18, n_fft=250, hop_length=125)
    model_loaded.to(device)

    ckpt_path = "logs/pretrain/encoder_only_epoch_50.pt"   

    model_loaded = load_pretrained_encoder_into_biot(
        model_loaded, ckpt_path, device
    )

    # === 3. confronto layer per layer ===
    compare_models(model_random, model_loaded)