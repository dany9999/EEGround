import torch
import numpy as np
import h5py
from BIOT_vanilla.biot import BIOTClassifier


# ----------------------------------------------------
# Normalizzazione quantile 0.95 (come nel dataset)
# ----------------------------------------------------
def quantile_normalize(x, q=0.95):
    q_vals = np.quantile(np.abs(x), q, axis=-1, keepdims=True) + 1e-8
    return x / q_vals


# ----------------------------------------------------
# Carica modello dal checkpoint Lightning
# ----------------------------------------------------
def load_trained_model(checkpoint_path, n_channels=16):
    model = BIOTClassifier(
        n_channels=n_channels,
        n_fft=250,
        hop_length=200
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Lightning salva i pesi come model.xxx → li rimuoviamo
    state = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            state[k.replace("model.", "")] = v

    model.load_state_dict(state)
    model.eval()

    print(" Modello caricato correttamente.")
    return model


# ----------------------------------------------------
# Inference su un singolo segmento EEG (16, 2000)
# ----------------------------------------------------
def infer_segment(model, x, threshold=0.5):
    x = quantile_normalize(x)  # IMPORTANTISSIMO
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, C, T)

    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    pred = int(prob >= threshold)
    return prob, pred


# ----------------------------------------------------
# Carica file .h5 e inferisci su TUTTI i segmenti
# ----------------------------------------------------
def infer_file(h5_path, model, threshold=0.5):
    with h5py.File(h5_path, "r") as f:
        signals = f["signals"][:]  # shape (segments, 23, 2000)

    signals = signals[:, :16, :]  # selezioniamo solo 16 canali
    results = []

    for seg in signals:
        prob, pred = infer_segment(model, seg, threshold)
        results.append((prob, pred))

    return np.array(results)  # shape (N, 2)


# ----------------------------------------------------
# MAIN USO DA TERMINALE
# ----------------------------------------------------
if __name__ == "__main__":
    checkpoint = "log/CHB-MIT-from_scratch/checkpoints/best-model.ckpt"

    # Se in validazione hai stimato una threshold dinamica → mettila qui:
    threshold = 0.18946151

    model = load_trained_model(checkpoint, n_channels=16)


    # ESEMPIO SU UN FILE:
    patient_test_seizure_1 = "chb22/eeg_chb22_20.h5" 
    patient_test_seizure_2 = "chb22/eeg_chb22_25.h5"
    patient_test_noseizure_1 = "chb22/eeg_chb22_19.h5"
    patient_test_noseizure_2 = "chb22/eeg_chb22_17.h5"

    h5_file = "../../Datasets/Bipolar/chb_mit/8sec/" + patient_test_noseizure_2

    results = infer_file(h5_file, model, threshold)

    print(f"\n=== RISULTATI INFERENZA {patient_test_noseizure_2} ===")
    for i, (prob, pred) in enumerate(results):
        print(f"Segmento {i:03d} → Prob: {prob:.4f}, Pred: {pred}")