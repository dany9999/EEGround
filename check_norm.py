import numpy as np
import matplotlib.pyplot as plt

# === Percorsi statici ===
pretrain_paths = {
    "TUAB_Abnormal_LE": "../../Datasets/Bipolar/TUH/Dataset_bipolar_TUH/TUAB/Abnormal/LE",
    "TUAB_Normal_LE": "../../Datasets/Bipolar/TUH/Dataset_bipolar_TUH/TUAB/Normal/LE",
    "TUEP_Abnormal_LE": "../../Datasets/Bipolar/TUH/Dataset_bipolar_TUH/TUEP/Abnormal/LE",
    "TUEP_Normal_LE": "../../Datasets/Bipolar/TUH/Dataset_bipolar_TUH/TUEP/Normal/LE",
    "TUSZ_Abnormal_LE": "../../Datasets/Bipolar/TUH/Dataset_bipolar_TUH/TUSZ/Abnormal/LE",
    "TUEV_Abnormal_LE": "../../Datasets/Bipolar/TUH/Dataset_bipolar_TUH/TUEV/Abnormal/LE",
}

# === Percorsi finetuning ===
finetune_paths = {
    "FINETUNE_run1": "mu_train_finetuning_4s_run1.npy",
    "FINETUNE_run2": "mu_train_finetuning_4s_run2.npy",
}
finetune_sigmas = {
    "FINETUNE_run1": "sigma_train_finetuning_4s_run1.npy",
    "FINETUNE_run2": "sigma_train_finetuning_4s_run2.npy",
}


def load_stats(name, mean_path, std_path):
    try:
        mu = np.load(mean_path).squeeze()
        sigma = np.load(std_path).squeeze()
        if mu.ndim != 1 or sigma.ndim != 1:
            print(f"⚠️ Forma non 1D in {name}: {mu.shape}, {sigma.shape}")
        return name, mu, sigma
    except Exception as e:
        print(f"❌ Errore caricando {name}: {e}")
        return None


stats = []

# --- Pretrain ---
for name, path in pretrain_paths.items():
    mean_path = f"{path}/mean.npy"
    std_path = f"{path}/standard_deviation.npy"
    s = load_stats(name, mean_path, std_path)
    if s:
        stats.append(s)

# --- Finetune ---
for name, mu_path in finetune_paths.items():
    if name in finetune_sigmas:
        sigma_path = finetune_sigmas[name]
        s = load_stats(name, mu_path, sigma_path)
        if s:
            stats.append(s)

# --- Riepilogo ---
print("\n=== RIEPILOGO ===")
for name, mu, sigma in stats:
    print(f"{name:<25} | μ_mean={np.mean(mu):>7.3f} | σ_mean={np.mean(sigma):>7.3f}")

# --- Plot comparativo ---
plt.figure(figsize=(12, 5))
for name, mu, _ in stats:
    plt.plot(mu, label=f"{name} μ")
plt.title("Media per canale (μ)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
for name, _, sigma in stats:
    plt.plot(sigma, label=f"{name} σ")
plt.title("Deviazione standard per canale (σ)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()