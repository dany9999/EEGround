import os
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
# Cartella base del pretrain
BASE_DIR = "/home/inbit/Scrivania/Datasets/Bipolar/TUH/Dataset_bipolar_TUH"
# Cartella corrente = dove hai le medie del finetuning
CURRENT_DIR = os.getcwd()

# === 1️⃣ Scansione del pretrain ===
stats = []

for root, dirs, files in os.walk(BASE_DIR):
    if "mean.npy" in files and "standard_deviation.npy" in files:
        mu_path = os.path.join(root, "mean.npy")
        sigma_path = os.path.join(root, "standard_deviation.npy")

        mu = np.load(mu_path)
        sigma = np.load(sigma_path)

        stats.append({
            "source": root.replace(BASE_DIR + "/", ""),
            "mu": mu,
            "sigma": sigma
        })

if not stats:
    print("❌ Nessun file mean/std trovato nel dataset TUH!")
    exit()

print(f"✅ Trovati {len(stats)} sottodataset TUH con mean/std.\n")

# === 2️⃣ Aggiunge le statistiche del finetuning ===
finetune_stats = []
for file in os.listdir(CURRENT_DIR):
    if file.startswith("mu_train_finetuning") and file.endswith(".npy"):
        run_id = file.split("_")[-1].replace(".npy", "")
        sigma_file = f"sigma_train_finetuning_{run_id}.npy"
        sigma_path = os.path.join(CURRENT_DIR, sigma_file)
        if os.path.exists(sigma_path):
            mu = np.load(os.path.join(CURRENT_DIR, file))
            sigma = np.load(sigma_path)
            finetune_stats.append({
                "source": f"FINETUNE_{run_id}",
                "mu": mu,
                "sigma": sigma
            })

if finetune_stats:
    stats += finetune_stats
    print(f"✅ Aggiunti {len(finetune_stats)} set di finetuning locali.\n")
else:
    print("⚠️ Nessun file finetuning trovato nella cartella corrente.\n")

# === 3️⃣ Stampa riepilogo numerico ===
print("=== RIEPILOGO MEDIE GLOBALI ===")
for s in stats:
    print(f"{s['source']:<50} | μ_mean={np.mean(s['mu']):>7.3f} | σ_mean={np.mean(s['sigma']):>7.3f}")

# === 4️⃣ Plot comparativo ===
plt.figure(figsize=(12, 5))
for s in stats:
    plt.plot(s["mu"], label=f"{s['source']} μ")
plt.title("Media per canale (μ)")
plt.xlabel("Canale")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
for s in stats:
    plt.plot(s["sigma"], label=f"{s['source']} σ")
plt.title("Deviazione standard per canale (σ)")
plt.xlabel("Canale")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# === 5️⃣ Analisi comparativa (solo se c'è finetune) ===
if finetune_stats:
    mu_ft = finetune_stats[0]["mu"]
    sigma_ft = finetune_stats[0]["sigma"]

    print("\n=== CONFRONTO CON FINETUNE ===")
    for s in stats:
        if "FINETUNE" in s["source"]:
            continue
        ratio = sigma_ft / s["sigma"]
        shift = (mu_ft - s["mu"]) / s["sigma"]
        print(f"{s['source']:<50} | mean ratio σ_ft/σ_pre = {np.mean(ratio):.3f} ± {np.std(ratio):.3f} | mean shift (μ_ft-μ_pre)/σ_pre = {np.mean(np.abs(shift)):.3f}")