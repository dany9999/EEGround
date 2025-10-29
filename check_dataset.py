import h5py
import os

# Percorso principale
base_folder = "/home/bit/Scrivania/Datasets/Bipolar/chb_mit/8sec"

# Dizionario per accumulare i conteggi
segment_counts = {}

print("Conteggio dei segmenti per ciascun soggetto...\n")

# Scansione di tutte le sottocartelle (es: chb01, chb02, ecc.)
for root, _, files in os.walk(base_folder):
    if not any(f.endswith(".h5") for f in files):
        continue  # salta le cartelle senza file .h5

    subject = os.path.basename(root)
    total_segments = 0

    for filename in sorted(files):
        if filename.endswith(".h5"):
            path = os.path.join(root, filename)
            try:
                with h5py.File(path, "r") as f:
                    keys = list(f.keys())
                    if len(keys) == 0:
                        continue
                    dset = f[keys[0]]
                    num_segments = dset.shape[0]
                    total_segments += num_segments
            except Exception as e:
                print(f" Errore con {subject}/{filename}: {e}")

    segment_counts[subject] = total_segments
    print(f"📂 {subject}: {total_segments} segmenti totali")

print("\n Conteggio completato!\n")

# Stampa un riepilogo ordinato
for subject, count in sorted(segment_counts.items()):
    print(f"{subject}: {count} segmenti")