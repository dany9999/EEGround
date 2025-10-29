# import h5py
# import os

# # Percorso della cartella (modifica se necessario)
# folder = "../../Datasets/Bipolar/chb_mit/8sec/chb04"

# print("Controllo delle shape dei file .h5\n")

# for filename in sorted(os.listdir(folder)):
#     if filename.endswith(".h5"):
#         path = os.path.join(folder, filename)
#         try:
#             with h5py.File(path, "r") as f:
#                 # Trova il nome del dataset principale
#                 keys = list(f.keys())
#                 if len(keys) == 0:
#                     print(f"⚠️ {filename}: nessun dataset trovato.")
#                     continue

#                 # Assumiamo che ci sia un solo dataset principale
#                 dset = f[keys[0]]
#                 shape = dset.shape
#                 print(f"{filename}: shape = {shape}")
#         except Exception as e:
#             print(f" Errore con {filename}: {e}")


import h5py
import os


base_folder = "../../Datasets/Bipolar/chb_mit/bipolar_data"
# Dizionario per accumulare i conteggi
segment_counts = {}

print(" Conteggio dei segmenti per ciascun soggetto...\n")

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
    print(f" {subject}: {total_segments} segmenti totali")

print("\n Conteggio completato!\n")

# Stampa un riepilogo ordinato
totale = sum(segment_counts.values())
for subject, count in sorted(segment_counts.items()):
    print(f"{subject}: {count} segmenti")
print(f"Totale: {totale} segmenti")

