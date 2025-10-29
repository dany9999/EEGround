import h5py
import os

# Percorso della cartella (modifica se necessario)
folder = "../../Datasets/Bipolar/chb_mit/8sec/chb01"

print("Controllo delle shape dei file .h5\n")

for filename in sorted(os.listdir(folder)):
    if filename.endswith(".h5"):
        path = os.path.join(folder, filename)
        try:
            with h5py.File(path, "r") as f:
                # Trova il nome del dataset principale
                keys = list(f.keys())
                if len(keys) == 0:
                    print(f"⚠️ {filename}: nessun dataset trovato.")
                    continue

                # Assumiamo che ci sia un solo dataset principale
                dset = f[keys[0]]
                shape = dset.shape
                print(f"{filename}: shape = {shape}")
        except Exception as e:
            print(f" Errore con {filename}: {e}")