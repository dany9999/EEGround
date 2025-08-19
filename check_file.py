import os
import h5py

def inspect_h5_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".h5")]
    print(f"Trovati {len(files)} file .h5 in {folder_path}\n")
    for file in sorted(files):
        file_path = os.path.join(folder_path, file)
        try:
            with h5py.File(file_path, 'r') as f:
                print(f"{file}:")
                def print_structure(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"  {name}: shape = {obj.shape}, dtype = {obj.dtype}")
                f.visititems(print_structure)
        except Exception as e:
            print(f"{file}: errore nel caricamento - {e}")



#conta num di samples in ogni cartella


def count_samples_in_h5_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".h5")]
    total_samples = 0
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            with h5py.File(file_path, 'r') as f:
                if 'signals' in f:
                    n_samples = f['signals'].shape[0]
                    total_samples += n_samples
                    print(f"{file}: {n_samples} samples")
                else:
                    print(f"{file}: 'signals' dataset non trovato")
        except Exception as e:
            print(f"{file}: errore nel caricamento - {e}")
    print(f"Totale campioni in {folder_path}: {total_samples}")

if __name__ == "__main__":
    dataset_path = os.path.abspath("../../Datasets/Bipolar/chb_mit/bipolar_data/")

    for patient_folder in os.listdir(dataset_path):
        patient_path = os.path.join(dataset_path, patient_folder)
        if os.path.isdir(patient_path):
            print(f"Controllando {patient_folder}...")
            count_samples_in_h5_files(patient_path)