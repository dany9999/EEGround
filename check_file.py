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

if __name__ == "__main__":
    dataset_path = os.path.abspath("../../Datasets/Bipolar/chb_mit/bipolar_data/chb01")
    inspect_h5_files(dataset_path)