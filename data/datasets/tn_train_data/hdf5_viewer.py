import h5py

def explore_hdf5(file_path):
    """
    Liest eine HDF5-Datei ein und zeigt die Struktur (Gruppen und Datensätze) an.
    """
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Datentyp: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    # Öffne die Datei im Lesemodus
    with h5py.File(file_path, 'r') as file:
        print(f"Inhalt der Datei: {file_path}")
        file.visititems(print_structure)

# Beispielaufruf - ersetze 'datei.hdf5' mit deinem Dateipfad
explore_hdf5('data/datasets/tn_train_data/scene_0004.hdf5')
