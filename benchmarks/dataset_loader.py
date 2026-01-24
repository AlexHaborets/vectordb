import os
import requests
import h5py
import numpy as np

DATASETS = {
    "siftsmall": "siftsmall-128-euclidean.hdf5",
    "sift": "sift-128-euclidean.hdf5",         
    "glove": "glove-100-angular.hdf5"
}

BASE_URL = "https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main"

def download_file(filename):
    url = f"{BASE_URL}/{filename}"
    if os.path.exists(filename):
        return

    print(f"Downloading {filename} from Hugging Face...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=32768):
                    f.write(chunk)
    except Exception as e:
        if os.path.exists(filename):
            os.remove(filename) 
        raise e

def load_dataset(name="siftsmall"):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Options: {list(DATASETS.keys())}")
    
    filename = DATASETS[name]
    download_file(filename)
    
    with h5py.File(filename, 'r') as f:
        train = np.array(f['train'])
        test = np.array(f['test'])
        neighbors = np.array(f['neighbors'])
        
    return train, test, neighbors