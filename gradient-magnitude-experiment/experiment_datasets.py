"""Locally cached public classification datasets for optimizer experiments."""
from pathlib import Path
import gzip
import io
import json
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import get_file

DATA_DIR = Path(__file__).resolve().parent / 'datasets'
DATASETS = {
    'iris': ('sklearn Iris', 'StandardScaler fitted on 120 pre-validation samples, matching exercise1a.py'),
    'digits': ('sklearn digits', 'Pixel values divided by 16'),
    'letter': ('UCI Letter Recognition', 'Feature values divided by 15'),
    'mnist': ('MNIST', 'Pixel values divided by 255'),
    'fashion_mnist': ('Fashion-MNIST', 'Pixel values divided by 255'),
}
BASE = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'


def download(name, url, dataset, file_hash=None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return get_file(name, origin=url, cache_dir=str(DATA_DIR), cache_subdir=dataset,
                    file_hash=file_hash)


def load_additional(dataset):
    if dataset == 'letter':
        path = download('letter-recognition.zip',
            'https://archive.ics.uci.edu/static/public/59/letter+recognition.zip', dataset)
        with zipfile.ZipFile(path) as z:
            frame = pd.read_csv(io.BytesIO(z.read('letter-recognition.data')), header=None)
        labels = frame[0].map(lambda value: ord(value) - ord('A')).to_numpy()
        features = frame.iloc[:, 1:].to_numpy(dtype='float32') / 15.0
        assert features.shape == (20000, 16) and set(labels) == set(range(26))
        trainval, test = train_test_split(np.arange(len(labels)), test_size=.2,
                                         stratify=labels, random_state=42)
        classes = 26
        # Convenient original feature table for inspecting the dataset.
        csv_path = DATA_DIR / dataset / 'letter-recognition.csv'
        if not csv_path.exists():
            frame.columns = ['letter','x_box','y_box','width','height','on_pixels','x_bar','y_bar',
                             'x2_bar','y2_bar','xy_bar','x2y_bar','xy2_bar','x_edge','x_edge_y',
                             'y_edge','y_edge_x']
            frame.to_csv(csv_path, index=False)
    elif dataset in {'mnist', 'fashion_mnist'}:
        if dataset == 'mnist':
            path = download('mnist.npz', BASE + 'mnist.npz', dataset,
                '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1')
            with np.load(path, allow_pickle=False) as d:
                images = np.concatenate([d['x_train'], d['x_test']])
                labels = np.concatenate([d['y_train'], d['y_test']])
        else:
            def read_idx(name, offset):
                path = download(name, BASE + name, dataset)
                with gzip.open(path, 'rb') as stream:
                    return np.frombuffer(stream.read(), dtype=np.uint8, offset=offset)
            train_labels = read_idx('train-labels-idx1-ubyte.gz', 8)
            test_labels = read_idx('t10k-labels-idx1-ubyte.gz', 8)
            train_images = read_idx('train-images-idx3-ubyte.gz', 16).reshape(60000,28,28)
            test_images = read_idx('t10k-images-idx3-ubyte.gz', 16).reshape(10000,28,28)
            images = np.concatenate([train_images, test_images])
            labels = np.concatenate([train_labels, test_labels])
        assert images.shape == (70000,28,28) and labels.shape == (70000,)
        assert set(labels) == set(range(10))
        features = images.reshape(70000,784).astype('float32') / 255.0
        trainval, test = np.arange(60000), np.arange(60000,70000)
        classes = 10
    else:
        raise ValueError(f'Unknown additional dataset: {dataset}')
    train, val = train_test_split(trainval, test_size=.1, stratify=labels[trainval], random_state=42)
    assert np.isfinite(features).all() and features.min() >= 0 and features.max() <= 1
    metadata = dict(dataset=DATASETS[dataset][0], samples=len(labels), classes=classes,
                    features=features.shape[1], train=len(train), validation=len(val), test=len(test),
                    preprocessing=DATASETS[dataset][1], split_seed=42)
    (DATA_DIR / dataset / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return features, np.eye(classes, dtype='float32')[labels], train, val, test, None


if __name__ == '__main__':
    for name in ['letter', 'mnist', 'fashion_mnist']:
        data = load_additional(name)
        print(name, data[0].shape, 'splits:', [len(part) for part in data[2:5]], flush=True)
        del data
