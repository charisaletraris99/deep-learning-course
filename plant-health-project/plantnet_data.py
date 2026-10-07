"""TensorFlow loaders for the locally downloaded Pl@ntNet 75,000-image subset.

Run this file to check one batch from each split. No model is trained.
"""
from pathlib import Path
import csv

DATA_DIR = Path(__file__).resolve().parent / 'datasets/plantnet-300k-v2/balanced-75000'

def species_names(data_dir=DATA_DIR):
    """Mapping from fixed output class ID 0..999 to scientific species name."""
    with (Path(data_dir)/'species_counts.csv').open(encoding='utf-8',newline='') as f:
        return {int(row['species_id']):row['species'] for row in csv.DictReader(f)}

def load_split(split, batch_size=32, image_size=(224,224), data_dir=DATA_DIR, shuffle=None):
    """Return (RGB float32 images in [0,1], int32 species IDs).

    The source's original split is preserved. Validation/test are not shuffled
    by default. Images keep their aspect ratio with zero padding.
    Use model-specific preprocessing if a pretrained model needs another range.
    """
    import tensorflow as tf
    if split not in {'train','val','test'}: raise ValueError('Choose train, val, or test')
    data_dir=Path(data_dir)
    with (data_dir/f'{split}.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    paths=[str(data_dir/r['relative_path']) for r in rows]
    labels=[int(r['species_id']) for r in rows]
    ds=tf.data.Dataset.from_tensor_slices((paths,tf.constant(labels,dtype=tf.int32)))
    if shuffle is None:shuffle=split=='train'
    if shuffle:ds=ds.shuffle(min(len(rows),10000),seed=20260927,reshuffle_each_iteration=True)
    def decode(path,label):
        image=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False)
        image.set_shape([None,None,3])
        image=tf.image.convert_image_dtype(image,tf.float32)
        return tf.image.resize_with_pad(image,image_size[0],image_size[1]),label
    return ds.map(decode,num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_my_photo(path,image_size=(224,224)):
    """Prepare your own photograph with exactly the same preprocessing."""
    import tensorflow as tf
    image=tf.io.decode_image(tf.io.read_file(str(path)),channels=3,expand_animations=False)
    image.set_shape([None,None,3])
    image=tf.image.convert_image_dtype(image,tf.float32)
    return tf.image.resize_with_pad(image,*image_size)[None,...]

if __name__=='__main__':
    for split in ['train','val','test']:
        images,labels=next(iter(load_split(split,batch_size=4,shuffle=False)))
        print(split,images.shape,labels.numpy().tolist())
