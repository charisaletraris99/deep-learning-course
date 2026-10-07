from pathlib import Path
import json
root=Path(__file__).resolve().parent
data=root/'datasets/plantnet-300k-v2/balanced-75000'
s=json.loads((data/'summary.json').read_text())
text=f'''# Pl@ntNet 75,000 image subset

Ready-to-use folder: `datasets/plantnet-300k-v2/balanced-75000`

**75,000 verified images covering all 1,000 species.** Downloading stopped at the user's request after 79,831 files had been saved. The active subset contains exactly 75,000; the extra 4,831 remain in the original `balanced-80000` download folder. Do not train using the old 80,000-row manifests: they refer to 169 images that were not downloaded.

## Class balance

The class allocation minimizes variance for a fixed total of 75,000, subject to the number of images already saved for each species. Capacity-constrained water filling gives every species another image until its available pool runs out. All 1,000 species are retained; no oversampling or synthetic images were added.

- Per-species minimum: {s['min_per_species']}
- Per-species maximum: {s['max_per_species']}
- Mean: 75
- Population standard deviation: {s['population_stddev']:.3f}
- Species using all locally available images: {s['species_at_local_capacity']}
- Counts of the remaining species: {s['unexhausted_quotas']}

Exactly 75 images for every species is impossible because many species have fewer images available. The optimality claim is relative to the saved pool, not all 306,087 images in the source.

Original source splits are preserved. Quotas within each species are proportional to available split sizes, using largest remainders. Final images are sampled uniformly without replacement from the saved species/split pools (seed 20260927). The preceding partial download used seeded circular archive-order blocks for efficiency, so the final subset is not an independent uniform sample of the entire source dataset.

| Split | Images | Species |
|---|---:|---:|
| train | {s['split_counts']['train']:,} | {s['species_per_split']['train']} |
| val | {s['split_counts']['val']:,} | {s['species_per_split']['val']} |
| test | {s['split_counts']['test']:,} | {s['species_per_split']['test']} |

No selected observation ID occurs in more than one split. Some species are missing from validation because the user stopped the download before that source block was fetched. Do not move training images into the test set to fill gaps.

## Files and storage

- `images/train/0000/...jpg`, `images/val/0000/...jpg`, `images/test/0000/...jpg`: ordinary extracted image files.
- `train.csv`, `val.csv`, `test.csv`: image paths, fixed species IDs, scientific names, organ, observation ID, photographer and license.
- `manifest.csv`: all 75,000 images, source ZIP locations and checksums.
- `species_counts.csv`: available and selected counts, plus class names.
- `class_names.json`: model output ID to scientific name.
- `summary.json`: final statistics and verification details.
- Parent folder: official CSV metadata, original README and Zenodo source record.

The subset images use NTFS hard links to the saved files, so there is no second copy consuming approximately 10 GB. Treat these images as read-only: editing a linked image changes the corresponding original too. Image processing during model loading does not modify the source files.

Every saved image was checked against its original ZIP CRC32 and size. Official metadata MD5 checksums were verified against Zenodo. The full 41.8 GB ZIP was not downloaded.

## TensorFlow quick start

Run from `plant-health-project` using the course environment with TensorFlow:

```python
from plantnet_data import load_split, species_names, load_my_photo

train = load_split('train')
validation = load_split('val')
test = load_split('test')
names = species_names()

# Classifier: 1,000 outputs and integer-label loss, e.g.
# SparseCategoricalCrossentropy, with from_logits matching the model.
# model.fit(train, validation_data=validation, epochs=...)
# model.evaluate(test)

# After training, predict a photograph of your own:
# batch = load_my_photo('my_plant.jpg')
# predicted_id = int(model.predict(batch).argmax(axis=1)[0])
# print(names[predicted_id])
```

The loader returns RGB float32 images in [0,1], resized with padding to 224 × 224, and fixed integer class IDs 0–999. Apply any additional preprocessing required by your chosen pretrained model. Do not independently renumber classes in the validation/test folders.

Use training/validation for model selection and reserve test for final evaluation. Macro-averaged metrics help expose performance on rare species. To measure accuracy on your own photographs, supply their true species labels. The classifier's output vocabulary covers only the listed 1,000 species.

## Source and scope

- Official record and DOI: https://zenodo.org/records/10419064 and https://doi.org/10.5281/zenodo.10419064
- Original image archive: https://zenodo.org/api/records/10419064/files/images.zip/content
- Metadata URLs: replace `images.zip` with `plantnet300K_metadata.csv`, `species_metadata.csv` or `README.md` in the same API URL.
- Cite Garcin et al. (2021), *Pl@ntNet-300K: a plant image dataset with high label ambiguity and a long-tailed distribution*, NeurIPS Datasets and Benchmarks.

Per-image author and license are preserved; licenses differ by image. The dataset labels species and photographed organs. It does not label disease, treatment, watering needs, or tree/shrub/herb growth form.

`Dataset_Sources.docx` contains the source documentation. `finalize_plantnet_75000.py` reproduces the subset from saved files. The older `plantnet_subset.py` describes the original 80,000-image download plan; do not resume it unless you intentionally want additional downloads.
'''
(root/'PLANTNET_QUICKSTART.md').write_text(text,encoding='utf-8')
readme=(root/'README.md').read_text(encoding='utf-8')
readme=readme.replace('80.000 εικόνων','75.000 εικόνων').replace('balanced-80000','balanced-75000').replace('Το `summary.json` καταγράφει την τελική κατάσταση και τον έλεγχο ακεραιότητας, ενώ το `progress.json` την πρόοδο.','Το `summary.json` επιβεβαιώνει την ολοκλήρωση και τον έλεγχο ακεραιότητας. Η επιλογή έγινε από τις 79.831 ήδη αποθηκευμένες εικόνες, μετά τη διακοπή της λήψης.')
(root/'README.md').write_text(readme,encoding='utf-8')
(root/'datasets/plantnet-300k-v2/balanced-80000/USE_BALANCED_75000.txt').write_text('Original interrupted download. 79,831 images saved; old 80,000-row manifests include missing files. Use ../balanced-75000 for training/testing. Downloads stopped at user request.\n',encoding='utf-8')
print('Updated 75k quickstart and project README')
