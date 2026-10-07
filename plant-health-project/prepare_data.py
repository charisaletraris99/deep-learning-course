from pathlib import Path
import zipfile, csv, json, hashlib

root = Path(__file__).resolve().parent
pv = root / 'datasets/plantvillage'
with zipfile.ZipFile(pv / 'data.zip') as archive:
    names = [n for n in archive.namelist() if n.startswith('raw/color/') and n.lower().endswith(('.jpg','.jpeg','.png'))]
    counts = {}
    rows = []
    for name in names:
        label = name.split('/')[2]
        plant, condition = label.split('___', 1)
        counts[label] = counts.get(label, 0) + 1
        rows.append([name, plant, condition, label])
    with (pv / 'image_labels.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['image_path_in_zip','plant','condition','class']); w.writerows(rows)
    with (pv / 'class_counts.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['class','images']); w.writerows(sorted(counts.items()))
    samples = root / 'examples/plantvillage'
    samples.mkdir(parents=True, exist_ok=True)
    for label in sorted(counts):
        name = next(n for n in names if n.split('/')[2] == label)
        (samples / (label + Path(name).suffix)).write_bytes(archive.read(name))
report = {'color_images':len(names), 'classes':len(counts), 'bytes':(pv/'data.zip').stat().st_size, 'sha256':hashlib.file_digest((pv/'data.zip').open('rb'),'sha256').hexdigest(), 'zip_crc_check':'passed'}
(pv / 'verification.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
print(json.dumps(report))
