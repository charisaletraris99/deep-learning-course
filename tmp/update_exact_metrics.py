from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import html
import re
import shutil

root = Path(r'C:\Users\Charis Local\Documents\University Of Cyprus PHD\deep-learning-course')
docx = root / 'assignment-2' / 'Exercise_1_Report.docx'
temp_dir = root / 'tmp'
temp_dir.mkdir(exist_ok=True)
backup = temp_dir / 'Exercise_1_Report_before_exact_metrics.docx'
out = temp_dir / 'Exercise_1_Report_updated.docx'
shutil.copy2(docx, backup)

with ZipFile(docx, 'r') as zin:
    files = {info.filename: zin.read(info.filename) for info in zin.infolist()}

xml = files['word/document.xml']

def replace_once(blob, old, new):
    oldb, newb = old.encode('utf-8'), new.encode('utf-8')
    count = blob.count(oldb)
    if count != 1:
        raise RuntimeError(f'Expected exactly one match for: {old[:60]!r}; found {count}')
    return blob.replace(oldb, newb, 1)

paragraph_updates = {
    'During roughly the first six epochs, all three losses fall. This means that the model improves on both the training data and unseen images. Around epoch 10, validation and test loss is near its lowest point, at about 0.22 and 0.26 respectively (is lower at epochs 6 where it is 0.21 and 0.25 but its accuracy is also lower).':
    'During roughly the first six epochs, all three losses fall. This means that the model improves on both the training data and unseen images. At epoch 10, validation loss is 0.2251 and test loss is 0.2647. Their lowest values occur at epoch 6: 0.2151 validation loss and 0.2479 test loss. I use epoch 10 in the discussion because it has better accuracy with a similar validation loss.',
    'After this point, training loss continues to fall, but validation and test loss generally rise, with some short fluctuations. By epoch 40, training loss is about 0.03, validation loss about 0.46 and test loss about 0.53. The model fits the training examples more closely while its probability predictions on unseen images get worse. This is evidence of overfitting from around epoch 10.':
    'After this point, training loss continues to fall, but validation and test loss generally rise, with some short fluctuations. By epoch 40, training loss is 0.0242, validation loss is 0.4837, and test loss is 0.5514. The model fits the training examples more closely while its probability predictions on unseen images get worse. This is evidence of overfitting from around epoch 10.',
    'The diagonal contains correct predictions. Adding its values gives 9,113 correct predictions out of 10,000 test images, or 91.13 percent accuracy. This matrix describes the selected model, not necessarily the model at epoch 40.':
    'The diagonal contains correct predictions. Adding its values gives 9,114 correct predictions out of 10,000 test images, or 91.14 percent accuracy. This matrix describes the selected model, not necessarily the model at epoch 40.',
    'Bags, sneakers, sandals and trousers are classified well: 990, 987, 986 and 983 correct predictions out of 1,000 for each class. Shirts are harder, with 730 correct predictions. The model predicts 133 shirts as T-shirts/tops. These errors are easily explained because of the similar clothing shapes.':
    'Bags, sneakers, sandals and trousers are classified well: 990, 987, 986 and 983 correct predictions out of 1,000 for each class. Shirts are harder, with 731 correct predictions. The model predicts 133 shirts as T-shirts/tops. These errors are easily explained because of the similar clothing shapes.',
    'Training settings. The residual CNN uses the same 40 epochs, batch size 128, Adam learning rate 0.001, data split, seed and the same trainable parameter as Exercise 1.':
    'Training settings. The residual CNN uses the same 40 epochs, batch size 128, Adam learning rate 0.001, data split, seed and the same number of trainable parameters as Exercise 1. Training and per-epoch test evaluation took 971.5 seconds for Exercise 1 and 798.1 seconds for the residual CNN.',
    'Comparison with Exercise 1. The two models achieve very similar results. The residual CNN delays the loss-based overfitting pattern by about two epochs, from after epoch 10 to after epoch 12. However, it has slightly lower mean accuracy and lower highest accuracy than the Exercise 1 CNN. The improvement is therefore a small delay in overfitting, not a clear accuracy improvement.':
    'Comparison with Exercise 1. The two models achieve very similar results. The residual CNN delays the loss-based overfitting pattern by about two epochs, from after epoch 10 to after epoch 12. Across all epochs, it has higher mean and highest validation accuracy. However, the Exercise 1 CNN has slightly higher mean and selected test accuracy and a slightly lower mean test loss. The improvement is therefore a small delay in overfitting, not a clear overall test-accuracy improvement.',
    'This confusion matrix shows which clothing classes the selected residual CNN classifies correctly and which classes it confuses. Both models confuse shirts with T-shirts or tops, which is expected because these categories have similar shapes. The residual CNN also has difficulty separating pullovers, coats, and shirts. This is similar to the Exercise 1 confusion matrix, so the residual shortcuts do not remove the difficult clothing-class confusions. The shirt result is noticeably better with residual connections: the residual CNN correctly classifies 769 of 1,000 shirts, compared with 730 of 1,000 shirts in Exercise 1. This is 39 more correct shirt predictions.':
    'This confusion matrix shows which clothing classes the selected residual CNN classifies correctly and which classes it confuses. Both models confuse shirts with T-shirts or tops, which is expected because these categories have similar shapes. The residual CNN also has difficulty separating pullovers, coats, and shirts. This is similar to the Exercise 1 confusion matrix, so the residual shortcuts do not remove the difficult clothing-class confusions. The shirt result is noticeably better with residual connections: the residual CNN correctly classifies 769 of 1,000 shirts, compared with 731 of 1,000 shirts in Exercise 1. This is 38 more correct shirt predictions.',
}
for old, new in paragraph_updates.items():
    xml = replace_once(xml, old, new)

# Find the comparison table by its header, then change only the two numeric value cells in each named row.
metric_pos = xml.find(b'>Metric</w:t>')
if metric_pos < 0:
    raise RuntimeError('Comparison table header was not found')
table_start = xml.rfind(b'<w:tbl', 0, metric_pos)
table_end = xml.find(b'</w:tbl>', metric_pos) + len(b'</w:tbl>')
table = xml[table_start:table_end]

def update_cell_text(cell, new_text):
    pattern = re.compile(rb'(<w:t(?:\s[^>]*)?>)(.*?)(</w:t>)')
    matches = list(pattern.finditer(cell))
    if not matches:
        raise RuntimeError('Cell has no text nodes')
    pieces=[]
    cursor=0
    for index, m in enumerate(matches):
        pieces.append(cell[cursor:m.start()])
        content = html.escape(new_text).encode('utf-8') if index == 0 else b''
        pieces.append(m.group(1) + content + m.group(3))
        cursor=m.end()
    pieces.append(cell[cursor:])
    return b''.join(pieces)

def update_table_row(table_blob, label, first_value, second_value):
    label_b = label.encode('utf-8')
    row_pattern = re.compile(rb'<w:tr(?:\s[^>]*)?>.*?</w:tr>')
    found = []
    for m in row_pattern.finditer(table_blob):
        if label_b in m.group(0):
            found.append(m)
    if len(found) != 1:
        raise RuntimeError(f'Expected one row for {label!r}; found {len(found)}')
    m = found[0]
    row = m.group(0)
    cells = list(re.finditer(rb'<w:tc(?:\s[^>]*)?>.*?</w:tc>', row))
    if len(cells) != 3:
        raise RuntimeError(f'Expected three cells for {label!r}; found {len(cells)}')
    rebuilt=[]
    cursor=0
    for index, cell_match in enumerate(cells):
        rebuilt.append(row[cursor:cell_match.start()])
        cell=cell_match.group(0)
        if index == 1:
            cell=update_cell_text(cell, first_value)
        elif index == 2:
            cell=update_cell_text(cell, second_value)
        rebuilt.append(cell)
        cursor=cell_match.end()
    rebuilt.append(row[cursor:])
    new_row=b''.join(rebuilt)
    return table_blob[:m.start()] + new_row + table_blob[m.end():]

metrics = [
    ('Mean validation accuracy', '92.31%', '92.41%'),
    ('Highest validation accuracy', '92.93%', '93.10%'),
    ('Mean validation loss', '0.3290', '0.3236'),
    ('Lowest validation loss', '0.2151', '0.2220'),
    ('Mean test accuracy', '91.36%', '91.30%'),
    ('Highest test accuracy', '92.11%', '91.96%'),
    ('Mean test loss', '0.3793', '0.3868'),
    ('Lowest test loss', '0.2479', '0.2574'),
    ('Selected test accuracy', '91.14%', '90.81%'),
]
for label, baseline, residual in metrics:
    table = update_table_row(table, label, baseline, residual)
xml = xml[:table_start] + table + xml[table_end:]
files['word/document.xml'] = xml
files['word/media/image1.png'] = (root / 'assignment-2' / 'outputs' / 'exercise_1_three_conv_gelu' / 'learning_curves.png').read_bytes()
files['word/media/image2.png'] = (root / 'assignment-2' / 'outputs' / 'exercise_1_three_conv_gelu' / 'confusion_matrix.png').read_bytes()

with ZipFile(out, 'w', ZIP_DEFLATED) as zout:
    for name, data in files.items():
        zout.writestr(name, data)

# Confirm the result is a readable DOCX archive and then atomically replace the report.
with ZipFile(out, 'r') as check:
    if check.testzip() is not None:
        raise RuntimeError('The updated DOCX archive is corrupt')
    if check.read('word/media/image1.png') != files['word/media/image1.png']:
        raise RuntimeError('Learning curves image was not embedded')
    if check.read('word/media/image2.png') != files['word/media/image2.png']:
        raise RuntimeError('Confusion matrix image was not embedded')
shutil.copy2(out, docx)
print('Updated:', docx)
print('Backup:', backup)
