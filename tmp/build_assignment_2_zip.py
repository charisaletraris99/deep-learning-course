from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib

root=Path(r'C:\Users\Charis Local\Documents\University Of Cyprus PHD\deep-learning-course')
assignment=root/'assignment-2'
output=assignment/'assignment_2_submission.zip'
files={
    'assignment_2_submission/Assignment_2.docx': assignment/'Assignment_2.docx',
    'assignment_2_submission/exercise_1/prepare_data.py': assignment/'excercise 1'/'prepare_data.py',
    'assignment_2_submission/exercise_1/train_cnn.py': assignment/'excercise 1'/'train_cnn.py',
    'assignment_2_submission/exercise_1/results/learning_curves.png': assignment/'outputs'/'exercise_1_three_conv_gelu'/'learning_curves.png',
    'assignment_2_submission/exercise_1/results/confusion_matrix.png': assignment/'outputs'/'exercise_1_three_conv_gelu'/'confusion_matrix.png',
    'assignment_2_submission/exercise_2/prepare_data.py': assignment/'excercise 2'/'prepare_data.py',
    'assignment_2_submission/exercise_2/train_residual.py': assignment/'excercise 2'/'train_residual.py',
    'assignment_2_submission/exercise_2/results/learning_curves.png': assignment/'outputs'/'exercise_2_residual_cnn_gelu'/'learning_curves.png',
    'assignment_2_submission/exercise_2/results/confusion_matrix.png': assignment/'outputs'/'exercise_2_residual_cnn_gelu'/'confusion_matrix.png',
}
missing=[str(path) for path in files.values() if not path.is_file()]
if missing:
    raise FileNotFoundError('Missing files: '+', '.join(missing))
temporary=output.with_suffix('.zip.tmp')
with ZipFile(temporary,'w',ZIP_DEFLATED) as archive:
    for arcname, source in files.items():
        archive.write(source, arcname)
temporary.replace(output)
with ZipFile(output) as archive:
    if archive.testzip() is not None:
        raise RuntimeError('ZIP integrity check failed')
    names=archive.namelist()
    if names != list(files):
        raise RuntimeError('Unexpected ZIP contents: '+repr(names))
    for arcname, source in files.items():
        if hashlib.sha256(archive.read(arcname)).digest() != hashlib.sha256(source.read_bytes()).digest():
            raise RuntimeError('Checksum mismatch: '+arcname)
print(output)
print('Files:', len(files))
print('Size:', output.stat().st_size, 'bytes')
for item in names:
    print(item)
