import os
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root.parent / 'Complete-MLOps-Pipeline-clean'
if out.exists():
    shutil.rmtree(out)
out.mkdir()

KEEP_DIRS = ['configs', 'src', 'tests', '.github']
KEEP_FILES = ['README.md', 'requirements.txt', '.gitignore']

for d in KEEP_DIRS:
    src = root / d
    if src.exists():
        shutil.copytree(src, out / d)

for f in KEEP_FILES:
    src = root / f
    if src.exists():
        shutil.copy2(src, out / f)

# Copy DVC pointer files and .dvc/config
data_dir = root / 'data'
out_data = out / 'data'
out_data.mkdir(exist_ok=True)
for f in data_dir.glob('*.dvc'):
    shutil.copy2(f, out_data / f.name)

_dvc = root / '.dvc'
if _dvc.exists():
    cfg = _dvc / 'config'
    if cfg.exists():
        out_dvc = out / '.dvc'
        out_dvc.mkdir(exist_ok=True)
        shutil.copy2(cfg, out_dvc / 'config')

print('Exported clean repo to', out)
