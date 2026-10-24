"""Runtime bookkeeping shared by command-line entries."""
from pathlib import Path
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'experiments/environment/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'experiments/environment/matplotlib'))
os.environ.setdefault('TORCH_HOME', 'D:/Python_Resources/Models/torch')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def environment():
    import torch
    import importlib.metadata as md
    return dict(python=sys.executable, utc=datetime.now(timezone.utc).isoformat(),
                versions={n: md.version(n) for n in ['torch', 'torchvision', 'ultralytics', 'numpy']},
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

def require_env(name='pigdet'):
    expected = Path('D:/Python_Resources/Envs') / name
    if Path(sys.prefix).resolve() != expected.resolve() and Path(sys.prefix).name.lower()!=name.lower():
        raise SystemExit(f'Activate {name} in your VSCode terminal first. Current Python: {sys.executable}')

def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def source_hashes():
    records = {p.relative_to(ROOT).as_posix():sha256(p)
            for folder in ['datasets','models','scripts','evaluation','tools']
            for p in (ROOT/folder).rglob('*.py')
            if 'third_party' not in p.parts and '__pycache__' not in p.parts}
    if (ROOT/'train.py').is_file():
        records['train.py']=sha256(ROOT/'train.py')
    return records
