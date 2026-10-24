"""Explicit, resumable-by-file official checkpoint download. No environment changes."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'fasterrcnn_r50_fpn': ('fasterrcnn_resnet50_fpn_coco-258fb6c6.pth', 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'),
    'retinanet_r50_fpn': ('retinanet_resnet50_fpn_coco-eeacb38b.pth', 'https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth'),
    'fcos_r50_fpn': ('fcos_resnet50_fpn_coco-99b0c9b7.pth', 'https://download.pytorch.org/models/fcos_resnet50_fpn_coco-99b0c9b7.pth'),
    'yolox_s': ('yolox_s.pth', 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth'),
    'yolov8s': ('yolov8s.pt', 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt'),
    'rtdetr_l': ('rtdetr-l.pt', 'https://github.com/ultralytics/assets/releases/download/v8.4.0/rtdetr-l.pt'),
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['all', *SOURCES], default='all')
    args = parser.parse_args()
    target = ROOT / 'models/pretrained'
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / 'weights_manifest.json'
    records = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, (filename, url) in SOURCES.items():
        if args.method not in ('all', name):
            continue
        path = target / filename
        if path.exists():
            if name not in records or sha256(path) != records[name]['sha256']:
                raise RuntimeError(f'Unverified existing file: {path}; inspect before replacing.')
        else:
            temp = path.with_suffix(path.suffix + '.part')
            for attempt in range(3):
                try:
                    print(f'Downloading {name}: {url}', flush=True)
                    request = urllib.request.Request(url, headers={'User-Agent': 'PigExperiment/1.0'})
                    with urllib.request.urlopen(request, timeout=90) as response, temp.open('wb') as out:
                        expected = response.headers.get('Content-Length')
                        resolved_url = response.url
                        for block in iter(lambda: response.read(1024 * 1024), b''):
                            out.write(block)
                    if temp.stat().st_size < 1000000 or (expected and temp.stat().st_size != int(expected)):
                        raise RuntimeError('Truncated or non-checkpoint download')
                    digest = sha256(temp)
                    if name in ('fasterrcnn_r50_fpn', 'retinanet_r50_fpn', 'fcos_r50_fpn'):
                        assert digest.startswith(filename.rsplit('-', 1)[1].split('.')[0])
                    temp.replace(path)
                    records[name] = dict(path=str(path), source=url, resolved_url=resolved_url,
                                         bytes=path.stat().st_size, sha256=digest,
                                         purpose='COCO pretrained initialization; not pig-finetuned',
                                         downloaded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                    manifest_path.write_text(json.dumps(records, indent=2), encoding='utf-8')
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        print(f'Verified {name}: {path.stat().st_size} bytes {sha256(path)}', flush=True)

if __name__ == '__main__':
    main()
