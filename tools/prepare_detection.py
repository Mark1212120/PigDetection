"""Export one fixed split to both COCO and YOLO, without modifying raw data."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from runtime import ROOT, sha256, write_json, require_env
require_env()
from PIL import Image

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True, choices=['grouped', 'exploratory', 'smoke'])
    p.add_argument('--groups', type=Path, help='CSV: stem,group,split; every image once; split=train/val/test')
    p.add_argument('--gap', type=int, default=20, help='Exploratory filename-block boundary exclusion per side')
    a = p.parse_args()
    require_env()
    raw = ROOT / 'data/raw/pig_detection'
    stems = sorted((x.stem for x in (raw / 'data').glob('*.jpg')), key=int)
    assert len(stems) == 1015 and all((raw / 'label' / f'{s}.xml').exists() for s in stems)
    if a.gap < 0:
        p.error('--gap must be nonnegative')
    groups_hash = None
    if a.protocol == 'grouped':
        if not a.groups:
            p.error('Grouped protocol requires an inspected source-group CSV; no invented source labels.')
        rows = list(csv.DictReader(a.groups.open(encoding='utf-8-sig', newline='')))
        assert len(rows) == len(stems) and {r['stem'] for r in rows} == set(stems)
        assigned = {}; group_splits = {}
        for r in rows:
            assert r['split'] in ('train', 'val', 'test') and r['group'].strip()
            assert group_splits.setdefault(r['group'], r['split']) == r['split'], 'Group crosses splits'
            assigned[r['stem']] = r['split']
        note = 'User-supplied source groups; provenance and cross-project overlap still require human confirmation.'
        groups_hash = sha256(a.groups)
    elif a.protocol == 'exploratory':
        n = len(stems); b1 = int(n * .7); b2 = int(n * .85)
        assigned = {s: ('train' if i < b1 else 'val' if i < b2 else 'test') for i, s in enumerate(stems)
                    if not (b1-a.gap <= i < b1+a.gap or b2-a.gap <= i < b2+a.gap)}
        note = 'EXPLORATORY same-scene filename blocks with exclusion gaps; filenames are NOT verified time/source groups; semantic leakage and video overlap remain unproven.'
    else:
        assigned = {s: ('train' if i < 2 else 'val' if i < 4 else 'test') for i, s in enumerate(stems[:6])}
        note = 'SMOKE ONLY: six real images, two per split; no reportable accuracy or independence claim.'
    out = ROOT / 'data/processed' / a.protocol
    if out.exists():
        raise SystemExit(f'Refusing to overwrite existing export: {out}')
    assert all(list(assigned.values()).count(split) > 0 for split in ('train', 'val', 'test'))
    out.mkdir(parents=True)
    all_hashes = {}; counts = {}; boxes_total = 0; image_digests = {}
    for split in ('train', 'val', 'test'):
        images = []; annotations = []; split_stems = [s for s in stems if assigned.get(s) == split]
        (out / 'images' / split).mkdir(parents=True)
        (out / 'labels' / split).mkdir(parents=True)
        for s in split_stems:
            source = raw / 'data' / f'{s}.jpg'; xml = raw / 'label' / f'{s}.xml'
            tree = ET.parse(xml).getroot()
            with Image.open(source) as image:
                w, h = image.size
            assert (w, h) == (int(tree.findtext('size/width')), int(tree.findtext('size/height')))
            image_hash = sha256(source)
            assert image_hash not in image_digests, f'Exact duplicate {s} and {image_digests.get(image_hash)}'
            image_digests[image_hash] = s
            all_hashes[s] = dict(image=image_hash, xml=sha256(xml), split=split)
            images.append(dict(id=int(s), file_name=f'{s}.jpg', width=w, height=h))
            lines = []
            for obj in tree.findall('object'):
                assert obj.findtext('name') == 'pig' and obj.findtext('difficult', '0') == '0'
                b = obj.find('bndbox')
                # Explicit VOC 1-based inclusive -> continuous zero-origin xyxy convention.
                x1 = float(b.findtext('xmin')) - 1; y1 = float(b.findtext('ymin')) - 1
                x2 = float(b.findtext('xmax')); y2 = float(b.findtext('ymax'))
                assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, (s, x1, y1, x2, y2)
                bw = x2 - x1; bh = y2 - y1
                annotations.append(dict(id=len(annotations) + 1, image_id=int(s), category_id=1,
                                        bbox=[x1, y1, bw, bh], area=bw * bh, iscrowd=0))
                lines.append(f'0 {(x1+x2)/2/w:.10f} {(y1+y2)/2/h:.10f} {bw/w:.10f} {bh/h:.10f}')
            (out / 'labels' / split / f'{s}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
            dest = out / 'images' / split / source.name
            try:
                os.link(source, dest)
            except OSError:
                shutil.copy2(source, dest)
        write_json(out / 'annotations' / f'{split}.json', dict(info={}, images=images, annotations=annotations,
                                                           categories=[dict(id=1, name='pig')]))
        (out / f'{split}.txt').write_text('\n'.join(split_stems) + '\n', encoding='utf-8')
        counts[split] = dict(images=len(images), boxes=len(annotations))
        boxes_total += len(annotations)
    (out / 'dataset.yaml').write_text(f'path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: pig\n', encoding='utf-8')
    manifest = dict(protocol=a.protocol, note=note, raw_root=str(raw), output=str(out), counts=counts,
                    boxes=boxes_total, groups_csv_sha256=groups_hash, excluded=[s for s in stems if s not in assigned],
                    gap=a.gap if a.protocol == 'exploratory' else None,
                    coordinate_assumption='VOC xmin/ymin minus one; xmax/ymax retained; COCO category=1, YOLO=0',
                    source_hashes=all_hashes)
    write_json(out / 'split_manifest.json', manifest)
    (out/'excluded.txt').write_text('\n'.join(manifest['excluded'])+'\n',encoding='utf-8')
    if a.protocol!='smoke':
        mirror=ROOT/'data/splits'
        mirror.mkdir(parents=True,exist_ok=True)
        for name in ['train.txt','val.txt','test.txt','excluded.txt','split_manifest.json']:
            shutil.copy2(out/name,mirror/name)
    print(json.dumps(dict(path=str(out), counts=counts, note=note), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
