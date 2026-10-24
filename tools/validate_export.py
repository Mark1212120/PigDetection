"""Verify disjoint split hashes, all COCO/YOLO boxes, and immutable raw annotations."""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET
from runtime import ROOT,require_env,sha256,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=ROOT/'data/processed/exploratory')
    a=p.parse_args();require_env();d=a.dataset.resolve();m=json.loads((d/'split_manifest.json').read_text())
    seen=set();hashes=set();boxes=0
    for split in ['train','val','test']:
        c=json.loads((d/'annotations'/f'{split}.json').read_text());by_id={r['id']:[] for r in c['images']}
        for r in c['annotations']:by_id[r['image_id']].append(r['bbox'])
        for image in c['images']:
            stem=Path(image['file_name']).stem;assert stem not in seen;seen.add(stem)
            entry=m['source_hashes'][stem];assert entry['split']==split
            image_hash=sha256(d/'images'/split/image['file_name']);assert image_hash==entry['image'] and image_hash not in hashes;hashes.add(image_hash)
            xml=Path(m['raw_root'])/'label'/f'{stem}.xml';assert sha256(xml)==entry['xml']
            labels=[list(map(float,l.split())) for l in (d/'labels'/split/f'{stem}.txt').read_text().splitlines()]
            raw=ET.parse(xml).getroot().findall('object')
            assert len(raw)==len(labels)==len(by_id[image['id']])
            for obj,yolo,coco in zip(raw,labels,by_id[image['id']]):
                x,y,w,h=coco;b=obj.find('bndbox')
                expected=[float(b.findtext('xmin'))-1,float(b.findtext('ymin'))-1,
                          float(b.findtext('xmax'))-float(b.findtext('xmin'))+1,
                          float(b.findtext('ymax'))-float(b.findtext('ymin'))+1]
                assert coco==expected and yolo[0]==0
                transformed=[(x+w/2)/image['width'],(y+h/2)/image['height'],w/image['width'],h/image['height']]
                assert max(abs(a-b) for a,b in zip(transformed,yolo[1:]))<1e-8
                boxes+=1
    assert len(seen)==sum(r['images'] for r in m['counts'].values())
    assert not seen.intersection(m['excluded']) and len(seen)+len(m['excluded'])==1015
    write_json(ROOT/'experiments/provenance/dataset_export_validation.json',dict(status='passed',dataset=str(d),
        split_sha256=sha256(d/'split_manifest.json'),images=len(seen),boxes=boxes,excluded=len(m['excluded']),
        checks=['disjoint IDs and exact image hashes','all VOC/COCO/YOLO boxes equivalent','raw XML hashes unchanged']))
    print(f'Export checked: {len(seen)} images, {boxes} boxes, {len(m["excluded"])} excluded')
if __name__=='__main__':main()
