"""Export pig checkpoint predictions in original-image COCO coordinates."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.runtime import ROOT, require_env, resolve, sha256, write_json, environment
require_env()
import argparse
import json
import time
import cv2
import torch
from models.detectors import METHODS, Predictor
from evaluation.coco_metrics import evaluate

def predict_split(predictor, data, split, limit=0):
    records = json.loads((data / 'annotations' / f'{split}.json').read_text())['images'][:limit or None]
    predictions=[]; times=[]
    for index, record in enumerate(records):
        image=cv2.imread(str(data / 'images' / split / record['file_name']))
        if image is None: raise RuntimeError(record)
        if predictor.device.type == 'cuda': torch.cuda.synchronize()
        begin=time.perf_counter(); rows=predictor(image)
        if predictor.device.type == 'cuda': torch.cuda.synchronize()
        elapsed=time.perf_counter()-begin
        if index >= 1: times.append(elapsed)
        for x1,y1,x2,y2,score in rows:
            predictions.append(dict(image_id=record['id'], category_id=1,
                                    bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],score=float(score)))
    return predictions, [r['id'] for r in records], dict(measured_images=len(times), excluded_warmup=1,
        mean_ms=1000*sum(times)/len(times) if times else None,
        scope='batch=1 adapter preprocessing+model+postprocessing; excludes disk decode; not standardized paper FPS')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',required=True,choices=METHODS); p.add_argument('--weights',required=True)
    p.add_argument('--dataset',default='data/processed/exploratory');p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();require_env();out=resolve(a.output)
    if out.exists():raise SystemExit(f'Choose a new output: {out}')
    data=resolve(a.dataset); weights=resolve(a.weights); split=json.loads((data/'split_manifest.json').read_text())
    predictor=Predictor(a.method,weights,a.device,a.imgsz)
    if predictor.training_split_sha256:
        from tools.provenance import validate_training_split
        validate_training_split(predictor.training_split_sha256,data)
    rows,ids,timing=predict_split(predictor,data,a.split,a.limit)
    metrics=evaluate(data/'annotations'/f'{a.split}.json',rows,ids)
    write_json(out/'predictions.json',rows)
    write_json(out/'metrics.json',dict(method=a.method,split=a.split,metrics=metrics,timing=timing,
        smoke_only=bool(a.limit or split['protocol']=='smoke' or predictor.smoke_only),protocol=split['protocol'],protocol_note=split['note'],
        environment=environment(),weights=str(weights),weights_sha256=sha256(weights),
        split_sha256=sha256(data/'split_manifest.json'),arguments=vars(a)))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
