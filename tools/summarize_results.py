"""Collect actual unified COCOeval results, refusing duplicate method rows."""
from pathlib import Path
import argparse
import csv
import json
from runtime import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True,nargs='+')
    p.add_argument('--output',type=Path,default=ROOT/'experiments/tables/detection.csv');a=p.parse_args()
    rows=[]; methods=set();protocols=set();splits=set()
    for path in a.results:
        r=json.loads(path.read_text(encoding='utf-8'))
        assert not r['smoke_only'],'Smoke results cannot be placed in the formal comparison table'
        assert r['method'] not in methods;methods.add(r['method']);protocols.add(r['split_sha256']);splits.add(r['split'])
        rows.append(dict(method=r['method'],AP=r['metrics']['AP'],AP50=r['metrics']['AP50'],AP75=r['metrics']['AP75'],
                         AR100=r['metrics']['AR100'],mean_ms=r['timing']['mean_ms'],source=str(path.resolve())))
    assert len(protocols)==1 and len(splits)==1, 'Mixed split versions or val/test results'
    assert not a.output.exists(), 'Choose a new output file'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(a.output)

if __name__=='__main__':main()
