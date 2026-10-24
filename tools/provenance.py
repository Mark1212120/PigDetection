"""Resolve byte-preserved server split provenance against the equivalent local export."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_training_split(training_sha,dataset):
 current=Path(dataset)/'split_manifest.json'
 if not training_sha or training_sha==digest(current):return
 bridge=ROOT/'experiments/final_results.json'
 if not bridge.is_file():raise ValueError('Checkpoint and evaluation split versions differ')
 b=json.loads(bridge.read_text(encoding='utf-8-sig'))
 if training_sha!=b.get('original_split_sha256') or digest(current)!=b.get('local_split_sha256'):
  raise ValueError('Checkpoint split does not match the audited server/local bridge')
 original=ROOT/'experiments/provenance/server_split_manifest.json'
 a=json.loads(current.read_text(encoding='utf-8-sig'));old=json.loads(original.read_text(encoding='utf-8-sig'))
 if digest(original)!=training_sha or {k:v for k,v in a.items() if k!='raw_root'}!={k:v for k,v in old.items() if k!='raw_root'}:
  raise ValueError('Server/local split membership or image/XML hashes differ')
