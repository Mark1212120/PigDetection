"""Identical COCOeval settings for all six methods; no per-model AP shortcuts."""
from contextlib import redirect_stdout
import io
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(gt_path, predictions, image_ids=None):
    gt = COCO(str(gt_path))
    rows = json.loads(open(predictions, encoding='utf-8').read()) if isinstance(predictions, (str, bytes)) or hasattr(predictions, 'read_text') else predictions
    if rows:
        dt = gt.loadRes(rows)
    else:
        dt = COCO(); dt.dataset = dict(images=gt.dataset['images'], categories=gt.dataset['categories'], annotations=[]); dt.createIndex()
    evaluator = COCOeval(gt, dt, 'bbox')
    evaluator.params.catIds = [1]
    evaluator.params.maxDets = [1, 10, 100]
    if image_ids is not None:
        evaluator.params.imgIds = image_ids
    evaluator.evaluate(); evaluator.accumulate()
    with redirect_stdout(io.StringIO()):
        evaluator.summarize()
    keys = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
    result = {k: float(v) for k, v in zip(keys, evaluator.stats)}
    result['units'] = 'fractions 0..1; -1 means undefined; COCO maxDets=100'
    precision = evaluator.eval['precision'][:, :, 0, 0, -1]
    valid = precision[precision >= 0]
    assert not len(valid) or np.isfinite(valid).all()
    return result
