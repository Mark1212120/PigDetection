"""Train one of six detectors in the already activated pigdet environment."""
from pathlib import Path
from copy import deepcopy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.runtime import ROOT, require_env, resolve, sha256, write_json, environment, source_hashes
from tools.training_guard import fixed_batch_trainer,install_batch_guard,completed_training
require_env()
import argparse
import csv
import json
import math
import random
import shutil
import time
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from datasets.pig import PigDataset, collate
from models.detectors import METHODS, build_native, Predictor
from evaluation.coco_metrics import evaluate

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--dataset', default='data/processed/exploratory')
    p.add_argument('--name', required=True, help='New run identifier; use --output to choose a records directory')
    p.add_argument('--output',help='Explicit new output directory; defaults to experiments/work_dirs/NAME')
    p.add_argument('--epochs', type=int)
    p.add_argument('--imgsz', type=int)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-steps', type=int, default=0, help='Native adapters only; marks the run SMOKE')
    p.add_argument('--limit', type=int, default=0, help='Native train/val image limit; SMOKE only')
    a = p.parse_args(); require_env()
    c = yaml.safe_load(resolve(a.config).read_text(encoding='utf-8'))
    method = c['method']; assert method in METHODS
    if a.epochs is not None: c['epochs'] = a.epochs
    if a.imgsz is not None: c['imgsz'] = a.imgsz
    assert c['epochs'] > 0 and c['imgsz'] >= 128 and c['imgsz'] % 32 == 0
    assert a.limit >= 0 and a.max_steps >= 0
    if method in ('yolov8s', 'rtdetr_l') and (a.max_steps or a.limit):
        p.error('Ultralytics smoke: use --dataset data/processed/smoke --epochs 1 instead of --limit/--max-steps')
    assert Path(a.name).name == a.name and a.name not in ('.', '..')
    out = resolve(a.output) if a.output else ROOT / 'experiments/work_dirs' / a.name
    if out.exists(): raise SystemExit(f'Run exists: {out}; choose another --name')
    data = resolve(a.dataset)
    split_path = data / 'split_manifest.json'
    split = json.loads(split_path.read_text(encoding='utf-8'))
    weights = resolve(c['pretrained'])
    weights_manifest = json.loads((ROOT / 'models/pretrained/weights_manifest.json').read_text())
    assert sha256(weights) == weights_manifest[method]['sha256'], 'Initialization weight hash changed'
    # Refuse stale labels/images before starting a potentially long job.
    for stem, entry in split['source_hashes'].items():
        assert sha256(Path(split['raw_root']) / 'label' / f'{stem}.xml') == entry['xml'], f'Stale XML: {stem}'
        assert sha256(data / 'images' / entry['split'] / f'{stem}.jpg') == entry['image'], f'Stale image: {stem}'
    seed = c['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    smoke = split['protocol'] == 'smoke' or bool(a.limit or a.max_steps)
    if smoke:
        # A two-image smoke epoch must actually execute optimizer.step().
        # Ultralytics otherwise may carry its accumulation window past
        # the end of the tiny run without an optimizer update.
        c['effective_batch']=c['batch']
    manifest = dict(status='started', environment=environment(), source_sha256=source_hashes(), method=method, config=c,
                    arguments=vars(a), dataset=str(data), split_sha256=sha256(split_path),
                    protocol=split['protocol'], protocol_note=split['note'], smoke_only=smoke,
                    pretrained=str(weights), pretrained_sha256=sha256(weights), output=str(out),
                    class_names=['pig'], deterministic='seeded; architecture-specific CUDA operations may be nondeterministic',
                    fairness='shared splits/epochs/nominal size; native architectural preprocessing retained; NOT original paper recipes')
    manifest['smoke_budget_note']='Smoke uses effective_batch=batch so optimizer updates are exercised; formal effective_batch is read from the configuration.' if smoke else None
    out.mkdir(parents=True)
    write_json(out / 'run_manifest.json', manifest)
    try:
        if method in ('yolov8s', 'rtdetr_l'):
            from ultralytics import YOLO, RTDETR
            wrapper = (YOLO if method == 'yolov8s' else RTDETR)(str(weights))
            from ultralytics.models.yolo.detect import DetectionTrainer
            from ultralytics.models.rtdetr.train import RTDETRTrainer
            trainer_class=fixed_batch_trainer(DetectionTrainer if method=='yolov8s' else RTDETRTrainer,
                c['batch'],c.get('validation_batch',1))
            install_batch_guard(wrapper,c['batch'])
            unified_best=[-math.inf]
            def select_by_coco(trainer):
                # Native Ultralytics fitness differs from pycocotools interpolation.
                # Select the six methods using the SAME validation evaluator.
                from scripts.export_detection import predict_split
                before=wrapper.model
                # Ultralytics predict may fuse conv/BN in place. Never fuse the
                # live EMA: the next epoch must still match the training state.
                wrapper.model=deepcopy(trainer.ema.ema if trainer.ema else trainer.model).float().eval()
                wrapper.predictor=None
                adapter=Predictor.__new__(Predictor)
                adapter.method=method;adapter.model=wrapper;adapter.device=torch.device(a.device)
                adapter.size=c['imgsz'];adapter.conf=.001;adapter.iou=.7
                try:
                    predictions,ids,timing=predict_split(adapter,data,'val')
                    metrics=evaluate(data/'annotations/val.json',predictions,ids)
                    write_json(out/'val_predictions.json',predictions)
                    write_json(out/'val_metrics.json',dict(epoch=trainer.epoch+1,metrics=metrics,smoke_only=smoke,timing=timing))
                    with (out/'unified_validation.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(dict(epoch=trainer.epoch+1,metrics=metrics))+'\n')
                    if metrics['AP']>unified_best[0]:
                        unified_best[0]=metrics['AP']
                        assert trainer.last.is_file(),'Epoch checkpoint missing'
                        shutil.copy2(trainer.last,trainer.last.parent/'best_coco.pt')
                finally:
                    wrapper.predictor=None;wrapper.model=before
            wrapper.add_callback('on_fit_epoch_end',select_by_coco)
            # No implicit AMP model-check downloads; no stochastic augmentation.
            wrapper.train(trainer=trainer_class,data=str(data / 'dataset.yaml'), epochs=c['epochs'], imgsz=c['imgsz'],
                          batch=c['batch'], nbs=c['effective_batch'], workers=c.get('workers',0), device=a.device,
                          optimizer='AdamW', lr0=c['lr'], lrf=.01, momentum=.9, weight_decay=c['weight_decay'],
                          warmup_epochs=0, cos_lr=True, seed=seed, deterministic=False,
                          amp=False, patience=0, cache=False, pretrained=True,
                          mosaic=0., mixup=0., copy_paste=0., hsv_h=0., hsv_s=0., hsv_v=0.,
                          degrees=0., translate=0., scale=0., shear=0., perspective=0.,
                          flipud=0., fliplr=0., close_mosaic=0, plots=False,
                          project=str(out), name='engine', exist_ok=False)
            manifest['actual_training']=completed_training(wrapper.trainer,c)
            best = out / 'engine/weights/best_coco.pt'
            assert best.is_file()
            manifest['checkpoint'] = str(best)
            manifest['selection'] = 'shared pycocotools validation AP50:95 via on_fit_epoch_end; best_coco.pt; engine best.pt is upstream fitness only'
            manifest['engine_settings'] = str(out / 'engine/args.yaml')
        else:
            device = torch.device(a.device)
            train = PigDataset(data, 'train', method, c['imgsz'], a.limit)
            loader = DataLoader(train,batch_size=c['batch'],shuffle=True,num_workers=c.get('workers',0),collate_fn=collate)
            model, loading = build_native(method, c['imgsz'], device, weights, initialize=True)
            manifest['initialization'] = loading
            optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c['epochs'], eta_min=c['lr'] * .01)
            accumulation = max(1, c['effective_batch'] // c['batch'])
            manifest['actual_effective_batch'] = c['batch'] * accumulation
            manifest['selection'] = 'validation COCO AP50:95; no test-set checkpoint selection'
            write_json(out / 'run_manifest.json', manifest)
            best_ap = -math.inf; step = 0
            with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
                log = csv.DictWriter(f, fieldnames=['epoch', 'steps', 'train_loss', 'val_AP', 'lr', 'seconds'])
                log.writeheader()
                for epoch in range(c['epochs']):
                    started = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
                    losses = []
                    for index, (images, targets) in enumerate(loader):
                        if method == 'yolox_s':
                            loss_dict = model(torch.stack(images).to(device), torch.stack(targets).to(device))
                            loss = loss_dict['total_loss']
                        else:
                            images = [x.to(device) for x in images]
                            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                            loss_dict = model(images, targets); loss = sum(loss_dict.values())
                        if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss: {loss_dict}')
                        # Exact mean over each accumulation group, including the final short group.
                        group_size = min(accumulation, len(loader) - (index // accumulation) * accumulation)
                        if a.max_steps:
                            group_size = min(group_size, a.max_steps - (step - index % accumulation))
                        (loss / group_size).backward()
                        losses.append(float(loss.detach())); step += 1
                        if (index + 1) % accumulation == 0 or index + 1 == len(loader) or (a.max_steps and step >= a.max_steps):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                            optimizer.step(); optimizer.zero_grad(set_to_none=True)
                        print(f'{method} epoch={epoch+1} step={step} loss={losses[-1]:.5f}', flush=True)
                        if a.max_steps and step >= a.max_steps: break
                    checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                      scheduler=scheduler.state_dict(), method=method, class_names=['pig'],
                                      epoch=epoch+1, config=c, smoke_only=smoke,
                                      split_sha256=manifest['split_sha256'])
                    torch.save(checkpoint, out / 'last.pt')
                    # Reuse the trained model for validation; no extra GPU model copy.
                    predictor = Predictor.__new__(Predictor)
                    predictor.method=method; predictor.model=model.eval(); predictor.size=c['imgsz']
                    predictor.device=device; predictor.conf=.001; predictor.iou=.7
                    from scripts.export_detection import predict_split
                    predictions, ids, timing = predict_split(predictor, data, 'val', a.limit)
                    write_json(out / 'val_predictions.json', predictions)
                    metrics = evaluate(data / 'annotations/val.json', predictions, ids)
                    write_json(out / 'val_metrics.json', dict(metrics=metrics, smoke_only=smoke, timing=timing))
                    if metrics['AP'] > best_ap:
                        best_ap = metrics['AP']; torch.save(checkpoint, out / 'best.pt')
                    lr = optimizer.param_groups[0]['lr']; scheduler.step()
                    log.writerow(dict(epoch=epoch+1, steps=step, train_loss=float(np.mean(losses)),
                                      val_AP=metrics['AP'], lr=lr, seconds=time.perf_counter()-started)); f.flush()
                    if a.max_steps and step >= a.max_steps: break
            manifest['checkpoint'] = str(out / 'best.pt')
            manifest['actual_training']=dict(batch=c['batch'],effective_batch=c['batch']*accumulation,
                validation_batch=1,epochs_completed=epoch+1,epochs_requested=c['epochs'],seed=seed,patience=0)
            if not smoke and epoch+1!=c['epochs']:raise ValueError('Native training stopped before its fixed epoch budget')
        manifest['checkpoint_sha256'] = sha256(manifest['checkpoint'])
        manifest['status'] = 'completed'; write_json(out / 'run_manifest.json', manifest)
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed', error=repr(exc)); write_json(out / 'run_manifest.json', manifest)
        raise
    print(f'Completed: {out}; smoke_only={smoke}', flush=True)

if __name__ == '__main__': main()
