"""Verify this project's runtime without downloading weights or training models."""
from pathlib import Path
import gc
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'experiments/environment/ultralytics'))
os.environ.setdefault('TORCH_HOME', 'D:/Python_Resources/Models/torch')
os.environ.setdefault('HF_HOME', 'D:/Python_Resources/Cache/huggingface')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'experiments/environment/matplotlib'))


def main():
    import cv2
    import numpy as np
    import torch
    import torchvision
    from torchvision.ops import nms
    from ultralytics import YOLO, RTDETR
    from yolox.exp import get_exp

    assert Path(sys.prefix).resolve() == Path('D:/Python_Resources/Envs/pigdet').resolve()
    assert torch.cuda.is_available(), 'CUDA unavailable'
    device = torch.device('cuda:0')
    image = cv2.imread(str(ROOT / 'data/raw/pig_detection/data/1.jpg'))
    assert image is not None and image.shape == (720, 1280, 3)
    tensor = torch.ones((128, 128), device=device)
    assert (tensor @ tensor).mean().item() == 128
    assert nms(torch.tensor([[0., 0., 10., 10.]], device=device),
               torch.tensor([0.9], device=device), 0.5).tolist() == [0]
    assert torch.from_numpy(np.ones(2)).numel() == 2

    outcomes = []
    constructors = [
        ('fasterrcnn_r50_fpn', torchvision.models.detection.fasterrcnn_resnet50_fpn),
        ('retinanet_r50_fpn', torchvision.models.detection.retinanet_resnet50_fpn),
        ('fcos_r50_fpn', torchvision.models.detection.fcos_resnet50_fpn),
    ]
    for name, constructor in constructors:
        model = constructor(weights=None, weights_backbone=None, num_classes=2,
                            min_size=128, max_size=128).to(device).eval()
        with torch.inference_mode():
            result = model([torch.zeros(3, 128, 128, device=device)])[0]
        assert set(['boxes', 'scores', 'labels']).issubset(result)
        assert torch.isfinite(result['boxes']).all()
        outcomes.append(dict(method=name, check='random-weight CUDA forward', passed=True))
        print(name + ': CUDA forward passed', flush=True)
        del model, result
        gc.collect(); torch.cuda.empty_cache()

    exp = get_exp(exp_name='yolox-s')
    exp.num_classes = 1
    model = exp.get_model().to(device).eval()
    with torch.inference_mode():
        result = model(torch.zeros(1, 3, 128, 128, device=device))
    assert result.shape[-1] == 6 and torch.isfinite(result).all()
    outcomes.append(dict(method='yolox_s', check='random-weight CUDA forward', passed=True))
    print('yolox_s: CUDA forward passed', flush=True)
    del model, result, exp
    gc.collect(); torch.cuda.empty_cache()

    for name, wrapper in [('yolov8s', YOLO), ('rtdetr-l', RTDETR)]:
        container = wrapper(name + '.yaml')
        model = container.model.to(device).eval()
        with torch.inference_mode():
            result = model(torch.zeros(1, 3, 128, 128, device=device))
        assert result is not None
        outcomes.append(dict(method=name, check='random-weight CUDA forward', passed=True))
        print(name + ': CUDA forward passed', flush=True)
        del model, result, container
        gc.collect(); torch.cuda.empty_cache()

    import onnx
    import onnxsim
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    gt.dataset = dict(info={}, images=[dict(id=1, width=128, height=128)],
                      categories=[dict(id=1, name='pig')],
                      annotations=[dict(id=1, image_id=1, category_id=1,
                                        bbox=[10, 10, 20, 20], area=400, iscrowd=0)])
    gt.createIndex()
    pred = gt.loadRes([dict(image_id=1, category_id=1, bbox=[10, 10, 20, 20], score=0.99)])
    evaluator = COCOeval(gt, pred, 'bbox')
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    assert abs(float(evaluator.stats[0]) - 1.) < 1e-8
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    packages = ['torch', 'torchvision', 'numpy', 'opencv-python', 'ultralytics',
                'yolox', 'pycocotools', 'onnx', 'onnxsim']
    report = dict(status='passed', environment='pigdet', python_executable=sys.executable,
                  python_version=platform.python_version(), platform=platform.platform(),
                  gpu=torch.cuda.get_device_name(0), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                  torch_cuda_runtime=torch.version.cuda,
                  versions={name: metadata.version(name) for name in packages},
                  checks=outcomes, coco_oracle_ap=float(evaluator.stats[0]),
                  pip_check=check.stdout.strip(),
                  scope='Environment smoke checks only; random weights, small synthetic input; no dataset training or accuracy evaluation.')
    destination = ROOT / 'experiments/environment/verification.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('All detection environment checks passed.', flush=True)


if __name__ == '__main__':
    main()
