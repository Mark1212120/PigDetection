"""Six adapters. This module never downloads missing weights implicitly."""
from pathlib import Path
import json
import cv2
import numpy as np
import torch
import torchvision

METHODS = ['fasterrcnn_r50_fpn', 'retinanet_r50_fpn', 'fcos_r50_fpn', 'yolox_s', 'yolov8s', 'rtdetr_l']
TV = dict(fasterrcnn_r50_fpn=torchvision.models.detection.fasterrcnn_resnet50_fpn,
          retinanet_r50_fpn=torchvision.models.detection.retinanet_resnet50_fpn,
          fcos_r50_fpn=torchvision.models.detection.fcos_resnet50_fpn)

def build_native(method, size, device, weight=None, initialize=False):
    if method in TV:
        model = TV[method](weights=None, weights_backbone=None, num_classes=2, min_size=size,
                           max_size=size, box_score_thresh=.001, box_detections_per_img=300) if method == 'fasterrcnn_r50_fpn' else TV[method](weights=None, weights_backbone=None, num_classes=2, min_size=size, max_size=size, score_thresh=.001, detections_per_img=300)
    elif method == 'yolox_s':
        from yolox.exp import get_exp
        exp = get_exp(exp_name='yolox-s'); exp.num_classes = 1
        model = exp.get_model()
    else:
        raise ValueError(method)
    load_record = {}
    if weight:
        weight = Path(weight)
        if not weight.is_file():
            raise FileNotFoundError(f'Download or supply weights first: {weight}')
        checkpoint = torch.load(weight, map_location='cpu', weights_only=False)
        state = checkpoint.get('model', checkpoint)
        if initialize:
            own = model.state_dict()
            matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
            missing, unexpected = model.load_state_dict(matched, strict=False)
            load_record = dict(loaded_tensors=len(matched), skipped_keys=[k for k in state if k not in matched],
                               newly_initialized_keys=list(missing), unexpected=list(unexpected))
        else:
            assert checkpoint.get('method') == method and checkpoint.get('class_names') == ['pig'], 'Not a pig project checkpoint'
            model.load_state_dict(state, strict=True)
            load_record = dict(smoke_only=bool(checkpoint.get('smoke_only',False)),
                               split_sha256=checkpoint.get('split_sha256'))
    return model.to(device), load_record

class Predictor:
    def __init__(self, method, weight, device='cuda:0', size=640, conf=.001, iou=.7):
        self.method = method; self.device = torch.device(device); self.size = size; self.conf = conf; self.iou = iou
        self.smoke_only=False;self.training_split_sha256=None;self.training_manifest=None
        if not Path(weight).is_file():
            raise FileNotFoundError(weight)
        if method in ('yolov8s', 'rtdetr_l'):
            from ultralytics import YOLO, RTDETR
            self.model = (YOLO if method == 'yolov8s' else RTDETR)(str(weight))
            assert list(self.model.names.values()) == ['pig'], 'Train a single-class pig checkpoint first; COCO has no pig class.'
        else:
            self.model, info = build_native(method, size, device, weight)
            self.smoke_only=info.get('smoke_only',False);self.training_split_sha256=info.get('split_sha256')
            self.model.eval()
        for parent in list(Path(weight).resolve().parents)[:4]:
            path=parent/'run_manifest.json'
            if path.is_file():
                provenance=json.loads(path.read_text(encoding='utf-8'))
                self.smoke_only=bool(provenance.get('smoke_only',False))
                self.training_split_sha256=provenance.get('split_sha256');self.training_manifest=str(path)
                break

    @torch.inference_mode()
    def __call__(self, image):
        h, w = image.shape[:2]
        if self.method in ('yolov8s', 'rtdetr_l'):
            result = self.model.predict(image, imgsz=self.size, conf=self.conf, iou=self.iou,
                                        max_det=300, device=str(self.device), verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy(); scores = result.boxes.conf.cpu().numpy()
        elif self.method == 'yolox_s':
            from yolox.data.data_augment import preproc
            from yolox.utils import postprocess
            array, ratio = preproc(image, (self.size, self.size))
            raw = self.model(torch.from_numpy(array).unsqueeze(0).to(self.device))
            result = postprocess(raw, 1, self.conf, self.iou, class_agnostic=True)[0]
            if result is None:
                return np.empty((0, 5), dtype=np.float32)
            result = result.cpu().numpy(); boxes = result[:, :4] / ratio; scores = result[:, 4] * result[:, 5]
        else:
            array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
            result = self.model([torch.from_numpy(array).float().to(self.device) / 255])[0]
            keep = (result['labels'] == 1) & (result['scores'] >= self.conf)
            boxes = result['boxes'][keep].cpu().numpy(); scores = result['scores'][keep].cpu().numpy()
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        rows = np.column_stack((boxes[keep], scores[keep])).astype(np.float32)
        return rows[np.argsort(-rows[:, 4])[:300]]
