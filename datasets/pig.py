"""Real-data COCO reader. TorchVision/YOLOX keep their native preprocessing."""
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class PigDataset(Dataset):
    def __init__(self, directory, split, method, size=640, limit=0):
        self.directory = Path(directory); self.split = split; self.method = method; self.size = size
        self.coco = json.loads((self.directory / 'annotations' / f'{split}.json').read_text())
        self.images = self.coco['images'][:limit or None]
        self.by_image = {r['id']: [] for r in self.images}
        for row in self.coco['annotations']:
            if row['image_id'] in self.by_image:
                self.by_image[row['image_id']].append(row)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        record = self.images[index]
        image = cv2.imread(str(self.directory / 'images' / self.split / record['file_name']))
        if image is None:
            raise RuntimeError(record)
        boxes = np.asarray([r['bbox'] for r in self.by_image[record['id']]], dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        if self.method == 'yolox_s':
            from yolox.data.data_augment import preproc
            image, ratio = preproc(image, (self.size, self.size))
            boxes *= ratio
            target = np.zeros((100, 5), dtype=np.float32)
            assert len(boxes) <= len(target)
            target[:len(boxes), 1:3] = (boxes[:, :2] + boxes[:, 2:]) / 2
            target[:len(boxes), 3:5] = boxes[:, 2:] - boxes[:, :2]
            return torch.from_numpy(image), torch.from_numpy(target)
        image = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()).float() / 255
        return image, dict(boxes=torch.from_numpy(boxes), labels=torch.ones(len(boxes), dtype=torch.int64),
                           image_id=torch.tensor(record['id']))

def collate(batch):
    return tuple(zip(*batch))
