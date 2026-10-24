"""Activate pigdet, then run: python train.py. Six detectors run sequentially."""
from pathlib import Path
import argparse
import sys
from tools.runtime import ROOT, require_env, sha256
from tools.experiment_flow import ExperimentFlow, read_json

METHODS = ['yolov8s', 'yolox_s', 'fasterrcnn_r50_fpn', 'retinanet_r50_fpn', 'fcos_r50_fpn', 'rtdetr_l']
TRACKING_ROOT = Path('D:/Projects/Pig_Tracking_test')
DATASET = ROOT / 'data/processed/exploratory'

def preflight():
    require_env()
    import yaml
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，请先检查 pigdet 环境。')
    for path in [DATASET/'split_manifest.json', DATASET/'dataset.yaml']:
        if not path.is_file():
            raise FileNotFoundError(f'缺少已准备的数据清单：{path}')
    split = read_json(DATASET/'split_manifest.json')
    if split['protocol'] == 'smoke':
        raise RuntimeError('默认入口不能使用 smoke 数据作为正式实验。')
    configs = {}
    for method in METHODS:
        config = yaml.safe_load((ROOT/'configs'/f'{method}.yaml').read_text(encoding='utf-8'))
        if config['method'] != method or min(config['epochs'], config['batch'], config['effective_batch']) < 1:
            raise ValueError(f'配置有误：{method}')
        if config['effective_batch'] % config['batch']:
            raise ValueError(f'effective_batch 必须是 batch 的整数倍：{method}')
        configs[method] = config
    print('检测项目：', ROOT, flush=True)
    print('顺序运行六种方法；每种参数读取对应 configs YAML。', flush=True)
    for method, c in configs.items():
        print(f'  {method}: epochs={c["epochs"]}, batch={c["batch"]}, seed={c["seed"]}, imgsz={c["imgsz"]}', flush=True)
    return configs

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Only check inputs/configuration; do not train')
    args = parser.parse_args()
    configs = preflight()
    if args.check:
        print('Server preflight passed; configurations recorded in configs/*.yaml.')
        return
    with ExperimentFlow(ROOT, 'six-detector server training and evaluation') as flow:
        flow.stage('校验数据', 'tools/validate_export.py', '--dataset', DATASET)
        flow.stage('校验/补下载初始化权重', 'tools/download_weights.py')
        runs = {}; metrics = []
        # Finish validation-based training/selection before evaluating any test split.
        for method in METHODS:
            config = configs[method]
            name = f'{method}_seed{config["seed"]}_{flow.token}'
            flow.stage(f'训练 {method}', 'scripts/train.py', '--config', ROOT/'configs'/f'{method}.yaml',
                       '--dataset', DATASET, '--name', name)
            manifest_path = ROOT/'experiments/runs'/name/'run_manifest.json'
            run = read_json(manifest_path)
            if run['status'] != 'completed' or run['smoke_only']:
                raise RuntimeError(f'不是已完成的正式训练：{manifest_path}')
            checkpoint = Path(run['checkpoint'])
            if sha256(checkpoint) != run['checkpoint_sha256']:
                raise RuntimeError(f'训练权重校验失败：{checkpoint}')
            runs[method] = dict(name=name, checkpoint=str(checkpoint), manifest=str(manifest_path), config=config)
        for method in METHODS:
            run = runs[method]
            result = ROOT/'experiments/results'/f'{run["name"]}_test'
            flow.stage(f'测试 {method}', 'scripts/export_detection.py', '--method', method,
                       '--weights', run['checkpoint'], '--dataset', DATASET,
                       '--imgsz', configs[method]['imgsz'], '--output', result)
            metrics.append(result/'metrics.json')
            run['metrics'] = str(result/'metrics.json')
        table = flow.directory/'detection.csv'
        flow.stage('生成检测对比表', 'tools/summarize_results.py', '--results', *metrics, '--output', table)
        flow.complete(dict(runs=runs, detection_table=str(table)))
        print(f'\n检测实验完成。对比表：{table}', flush=True)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f'运行停止：{error}', file=sys.stderr)
        raise SystemExit(1)
