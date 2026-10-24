"""Activate pigdet, then run: python train.py. Six detectors run sequentially."""
from pathlib import Path
import argparse
import sys
from tools.runtime import ROOT, require_env, sha256
from tools.experiment_flow import ExperimentFlow, read_json

METHODS = ['yolov8s', 'yolox_s', 'fasterrcnn_r50_fpn', 'retinanet_r50_fpn', 'fcos_r50_fpn', 'rtdetr_l']
DATASET = ROOT / 'data/processed/exploratory'

def preflight(profile='local_40'):
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
        config_dir=ROOT/'configs' if profile=='local_40' else ROOT/'configs/server_100'
        config = yaml.safe_load((config_dir/f'{method}.yaml').read_text(encoding='utf-8'))
        if config['method'] != method or min(config['epochs'], config['batch'], config['effective_batch']) < 1:
            raise ValueError(f'配置有误：{method}')
        if config['effective_batch'] % config['batch']:
            raise ValueError(f'effective_batch 必须是 batch 的整数倍：{method}')
        expected=dict(epochs=40,batch=4,effective_batch=4,validation_batch=1,seed=42) if profile=='local_40' else dict(epochs=100,effective_batch=32,seed=42)
        if any(config.get(k)!=v for k,v in expected.items()):
            raise ValueError(f'{method}: 配置与 {profile} 协议不符：{expected}')
        configs[method] = config
    print('检测项目：', ROOT, flush=True)
    print('顺序运行六种方法；每种参数读取对应 configs YAML。', flush=True)
    for method, c in configs.items():
        print(f'  {method}: epochs={c["epochs"]}, batch={c["batch"]}, seed={c["seed"]}, imgsz={c["imgsz"]}', flush=True)
    return configs

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Only check inputs/configuration; do not train')
    parser.add_argument('--profile',choices=['local_40','server_100'],default='local_40',help='Default local 40-epoch recipe; server_100 uses the recorded 3090 batch budget')
    args = parser.parse_args()
    configs = preflight(args.profile)
    if args.check:
        print(f'入口检查通过；没有启动训练。当前方案：{args.profile}，实际显存需求未在此检查。')
        return
    with ExperimentFlow(ROOT, 'six-detector training and evaluation; tracking uses its own independent detector') as flow:
        flow.record['frozen_configs']=configs
        flow.record['profile']=args.profile
        flow.stage('校验数据', 'tools/validate_export.py', '--dataset', DATASET)
        flow.stage('校验/补下载初始化权重', 'tools/download_weights.py')
        runs = {}; metrics = []
        # Finish validation-based training/selection before evaluating any test split.
        for method in METHODS:
            config = configs[method]
            name = f'{method}_seed{config["seed"]}_{flow.token}'
            config_dir=ROOT/'configs' if args.profile=='local_40' else ROOT/'configs/server_100'
            training_dir=flow.directory/'training'/method
            flow.stage(f'训练 {method}', 'scripts/train.py', '--config', config_dir/f'{method}.yaml',
                       '--dataset', DATASET, '--name', name,'--output',training_dir)
            manifest_path = training_dir/'run_manifest.json'
            run = read_json(manifest_path)
            if run['config']!=config:raise RuntimeError(f'训练期间配置发生改变：{method}')
            if run['status'] != 'completed' or run['smoke_only']:
                raise RuntimeError(f'不是已完成的正式训练：{manifest_path}')
            checkpoint = Path(run['checkpoint'])
            if sha256(checkpoint) != run['checkpoint_sha256']:
                raise RuntimeError(f'训练权重校验失败：{checkpoint}')
            runs[method] = dict(name=name,training_dir=str(training_dir),checkpoint=str(checkpoint),manifest=str(manifest_path),config=config)
        for method in METHODS:
            run = runs[method]
            result = flow.directory/'test'/method
            flow.stage(f'测试 {method}', 'scripts/export_detection.py', '--method', method,
                       '--weights', run['checkpoint'], '--dataset', DATASET,
                       '--imgsz', configs[method]['imgsz'], '--output', result)
            metrics.append(result/'metrics.json')
            run['metrics'] = str(result/'metrics.json')
            run['test_dir']=str(result)
        table = flow.directory/'detection.csv'
        flow.stage('生成检测对比表', 'tools/summarize_results.py', '--results', *metrics, '--output', table)
        from tools.experiment_flow import save_json
        index=flow.directory/'result_index.json'
        save_json(index,dict(schema_version=1,batch_id=flow.token,profile=args.profile,dataset=str(DATASET),
            hardware=read_json(next(iter(runs.values()))['manifest'])['environment']['gpu'],models=runs))
        report=flow.directory/'report'
        flow.stage('自动生成训练曲线与测试可视化','tools/build_experiment_report.py','--index',index,'--output',report)
        if read_json(report/'report_manifest.json')['status']!='completed':raise RuntimeError('图表报告未完整生成')
        flow.complete(dict(runs=runs,detection_table=str(table),report=str(report/'index.html'),
                           tracking_detector_route='standalone tracking project; no caches generated by detection project'))
        print(f'\n检测实验完成。对比表：{table}', flush=True)
        print(f'可视化总报告：{report/"index.html"}',flush=True)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f'运行停止：{error}', file=sys.stderr)
        raise SystemExit(1)
