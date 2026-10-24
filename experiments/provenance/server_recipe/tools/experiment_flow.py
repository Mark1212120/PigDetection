"""Run existing experiment stages with live output and a durable progress record."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import subprocess
import sys

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

class ExperimentFlow:
    def __init__(self, root, task):
        self.root = Path(root)
        self.token = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        self.directory = self.root / 'experiments/pipelines' / self.token
        self.manifest = self.directory / 'pipeline_manifest.json'
        self.record = dict(task=task, status='started', run_id=self.token, python=sys.executable,
                           root=str(self.root), stages=[], restart_policy='new run; no automatic checkpoint resume')

    def __enter__(self):
        save_json(self.manifest, self.record)
        return self

    def stage(self, label, script, *arguments):
        command = [sys.executable, '-u', str(self.root / script), *map(str, arguments)]
        stage = dict(label=label, command=command, status='running')
        self.record['stages'].append(stage)
        save_json(self.manifest, self.record)
        print(f'\n[{len(self.record["stages"])}] {label}', flush=True)
        try:
            result = subprocess.run(command, cwd=self.root, check=True)
        except BaseException as error:
            stage.update(status='failed', error=repr(error))
            raise
        stage.update(status='completed', returncode=result.returncode)
        save_json(self.manifest, self.record)

    def complete(self, outputs):
        self.record.update(status='completed', outputs=outputs)
        save_json(self.manifest, self.record)
        save_json(self.root / 'experiments/latest_pipeline.json', self.record)

    def __exit__(self, exc_type, error, traceback):
        if error is not None:
            self.record.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
            save_json(self.manifest, self.record)
            print(f'\n流程停止，记录位置：{self.manifest}', flush=True)
            print('未修改 batch、分辨率或旧结果；再次启动会新建一轮实验，重新执行流程，不自动续跑。', flush=True)
        return False
