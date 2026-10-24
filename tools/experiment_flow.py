"""Run existing experiment stages with live output and a durable progress record."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import subprocess
import sys
import os

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
        self.directory = self.root / 'experiments/records' / self.token
        self.manifest = self.directory / 'pipeline_manifest.json'
        self.record = dict(task=task, status='started', run_id=self.token, python=sys.executable,
                           root=str(self.root), stages=[], restart_policy='new run; no automatic checkpoint resume')

    def __enter__(self):
        save_json(self.manifest, self.record)
        return self

    def stage(self, label, script, *arguments):
        command = [sys.executable, '-u', str(self.root / script), *map(str, arguments)]
        log_path = self.directory / f'stage_{len(self.record["stages"])+1:02d}.log'
        stage = dict(label=label, command=command, status='running', log=str(log_path))
        self.record['stages'].append(stage)
        save_json(self.manifest, self.record)
        print(f'\n[{len(self.record["stages"])}] {label}', flush=True)
        try:
            env = dict(os.environ, PYTHONIOENCODING='utf-8')
            with log_path.open('w', encoding='utf-8') as log:
                with subprocess.Popen(command, cwd=self.root, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
                    try:
                        for line in process.stdout:
                            sys.stdout.write(line);sys.stdout.flush()
                            log.write(line);log.flush()
                        returncode = process.wait()
                        if returncode:raise subprocess.CalledProcessError(returncode,command)
                    except BaseException:
                        if process.poll() is None:
                            process.terminate()
                            try:process.wait(timeout=10)
                            except subprocess.TimeoutExpired:process.kill();process.wait()
                        raise
        except BaseException as error:
            stage.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed', error=repr(error))
            if isinstance(error,subprocess.CalledProcessError):stage['returncode']=error.returncode
            save_json(self.manifest,self.record)
            raise
        stage.update(status='completed', returncode=returncode)
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
            print('流程已停止。无参 python train.py 会新建批次；本检测项目没有自动断点续训入口。',flush=True)
        return False
