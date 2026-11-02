import hashlib
import importlib.metadata
import json
from pathlib import Path


def read_config(path='config.json'):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    for key in ['data_dir', 'local_dir']:
        cfg[key] = str((path.parent / cfg[key]).resolve())
    cfg['config_path'] = str(path)
    return cfg


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def versions():
    return {name: importlib.metadata.version(name) for name in
        ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'catboost', 'scipy', 'joblib']}
