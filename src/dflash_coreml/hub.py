"""Resolve explicitly selected local files or Hugging Face snapshots."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

CHECKPOINT_FILES=['*.json','*.safetensors','tokenizer.model','*.tiktoken','README.md','LICENSE*']
COREML_FILES=['model-manifest.json','models/**','draft/config.json','draft/model.safetensors','README.md','LICENSE*','NOTICE']

def resolve_model(value,revision=None,*,coreml=False):
    value=str(value)
    path=Path(value).expanduser()
    if path.exists():
        return path.resolve()
    if path.is_absolute() or value.startswith(('.', '~')):
        raise FileNotFoundError(f'Local model path does not exist: {path}')
    if value.count('/')!=1 or any(part in ('','..') for part in value.split('/')):
        raise ValueError('Supply a local directory or a Hugging Face owner/model ID')
    from huggingface_hub import snapshot_download
    folder=snapshot_download(repo_id=value,revision=revision,
                             allow_patterns=COREML_FILES if coreml else CHECKPOINT_FILES)
    return _materialize_coreml_snapshot(folder) if coreml else Path(folder)


def _materialize_coreml_snapshot(folder):
    """Core ML compilation needs ordinary package files, not HF blob symlinks.

    APFS clones isolate writes without duplicating weight storage. Publish the
    prepared directory atomically so concurrent callers never see a partial copy.
    """
    folder = Path(folder)
    parent = folder.parent.parent / 'coreml-packages'
    destination = parent / folder.name
    if destination.is_dir():
        return destination
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.preparing-', dir=parent) as temporary:
        staged = Path(temporary)
        for source in sorted(folder.rglob('*')):
            if not source.is_file():
                continue
            target = staged / source.relative_to(folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            cloned = False
            if sys.platform == 'darwin':
                cloned = subprocess.run(['cp', '-c', str(source.resolve()), str(target)],
                                        capture_output=True).returncode == 0
            if not cloned:
                if target.exists():
                    target.unlink()
                shutil.copy2(source, target)
        try:
            os.rename(staged, destination)
        except OSError:
            if not destination.is_dir():
                raise
    return destination
