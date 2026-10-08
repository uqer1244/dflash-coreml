"""Resolve explicitly selected local files or Hugging Face snapshots."""
from pathlib import Path

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
    return Path(folder)
