"""Local preparation utility for the existing research checkout, no downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dflash_coreml.checkpoint import selector_hashes

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--output', type=Path, default=Path('local-manifest.json'))
parser.add_argument('--target', type=Path, help='Local generation target; add checkpoint binding')
args = parser.parse_args()
root = args.workspace.resolve()/'ane-dflash'
full = json.loads((root/'phase3/neo-compact-20261007/all_ad_L4-2+2+1-packages.json').read_text())
context = json.loads((root/'phase4_1/v1/context-packages.json').read_text())
output = args.output.resolve()
data = {'schema_version':1, 'status':'experimental-selector-gate-failed',
        'contract':{'hidden_size':5120,'feature_size':25600,'block_size':6,'capacity':31,'rope_theta':10000000,'shards':3}}
for key, rows in [('packages',full),('context_packages',context)]:
    entries = []
    for row in rows:
        path = Path(row['package'])
        # Recorded manifests contain old absolute paths; resolve from workspace marker.
        relative = str(path).split('/ane-dflash/',1)[1]
        path = root/relative
        files = {}
        for f in sorted(path.rglob('*')):
            if f.is_file():
                with f.open('rb') as stream:files[str(f.relative_to(path))] = hashlib.file_digest(stream,'sha256').hexdigest()
        entries.append({'path':os.path.relpath(path,output.parent),'files':files})
    data[key] = entries
if args.target:
    draft=root.parent/'models/Qwen3.8-27B-DFlash2-BF16'
    target=args.target.resolve()
    weights={}
    for f in sorted(target.glob('*.safetensors')):
        with f.open('rb') as stream:weights[f.name]=hashlib.file_digest(stream,'sha256').hexdigest()
    if not weights:raise ValueError('Target has no safetensors weights')
    data['generation_binding']={
        'draft_config':hashlib.sha256((draft/'config.json').read_bytes()).hexdigest(),
        'target_config':hashlib.sha256((target/'config.json').read_bytes()).hexdigest(),
        'target_weights':weights,
        'selector_tensors':selector_hashes(draft/'model.safetensors'),
    }
output.write_text(json.dumps(data,indent=2)+'\n')
print(output)
