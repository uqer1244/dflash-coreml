"""Prepare a local Hugging Face Core ML bundle. Never upload or change sources."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--draft',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--compact-context',action='store_true',help='Repack unused context constants (requires coremltools 9)')
args=parser.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dflash_coreml.manifest import load_manifest
from dflash_coreml.checkpoint import selector_hashes
raw=json.loads(args.manifest.read_text())
validated=load_manifest(args.manifest,True)
if not raw.get('generation_binding'):raise ValueError('Manifest requires generation binding')
if args.output.exists():raise FileExistsError('Choose a new export directory')
args.output.mkdir(parents=True)
try:
 for key in ['packages','context_packages']:
  for i,path in enumerate(validated[key]):
   relative=f'models/{"full" if key=="packages" else "context"}-{i}.mlpackage'
   dest=args.output/relative
   dest.parent.mkdir(exist_ok=True)
   if args.compact_context and key=='context_packages':
    from compact_context import compact
    compact(Path(path),dest)
    raw[key][i]['files']={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob('*')) if p.is_file()}
   elif sys.platform=='darwin':
    # Copy-on-write clone avoids duplicating multi-GB local weight blobs on APFS.
    result=subprocess.run(['cp','-cR',path,str(dest)],capture_output=True)
    if result.returncode:
     if dest.exists():shutil.rmtree(dest)
     shutil.copytree(path,dest)
   else:shutil.copytree(path,dest)
   raw[key][i]['path']=relative
 draft=args.output/'draft';draft.mkdir()
 shutil.copy2(args.draft/'config.json',draft/'config.json')
 import mlx.core as mx
 weights=mx.load(str(args.draft/'model.safetensors'))
 selected={k:v for k,v in weights.items() if k.startswith('candidate_selector.')}
 mx.eval(selected)
 mx.save_safetensors(str(draft/'model.safetensors'),selected)
 if selector_hashes(draft/'model.safetensors')!=raw['generation_binding']['selector_tensors']:
  raise ValueError('Exported selector is not byte-exact')
 raw['selector_directory']='draft'
 raw['source']={'repo_id':'z-lab/Qwen3.8-27B-DFlash2',
                'revision':'50307d4c4cde6860d4eee73e2547cd786fe8e8a4',
                'checkpoint_sha256':'67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c'}
 manifest=args.output/'model-manifest.json'
 manifest.write_text(json.dumps(raw,indent=2)+'\n')
 load_manifest(manifest,True)
 card='''---
license: apache-2.0
base_model:
  - z-lab/Qwen3.8-27B-DFlash2
tags:
  - coreml
  - dflash2
  - speculative-decoding
  - apple-neural-engine
  - experimental
---

# Qwen3.8-27B-DFlash2 Core ML — experimental

Stateful Core ML conversion of the DFlash2 drafter backbone, using three2+2+1 layer shards with mixed INT8/FP16 precision, block6 and capacity31. Includes matching context-only graphs and original BF16 selector weights. The original checkpoint uses block8; this conversion's fixed block6 is explicit.

This is a drafter, not a standalone language model. It requires the compatible MLX target and the `dflash-coreml` Python/Swift runner. The target shared LM head, Top16, selector and verification remain outside Core ML. No Python remote code is bundled or executed by this model snapshot.

## Contents

- `models/full-0..2.mlpackage`: fusion, five drafter layers, final norm and state updates.
- `models/context-0..2.mlpackage`: matching sequential context-only commit graphs.
- `draft/config.json` and `draft/model.safetensors`: selector-only weights; the entire original BF16 drafter download is unnecessary.
- `model-manifest.json`: per-file SHA256 and exact target/selector binding.

## Run

Install the standalone runner source repository, its `[generate]` dependencies, and build `native/build.sh`. Then:

```sh
dflash-coreml generate \\
  --target /path/to/compatible-Bonsai2-target \\
  --coreml-model YOUR_HF_ACCOUNT/Qwen3.8-27B-DFlash2-CoreML \\
  --coreml-revision YOUR_PUBLISHED_COMMIT \\
  --bridge build/ane-bridge \\
  --prompt "What is the capital of South Korea?" --max-tokens 128
```

Replace repository/commit placeholders after publication. The compatible target is the verified `nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX` artifact; this bundle binds its exact config/checkpoint hashes. Other target weights fail validation. A local bundle directory also works with `--coreml-model`.

## Status and limitations

Experimental, M3 Pro18GB validated. Normal selector replay91.07% is below the95% fidelity gate. Sequential context optimization preserves the previous Core ML state/output; S=2 batching was rejected. General lossless output and GPU superiority are not established. The runner is greedy/text-only; no sampling or server batching claim.

Historical ANE-path budget5→2 improved whole-output median7.342→7.924tok/s on four prompts/five repeats. This is not a GPU comparison. Drafter-only combined power measurements are not total generation energy. Detailed results accompany the runner source release.

Derived from [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), a mirror of [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2). Apache-2.0. Original revision/hash are recorded in the manifest. Conversion and package assembly do not alter the original source checkpoint.
'''
 (args.output/'README.md').write_text(card)
 license_source=Path(__file__).resolve().parents[1]/'LICENSES/Apache-2.0.txt'
 shutil.copy2(license_source,args.output/'LICENSE')
 (args.output/'NOTICE').write_text('Derived from Qwen3.8-27B-DFlash2 by Z Lab / Inco. Apache-2.0.\nSee model-manifest.json for exact original revision/checkpoint hash.\nConversion: stateful Core ML 2+2+1 mixed INT8/FP16, block6/capacity31.\n')
 total=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
 print(f'Prepared {args.output}; {total/2**30:.3f} logical GiB; package and selector hashes PASS')
except BaseException:
 # An incomplete export must not be mistaken for a publishable model bundle.
 (args.output/'EXPORT_INCOMPLETE').write_text('Export failed; remove this new directory and retry.\n')
 raise
