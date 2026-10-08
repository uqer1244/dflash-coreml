"""Download your HF Core ML drafter bundle and generate with a separate MLX target.

After publishing weights:
python examples/mlx_generate.py --hf-owner YOUR_HF_ACCOUNT --prompt 'Hello'
"""
import argparse
from pathlib import Path
from dflash_coreml.generation import Generator
from dflash_coreml.hub import resolve_model

MODEL_NAME='Qwen3.8-27B-DFlash2-CoreML'
parser=argparse.ArgumentParser(description='MLX target + Hugging Face Core ML DFlash2 example')
parser.add_argument('--hf-owner',required=True,help='Account that published the Core ML model')
parser.add_argument('--revision',help='Published Core ML commit hash')
parser.add_argument('--model',default='nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX')
parser.add_argument('--target-revision')
parser.add_argument('--bridge',type=Path,default=Path('build/ane-bridge'))
parser.add_argument('--prompt',required=True)
parser.add_argument('--max-tokens',type=int,default=128)
parser.add_argument('--no-thinking',action='store_true')
args=parser.parse_args()
if not args.hf_owner or '/' in args.hf_owner or args.hf_owner in ('.','..'):
    parser.error('--hf-owner must be one Hugging Face account name')
if args.max_tokens<1:parser.error('--max-tokens must be >=1')
bundle=resolve_model(f'{args.hf_owner}/{MODEL_NAME}',args.revision,coreml=True)
target=resolve_model(args.model,args.target_revision)
with Generator(target,bundle/'draft',bundle/'model-manifest.json',args.bridge) as runner:
    result=runner.generate(args.prompt,max_tokens=args.max_tokens,
                          enable_thinking=False if args.no_thinking else None)
print(result['text'])
