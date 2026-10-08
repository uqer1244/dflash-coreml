"""Standalone CLI. Model/runtime imports are lazy so --help needs only NumPy."""
import argparse
import json
from pathlib import Path
import platform
import sys
from .manifest import load_manifest

def main():
    parser=argparse.ArgumentParser(description='Experimental Core ML DFlash2 standalone generation')
    sub=parser.add_subparsers(dest='command',required=True)
    check=sub.add_parser('check',help='Validate backbone manifest and packages')
    check.add_argument('--manifest',required=True)
    check.add_argument('--verify-hashes',action='store_true')
    run=sub.add_parser('generate',help='Generate text with MLX target + Core ML drafter')
    run.add_argument('--model','--target',dest='target',required=True,help='MLX target: local directory or Hugging Face model ID')
    run.add_argument('--target-revision',help='Pinned target HF revision')
    run.add_argument('--draft',help='Local DFlash2 checkpoint or Hugging Face model ID')
    run.add_argument('--draft-revision',help='Pinned drafter HF revision')
    run.add_argument('--manifest',type=Path)
    run.add_argument('--draft-model','--coreml-model',dest='coreml_model',help='Core ML drafter bundle: local directory or HF ID')
    run.add_argument('--coreml-revision',help='Pinned Core ML bundle HF revision')
    run.add_argument('--bridge',type=Path,default=Path('build/ane-bridge'))
    source=run.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt')
    source.add_argument('--prompt-file',type=Path)
    run.add_argument('--max-tokens',type=int,default=128)
    run.add_argument('--proposal-count',type=int,choices=range(1,6),default=2)
    run.add_argument('--history',choices=['stock','deferred'],default='deferred')
    run.add_argument('--vanilla',action='store_true',help='Target-only control, no Core ML model')
    run.add_argument('--no-thinking',action='store_true')
    run.add_argument('--timeout',type=float,default=120)
    run.add_argument('--json-output',type=Path,help='Save token IDs/timings; existing files are rejected')
    args=parser.parse_args()
    if args.command=='check':
        manifest=load_manifest(args.manifest,args.verify_hashes)
        print(f'Manifest PASS: {len(manifest["packages"])} full + {len(manifest["context_packages"])} context shards')
        print(f'Host: {platform.system()} {platform.machine()}; execution requires Apple Silicon/macOS15+')
        return
    if args.max_tokens<1:parser.error('--max-tokens must be >=1')
    if args.timeout<=0:parser.error('--timeout must be positive')
    if args.manifest and args.coreml_model:parser.error('Choose --manifest or --coreml-model')
    if not args.vanilla and not args.manifest and not args.coreml_model:
        args.coreml_model='uqer1244/Qwen3.8-27B-DFlash2-CoreML'
    if not args.vanilla and not ((args.manifest or args.coreml_model) and args.bridge):
        parser.error('Core ML generation requires --manifest/--coreml-model and --bridge')
    if args.json_output and args.json_output.exists():parser.error('--json-output already exists')
    prompt=args.prompt if args.prompt is not None else args.prompt_file.read_text()
    if not prompt.strip():parser.error('Prompt is empty')
    from .generation import Generator
    print('Loading standalone target and drafter...',file=sys.stderr,flush=True)
    try:
        from .hub import resolve_model
        target=resolve_model(args.target,args.target_revision)
        draft=resolve_model(args.draft,args.draft_revision) if args.draft else None
        manifest=args.manifest
        if args.coreml_model and not args.vanilla:
            manifest=resolve_model(args.coreml_model,args.coreml_revision,coreml=True)/'model-manifest.json'
        if not draft and not args.vanilla:
            data=json.loads(manifest.read_text())
            relative=data.get('selector_directory')
            if not relative:raise ValueError('Manifest lacks a bundled selector; supply --draft')
            draft=(manifest.parent/relative).resolve()
            if not draft.is_relative_to(manifest.parent.resolve()):raise ValueError('Selector path escapes model bundle')
        with Generator(target,draft,manifest,args.bridge,
                       proposal_count=args.proposal_count,timeout=args.timeout,
                       vanilla=args.vanilla,history=args.history) as generator:
            result=generator.generate(prompt,max_tokens=args.max_tokens,
                                      enable_thinking=False if args.no_thinking else None)
        print(result['text'])
        print(f"{result['generated_tokens']} tokens, {result['output_tokens_per_second']:.2f} tok/s, {result['finish_reason']}",file=sys.stderr)
        if args.json_output:
            with args.json_output.open('x') as stream:json.dump(result,stream,indent=2,ensure_ascii=False)
    except (ValueError,RuntimeError,TimeoutError) as exc:
        parser.exit(1,f'Generation failed: {exc}\n')

if __name__=='__main__':main()
