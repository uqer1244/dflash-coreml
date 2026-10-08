"""Minimal standalone runner; heavy runtime imports happen after argument parsing."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description='MLX target + Core ML DFlash2 drafter')
    run = parser.add_subparsers(dest='command', required=True).add_parser('generate')
    run.add_argument('--model', required=True, help='Compatible MLX target directory or HF model ID')
    run.add_argument('--prompt', required=True)
    run.add_argument('--max-tokens', type=int, default=128)
    run.add_argument('--no-thinking', action='store_true')
    run.add_argument('--vanilla', action='store_true', help='Run the target alone')
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error('--max-tokens must be >=1')
    if not args.prompt.strip():
        parser.error('Prompt is empty')
    from .generation import Generator
    from .hub import resolve_model
    print('Loading models...', file=sys.stderr, flush=True)
    try:
        target = resolve_model(args.model)
        bundle = None if args.vanilla else resolve_model(
            'uqer1244/Qwen3.8-27B-DFlash2-CoreML', coreml=True)
        with Generator(target, bundle/'draft' if bundle else None,
                       bundle/'model-manifest.json' if bundle else None,
                       Path('build/ane-bridge'), vanilla=args.vanilla) as generator:
            result = generator.generate(args.prompt, max_tokens=args.max_tokens,
                                        enable_thinking=False if args.no_thinking else None)
        print(result['text'])
    except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
        parser.exit(1, f'Generation failed: {exc}\n')


if __name__ == '__main__':
    main()
