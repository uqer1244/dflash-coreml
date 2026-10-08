"""Run an MLX target with the default Hugging Face Core ML drafter.

python examples/mlx_generate.py --model /path/to/target --prompt 'Hello'
"""
import sys
from dflash_coreml.cli import main

if __name__ == '__main__':
    sys.argv.insert(1, 'generate')
    main()
