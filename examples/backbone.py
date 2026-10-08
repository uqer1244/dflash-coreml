"""Pass caller-provided embeddings and committed target features to Core ML."""
import argparse
import numpy as np
from dflash_coreml import CoreMLBackbone

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', required=True)
parser.add_argument('--bridge', required=True)
parser.add_argument('--inputs', required=True, help='NPZ: embedding [1,6,5120], features [1,N,25600]')
parser.add_argument('--position', type=int, default=0)
parser.add_argument('--output', required=True)
args = parser.parse_args()
with np.load(args.inputs, allow_pickle=False) as data:
    embedding, features = data['embedding'], data['features']
with CoreMLBackbone(args.manifest, args.bridge) as backend:
    backend.reset(args.position)
    hidden, calls = backend.forward(embedding, features)
    np.savez(args.output, hidden=hidden)
    print(f'{len(calls)} committed features; next position {backend.offset}')
