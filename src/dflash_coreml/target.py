# SPDX-License-Identifier: Apache-2.0
# Adapted from oMLX's prism_hadamard_qwen35 loader; text-only local loading added.
"""Schema2 Bonsai2 text loader without an oMLX app or vision allocations."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import mlx.core as mx
from mlx import nn
from mlx.utils import tree_unflatten
from mlx_vlm.models.qwen3_5.config import ModelConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel

class PackedLinear(nn.Module):
    def __init__(self, width, rows, block):
        super().__init__()
        if width%128 or block not in (0,512,1024,2048,4096) or (block and width%block):
            raise ValueError('Unsupported packed width/Hadamard block')
        self.block=block
        self.bits=2
        self.group_size=128
        self.mode='affine'
        self.weight=mx.zeros((rows,width//16),dtype=mx.uint32)
        self.scales=mx.zeros((rows,width//128),dtype=mx.float16)
        self.biases=mx.zeros_like(self.scales)
        if block:self.signs=mx.ones((width,),dtype=mx.float32)
        self.freeze()
    def _rotate(self,x,inverse=False):
        shape,dtype=x.shape,x.dtype
        x=x.astype(mx.float32)
        if not inverse:x=x*self.signs
        x=mx.hadamard_transform(x.reshape(-1,self.block),scale=1/math.sqrt(self.block)).reshape(shape)
        if inverse:x=x*self.signs
        return x.astype(dtype)
    def __call__(self,x):
        if self.block:x=self._rotate(x)
        return mx.quantized_matmul(x,self.weight,self.scales,self.biases,transpose=True,group_size=128,bits=2)

class PackedEmbedding(PackedLinear):
    def __call__(self,indices):
        shape=indices.shape
        rows=indices.reshape(-1)
        x=mx.dequantize(self.weight[rows],self.scales[rows],self.biases[rows],group_size=128,bits=2).reshape(*shape,-1).astype(mx.float16)
        return self._rotate(x,True) if self.block else x
    def as_linear(self,x):return super().__call__(x)

def load_model(path,lazy=True):
    path=Path(path)
    data=json.loads((path/'config.json').read_text())
    required={'model_type':'prism_hadamard_qwen35','schema_version':2,'base_model_type':'qwen3_5',
              'tensor_namespace':'mlx-vlm-qwen3_5','gdn_activation_layout':'grouped'}
    if any(data.get(k)!=v for k,v in required.items()) or data.get('quantization')!={'bits':2,'group_size':128,'mode':'affine'}:
        raise ValueError('Unsupported Bonsai2 schema/layout/quantization')
    cfg=ModelConfig.from_dict({**data,'model_type':'qwen3_5'})
    lm=LanguageModel(cfg.text_config,cfg)
    modules=dict(lm.named_modules())
    replacements=[]
    seen=set()
    for record in data['modules']:
        name=record['path']
        original=modules.get(name)
        if name in seen or not isinstance(original,(nn.Linear,nn.Embedding)):
            raise ValueError(f'Invalid packed module: {name}')
        seen.add(name)
        is_embedding=isinstance(original,nn.Embedding)
        if record['embedding']!=is_embedding or record['dtype']!='float16' or 'bias' in original:
            raise ValueError(f'Unsupported packed module kind/dtype/bias: {name}')
        rows,width=original.weight.shape
        cls=PackedEmbedding if is_embedding else PackedLinear
        replacements.append((name,cls(width,rows,record['block'])))
    lm.update_modules(tree_unflatten(replacements))
    weights={}
    for shard in sorted(path.glob('*.safetensors')):
        for name,value in mx.load(str(shard)).items():
            if name.startswith('language_model.'):
                key=name.removeprefix('language_model.')
                if key in weights:raise ValueError(f'Duplicate text tensor: {key}')
                weights[key]=value
    checks=[]
    for record in data['modules']:
        if record['block']:
            signs=weights[record['path']+'.signs']
            checks.append(mx.all((signs==1)|(signs==-1)))
    if checks and not bool(mx.all(mx.stack(checks)).item()):
        raise ValueError('Invalid Hadamard sign values')
    lm.load_weights(list(weights.items()),strict=True)
    lm.eval()
    if not lazy:mx.eval(lm.parameters())
    return SimpleNamespace(language_model=lm)
