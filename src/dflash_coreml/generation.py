"""Standalone greedy text generation with an MLX target and Core ML draft."""
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

LAYER_IDS = [5,19,33,47,61]

def resolve_prefix(proposals, truth, remaining, eos):
    """Resolve verified proposals and one correction/bonus before cache commit."""
    if remaining < 1 or len(truth) != len(proposals)+1:
        raise ValueError('Invalid verification result or remaining token limit')
    accepted = next((i for i,(a,b) in enumerate(zip(proposals,truth)) if a != b),len(proposals))
    emitted = (proposals[:accepted]+[truth[accepted]])[:remaining]
    stop = next((i for i,token in enumerate(emitted) if token in eos),None)
    if stop is not None:
        emitted = emitted[:stop+1]
    return emitted, min(accepted,len(emitted)), accepted

class Generator:
    """Local, text-only, single-request greedy runtime. No oMLX app import."""
    def __init__(self, target, draft, manifest=None, bridge=None, *, proposal_count=2,
                 timeout=120, vanilla=False, history='deferred', verify_hashes=True):
        if proposal_count not in range(1,6):
            raise ValueError('proposal_count must be 1..5')
        self.target = Path(target).resolve()
        self.draft = Path(draft).resolve() if draft else None
        self.vanilla = vanilla
        self.closed = False
        self.backbone = None
        self.proposal_count = proposal_count
        self.history = history
        from .runtime import imports
        self.mx, load_model, AutoTokenizer = imports()
        mx = self.mx
        config = json.loads((self.target/'config.json').read_text())
        if config.get('model_type') != 'prism_hadamard_qwen35':
            raise ValueError('This preview supports only the verified prism_hadamard_qwen35 target layout')
        self.config_sha256 = hashlib.sha256((self.target/'config.json').read_bytes()).hexdigest()
        try:
            if not vanilla:
                if not self.draft or not manifest or not bridge:
                    raise ValueError('Core ML mode requires draft, manifest and bridge paths')
                self._load_draft_config()
                from .backend import CoreMLBackbone
                self.backbone = CoreMLBackbone(manifest,bridge,timeout=timeout,verify_hashes=verify_hashes)
                self._validate_model_binding(manifest)
            model = load_model(self.target,lazy=True)
            if hasattr(model,'vision_tower'):
                del model.vision_tower
            gc.collect()
            self.lm = model.language_model
            if not hasattr(self.lm,'rollback_speculative_cache'):
                raise RuntimeError('mlx-vlm runtime lacks the required speculative verifier; install the pinned runtime')
            mx.eval(self.lm.parameters())
            self.tokenizer = AutoTokenizer.from_pretrained(self.target,local_files_only=True)
            if not vanilla:
                self._load_selector()
        except BaseException:
            self.close()
            raise

    def _load_draft_config(self):
        self.draft_config = json.loads((self.draft/'config.json').read_text())
        cfg = self.draft_config
        d = cfg.get('dflash_config',{})
        if (cfg.get('hidden_size'),cfg.get('num_hidden_layers'),cfg.get('vocab_size'),
            d.get('target_layer_ids'),d.get('selector_top_k'),d.get('selector_rank')) != (5120,5,248320,LAYER_IDS,16,256):
            raise ValueError('Draft checkpoint does not match the converted backbone/selector contract')
        self.mask_token = d['mask_token_id']
        self.input_scale = d.get('input_embedding_scale',1)
        self.output_multiplier = d.get('output_multiplier',1)
        self.softcap = d.get('final_logit_softcapping')

    def _validate_model_binding(self, manifest):
        from .checkpoint import selector_hashes
        data = json.loads(Path(manifest).read_text())
        binding = data.get('generation_binding')
        if not binding:
            raise ValueError('Generation needs generation_binding hashes; regenerate the local manifest')
        paths = {'draft_config':self.draft/'config.json','target_config':self.target/'config.json'}
        for key,path in paths.items():
            if hashlib.sha256(path.read_bytes()).hexdigest()!=binding[key]:
                raise ValueError(f'{key} does not match the verified manifest')
        # Only selector tensors are loaded by this package; bind their exact serialized values.
        if selector_hashes(self.draft/'model.safetensors') != binding['selector_tensors']:
            raise ValueError('Selector checkpoint does not match the verified manifest')
        for name,digest in binding['target_weights'].items():
            with (self.target/name).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:
                    raise ValueError(f'Target checkpoint mismatch: {name}')

    def _load_selector(self):
        from mlx import nn
        mx = self.mx
        self.projection = nn.Linear(5120,256,bias=False)
        self.predecessor = nn.Embedding(248320,256)
        self.successor = nn.Embedding(248320,256)
        # MLX arrays lazily map the source; retain/evaluate only selector parameters.
        weights = mx.load(str(self.draft/'model.safetensors'))
        self.projection.load_weights([('weight',weights['candidate_selector.hidden_projection.weight'])],strict=True)
        for module,name in [(self.predecessor,'predecessor'),(self.successor,'successor')]:
            key=f'candidate_selector.{name}_codebook'
            module.load_weights([('weight',weights[key if key in weights else key+'.weight'])],strict=True)
        mx.eval(self.projection.parameters(),self.predecessor.parameters(),self.successor.parameters())
        del weights
        gc.collect()
        mx.clear_cache()

    def _propose(self, anchor, features):
        mx = self.mx
        start = time.perf_counter()
        block = mx.array([[anchor]+[self.mask_token]*5])
        embedding = self.lm.model.embed_tokens(block)*self.input_scale
        mx.eval(embedding,features)
        hidden, calls = self.backbone.forward(np.array(embedding),np.array(features))
        h = mx.array(hidden[:,1:])
        logits = self.lm.lm_head(h.astype(mx.float16))*self.output_multiplier
        if self.softcap:
            logits = mx.tanh(logits/self.softcap)*self.softcap
        candidates = mx.argpartition(logits,-16,axis=-1)[...,-16:]
        unary = mx.take_along_axis(logits,candidates,axis=-1)
        projected = self.projection(h)
        predecessor = block[:,0]
        path = []
        for pos in range(5):
            scores = unary[:,pos]+mx.sum(self.predecessor(predecessor)[:,None]*projected[:,pos,None]*self.successor(candidates[:,pos]),axis=-1)
            selected = mx.argmax(scores,axis=-1)
            predecessor = mx.take_along_axis(candidates[:,pos],selected[:,None],axis=-1)[:,0]
            path.append(predecessor)
        ids = mx.stack(path,1)
        mx.eval(ids,logits)
        if not bool(mx.all(mx.isfinite(logits)).item()):
            raise RuntimeError('Non-finite draft logits')
        return ids[0].tolist(), {'draft_ms':(time.perf_counter()-start)*1000,
                                 'coreml_calls':len(calls)}

    def _verify(self, tokens, cache):
        mx = self.mx
        offset = cache[self.lm.model.fa_idx].offset
        positions = mx.arange(offset,offset+len(tokens),dtype=mx.int32)[None]
        out = self.lm(mx.array([tokens]),cache=cache,position_ids=positions,
                      capture_layer_ids=LAYER_IDS,speculative_verify=True)
        features = mx.concatenate(out.hidden_states[:5],-1)
        truth = mx.argmax(out.logits,-1)
        mx.eval(features,truth)
        if not bool(mx.all(mx.isfinite(features)).item()) or not bool(mx.all(mx.isfinite(out.logits)).item()):
            out.gdn_states.abort()
            raise RuntimeError('Non-finite target verification')
        return out, features, truth[0].tolist()

    def generate(self, prompt, *, max_tokens=128, enable_thinking=None):
        if self.closed:
            raise RuntimeError('Generator is closed')
        if not isinstance(prompt,str) or not prompt.strip() or max_tokens < 1:
            raise ValueError('Supply a nonempty prompt and max_tokens>=1')
        from .runtime import verifier_history
        with verifier_history(self.history):
            return self._generate(prompt,max_tokens,enable_thinking)

    def _generate(self,prompt,max_tokens,enable_thinking):
        mx = self.mx
        template = {} if enable_thinking is None else {'enable_thinking':enable_thinking}
        text = self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,**template)
        prompt_ids = self.tokenizer.encode(text,add_special_tokens=False)
        if not prompt_ids:
            raise ValueError('Prompt tokenization is empty')
        cache = self.lm.make_cache()
        self.lm._position_ids = None
        self.lm._rope_deltas = None
        sink = []
        start = time.perf_counter()
        capture = {} if self.vanilla else dict(capture_layer_ids=LAYER_IDS,hidden_sink=sink)
        hidden = self.lm.model(mx.array([prompt_ids]),cache=cache,**capture)
        logits = self.lm.lm_head(hidden[:,-1:])
        features = None if self.vanilla else mx.concatenate(sink,-1)
        mx.eval(logits,features)
        if not bool(mx.all(mx.isfinite(logits)).item()) or (features is not None and not bool(mx.all(mx.isfinite(features)).item())):
            raise RuntimeError('Non-finite prefill')
        tokens = [int(mx.argmax(logits[0,0]).item())]
        ttft = (time.perf_counter()-start)*1000
        # Preview defaults to the previously tested tokenizer EOS semantics.
        eos = {int(self.tokenizer.eos_token_id)}
        cycles = []
        if self.backbone:
            skip = max(0,len(prompt_ids)-31)
            self.backbone.reset(skip)
            features = features[:,skip:]
        while len(tokens)<max_tokens and tokens[-1] not in eos:
            cycle_start = time.perf_counter()
            before = len(prompt_ids)+len(tokens)-1
            if self.vanilla:
                # Ordinary MLX singleton forward: no speculative state/history capture.
                out=self.lm(mx.array([[tokens[-1]]]),cache=cache,
                            position_ids=mx.array([[before]],dtype=mx.int32))
                chosen=mx.argmax(out.logits[0,-1]).item()
                if not bool(mx.all(mx.isfinite(out.logits)).item()):
                    raise RuntimeError('Non-finite vanilla logits')
                offsets=[c.offset for c in cache if hasattr(c,'offset')]
                if any(pos!=before+1 for pos in offsets):
                    raise RuntimeError('Vanilla cache offset mismatch')
                tokens.append(int(chosen))
                cycles.append({'emitted':[int(chosen)],'cycle_ms':(time.perf_counter()-cycle_start)*1000})
                continue
            proposals, details = self._propose(tokens[-1],features) if self.backbone else ([],{})
            proposals = proposals[:min(self.proposal_count,max_tokens-len(tokens))]
            if self.backbone and self.backbone.offset!=before:
                raise RuntimeError('Draft context offset mismatch')
            t = time.perf_counter()
            out, captured, truth = self._verify([tokens[-1]]+proposals,cache)
            verify_ms = (time.perf_counter()-t)*1000
            try:
                emitted, accepted, verified_accepted = resolve_prefix(proposals,truth,max_tokens-len(tokens),eos)
                t = time.perf_counter()
                self.lm.rollback_speculative_cache(cache,out.gdn_states,accepted,len(proposals)+1)
                features = captured[:,:accepted+1]
                mx.eval([c.state for c in cache],features)
                offsets = [c.offset for c in cache if hasattr(c,'offset')]
                if any(pos!=before+accepted+1 for pos in offsets) or out.gdn_states.active:
                    raise RuntimeError('Target commit/offset mismatch')
                commit_ms = (time.perf_counter()-t)*1000
            finally:
                if out.gdn_states.active:
                    out.gdn_states.abort()
            tokens.extend(emitted)
            cycles.append(dict(proposals=proposals,accepted=accepted,verified_accepted=verified_accepted,
                               emitted=emitted,verification_ms=verify_ms,commit_ms=commit_ms,
                               cycle_ms=(time.perf_counter()-cycle_start)*1000,**details))
        elapsed = time.perf_counter()-start
        return {'text':self.tokenizer.decode(tokens,skip_special_tokens=True),'token_ids':tokens,
                'finish_reason':'eos' if tokens[-1] in eos else 'length',
                'prompt_tokens':len(prompt_ids),'generated_tokens':len(tokens),'ttft_ms':ttft,
                'generation_seconds':elapsed,'output_tokens_per_second':len(tokens)/elapsed,
                'proposal_count':self.proposal_count if self.backbone else 0,
                'mode':'coreml' if self.backbone else 'vanilla','history':self.history,
                'target_config_sha256':self.config_sha256,'cycles':cycles}

    def close(self):
        if self.closed:return
        self.closed = True
        if self.backbone:self.backbone.close()
        for name in ('lm','tokenizer','projection','predecessor','successor'):
            if hasattr(self,name):delattr(self,name)
        gc.collect()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
