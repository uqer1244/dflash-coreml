"""Pinned public MLX runtime imports; never add an app bundle to sys.path."""
from contextlib import contextmanager
import importlib.metadata
import threading

_LOCK=threading.RLock()
VERSIONS={'mlx':'0.32.2','mlx-vlm':'0.7.1','mlx-lm':'0.32.0','transformers':'5.17.0'}

def imports():
    for name,expected in VERSIONS.items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError('Install standalone runtime dependencies: pip install ".[generate]"') from None
        if actual!=expected:
            raise RuntimeError(f'{name}=={expected} required for this preview; found {actual}')
    import mlx.core as mx
    from .target import load_model
    from transformers import AutoTokenizer
    try:
        from mlx_vlm.models.qwen3_5 import speculative_verifier
    except ImportError as exc:
        raise RuntimeError('Runtime lacks required target/verifier support') from exc
    mx.set_cache_limit(128*2**20)
    return mx,load_model,AutoTokenizer

@contextmanager
def verifier_history(mode):
    if mode not in ('stock','deferred'):raise ValueError('history must be stock or deferred')
    with _LOCK:
        if mode=='stock':
            yield
            return
        from . import history
        sv=history.sv
        cache=history.ArraysCache
        original_update=sv.gated_delta_update
        original_select=cache._select_speculative_states
        sv.gated_delta_update=history.update
        cache._select_speculative_states=staticmethod(history.select)
        try:yield
        finally:
            sv.gated_delta_update=original_update
            cache._select_speculative_states=staticmethod(original_select)
