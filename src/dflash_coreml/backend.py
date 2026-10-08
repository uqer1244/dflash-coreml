"""Single-request synchronous backbone adapter over the verified Swift protocol."""
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading

import numpy as np
from .manifest import load_manifest

SHAPES = {"x": (1,5120,1,6), "context_input": (1,25600,1,1),
          "cos": (1,1,7,128), "sin": (1,1,7,128),
          "mask": (1,1,6,37), "output": (1,5120,1,6)}

class CoreMLBackbone:
    """One independent bridge/state per instance. Caller serializes requests.

    Inputs are embeddings [1,6,5120] and committed features [1,N,25600].
    Output is hidden [1,6,5120]. LM head/Top16/selector are external.
    """
    def __init__(self, manifest, bridge, *, timeout=120, verify_hashes=True):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        config = load_manifest(manifest, verify_hashes)
        executable = Path(bridge).resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"Build the Swift bridge first: {executable}")
        self.timeout = timeout
        self.offset = None
        self.closed = False
        self._temporary = tempfile.TemporaryDirectory(prefix='dflash-coreml-')
        folder = Path(self._temporary.name)
        tensors, size = {}, 0
        for name, shape in SHAPES.items():
            tensors[name] = {"shape": list(shape), "offset": size}
            size = (size + int(np.prod(shape))*2 + 255)//256*256
        self._buffer = np.memmap(folder/'buffer.bin', dtype=np.uint8, mode='w+', shape=(size,))
        self._arrays = {name: np.ndarray(SHAPES[name], np.float16, buffer=self._buffer,
                                       offset=row['offset']) for name,row in tensors.items()}
        config.update(directory=str(folder), buffer=str(folder/'buffer.bin'),
                      buffer_bytes=size, tensors=tensors)
        path = folder/'config.json'
        path.write_text(json.dumps(config))
        self._stderr = (folder/'stderr.log').open('w+')
        self._proc = None
        self._replies = queue.Queue()
        try:
            self._proc = subprocess.Popen([str(executable), str(path)], stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE, stderr=self._stderr, text=True)
            self._reader = threading.Thread(target=self._read, daemon=True)
            self._reader.start()
            if not self._receive().get('ready'):
                raise RuntimeError('Bridge did not report readiness')
        except BaseException:
            self.close()
            raise

    def _read(self):
        for line in self._proc.stdout:
            self._replies.put(line)
        self._replies.put(None)

    def _receive(self):
        try:
            line = self._replies.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError('Core ML bridge timed out; backend closed') from None
        if line is None:
            self._stderr.flush()
            self._stderr.seek(0)
            detail = self._stderr.read()[-4000:]
            self.close()
            raise RuntimeError(f'Core ML bridge exited: {detail}')
        try:
            return json.loads(line)
        except ValueError:
            self.close()
            raise RuntimeError('Invalid bridge response') from None

    def _request(self, request):
        if self.closed:
            raise RuntimeError('Backend is closed')
        try:
            self._proc.stdin.write(json.dumps(request)+'\n')
            self._proc.stdin.flush()
        except (OSError, ValueError):
            self.close()
            raise RuntimeError('Core ML bridge is unavailable') from None
        return self._receive()

    def reset(self, position=0):
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise ValueError('position must be a nonnegative integer')
        response = self._request({'command':'reset', 'position':position})
        if not response.get('reset'):
            self.close()
            raise RuntimeError('Reset failed')
        self.offset = self.initial = position

    def forward(self, embedding, features):
        if self.closed:
            raise RuntimeError('Backend is closed')
        if self.offset is None:
            raise RuntimeError('Call reset before forward')
        embedding = np.asarray(embedding)
        features = np.asarray(features)
        if embedding.shape != (1,6,5120) or features.ndim != 3 or features.shape[0] != 1 or features.shape[1] < 1 or features.shape[2] != 25600:
            raise ValueError('Expected embedding [1,6,5120] and features [1,N,25600], N>=1')
        with np.errstate(over='ignore', invalid='ignore'):
            embedding = embedding.astype(np.float16)
            features = features.astype(np.float16)
        if not np.isfinite(embedding).all() or not np.isfinite(features).all():
            raise ValueError('Inputs must be finite after FP16 conversion')
        self._arrays['x'][:] = embedding.transpose(0,2,1)[:,:,None,:]
        calls = []
        for i in range(features.shape[1]):
            pos = self.offset
            self._arrays['context_input'][:] = features[:,i:i+1].transpose(0,2,1)[:,:,None,:]
            positions = np.arange(pos,pos+7,dtype=np.float32)
            inv = 1/(10000000**(np.arange(64,dtype=np.float32)/64))
            angles = positions[:,None]*inv
            angles = np.concatenate([angles,angles],-1)
            self._arrays['cos'][:] = np.cos(angles)[None,None].astype(np.float16)
            self._arrays['sin'][:] = np.sin(angles)[None,None].astype(np.float16)
            self._arrays['mask'].fill(0)
            valid = min(pos-self.initial+1,31)
            self._arrays['mask'][:,:,:,:31-valid] = -10000
            last = i == features.shape[1]-1
            row = self._request({'command':'predict' if last else 'commit',
                                 'position':pos, 'output':last})
            if not (row.get('finite') and row.get('same_object_handoff') and
                    row.get('context_only') == (not last) and row.get('next_position') == pos+1):
                self.close()
                raise RuntimeError('Bridge state/finite contract failed')
            self.offset += 1
            calls.append(row)
        hidden = self._arrays['output'].copy().squeeze(2).transpose(0,2,1)
        return hidden, calls

    def close(self):
        if self.closed:
            return
        self.closed = True
        proc = self._proc
        if proc is not None:
            if proc.poll() is None:
                try:
                    proc.stdin.write('{"command":"quit"}\n')
                    proc.stdin.flush()
                    proc.wait(timeout=5)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    proc.kill()
                    proc.wait(timeout=5)
            for stream in (proc.stdin,proc.stdout):
                stream.close()
            if hasattr(self, '_reader'):
                self._reader.join(timeout=5)
        self._stderr.close()
        self._arrays.clear()
        self._buffer._mmap.close()
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
