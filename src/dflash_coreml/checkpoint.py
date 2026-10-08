"""Hash exact serialized selector bytes, including BF16, without tensor conversion."""
import hashlib
import json
import struct

def selector_hashes(path):
    result = {}
    with open(path,'rb') as stream:
        size_bytes=stream.read(8)
        if len(size_bytes)!=8:raise ValueError('Invalid safetensors file')
        size=struct.unpack('<Q',size_bytes)[0]
        if size>64*2**20:raise ValueError('Safetensors header exceeds limit')
        header=json.loads(stream.read(size))
        for name,row in header.items():
            if not name.startswith('candidate_selector.'):continue
            start,end=row['data_offsets']
            if not 0<=start<=end:raise ValueError('Invalid selector offsets')
            stream.seek(8+size+start)
            remaining=end-start;digest=hashlib.sha256()
            while remaining:
                chunk=stream.read(min(remaining,2**20))
                if not chunk:raise ValueError('Truncated selector tensor')
                digest.update(chunk);remaining-=len(chunk)
            result[name]=digest.hexdigest()
    if len(result)!=3:raise ValueError('Expected exactly three selector tensors')
    return result
