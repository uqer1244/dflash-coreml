"""Repack referenced context constants without changing the graph (coremltools 9)."""
import argparse, hashlib, json
from pathlib import Path
import coremltools as ct
from coremltools.libmilstoragepython import _BlobStorageReader, _BlobStorageWriter


def values(message):
    if not hasattr(message, 'DESCRIPTOR'):
        return
    if message.DESCRIPTOR.full_name == 'CoreML.Specification.MILSpec.Value' and message.HasField('blobFileValue'):
        yield message
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        if field.is_repeated:
            children = [value[k] for k in sorted(value)] if field.message_type.GetOptions().map_entry else value
            for child in children:
                yield from values(child)
        else:
            yield from values(value)


def compact(source, destination):
    assert not destination.exists(), destination
    spec = ct.utils.load_spec(str(source))
    original = spec.SerializeToString(deterministic=True)
    refs = list(values(spec))
    assert refs and all(Path(v.blobFileValue.fileName).name == 'weight.bin' for v in refs), 'Expected one weight.bin' 
    weights = destination.parent / (destination.name + '.weights')
    weights.mkdir()
    output = weights / 'weight.bin'
    writer = _BlobStorageWriter(str(output))
    readers, offsets, audit = {}, {}, []
    methods = {'FLOAT16': 'fp16', 'INT8': 'int8', 'FLOAT32': 'float', 'UINT8': 'uint8'}
    for value in refs:
        blob = value.blobFileValue
        dtype = ct.proto.MIL_pb2.DataType.Name(value.type.tensorType.dataType)
        method = methods[dtype]
        old = (blob.fileName, blob.offset, dtype)
        path = source / 'Data/com.apple.CoreML/weights' / Path(blob.fileName).name
        reader = readers.setdefault(str(path), _BlobStorageReader(str(path)))
        data = getattr(reader, 'read_' + method + '_data')(blob.offset)
        if old not in offsets:
            offsets[old] = getattr(writer, 'write_' + method + '_data')(data)
            audit.append({'dtype': dtype, 'bytes': data.nbytes, 'sha256': hashlib.sha256(data.tobytes()).hexdigest()})
        blob.offset = offsets[old]
    writer = None
    reader = _BlobStorageReader(str(output))
    oldspec = ct.proto.Model_pb2.Model(); oldspec.ParseFromString(original)
    for value, oldvalue in zip(refs, values(oldspec)):
        dtype = ct.proto.MIL_pb2.DataType.Name(value.type.tensorType.dataType)
        newdata = getattr(reader, 'read_' + methods[dtype] + '_data')(value.blobFileValue.offset)
        # Compare to original protobuf reference, independently from the write cache.
        oldpath = source / 'Data/com.apple.CoreML/weights' / Path(oldvalue.blobFileValue.fileName).name
        olddata = getattr(readers[str(oldpath)], 'read_' + methods[dtype] + '_data')(oldvalue.blobFileValue.offset)
        assert newdata.tobytes() == olddata.tobytes()
    restored = ct.proto.Model_pb2.Model(); restored.CopyFrom(spec)
    oldspec = ct.proto.Model_pb2.Model(); oldspec.ParseFromString(original)
    for new, old in zip(values(restored), values(oldspec)):
        new.blobFileValue.CopyFrom(old.blobFileValue)
    assert restored.SerializeToString(deterministic=True) == original, 'Graph changed beyond blob offsets'
    ct.models.MLModel(spec, weights_dir=str(weights), skip_model_load=True).save(str(destination))
    import shutil
    shutil.rmtree(weights)
    return {'package': source.name, 'before_bytes': sum(p.stat().st_size for p in source.rglob('*') if p.is_file()),
            'after_bytes': sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()),
            'graph_unchanged_except_blob_offsets': True, 'constants_byte_exact': True, 'constants': audit}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('bundle', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(); args.output.mkdir()
    rows = [compact(args.bundle / 'models' / f'context-{i}.mlpackage', args.output / f'context-{i}.mlpackage') for i in range(3)]
    (args.output / 'audit.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2))
