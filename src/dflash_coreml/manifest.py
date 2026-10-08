"""Explicit model configuration, independent of a user's oMLX installation."""
import hashlib
import json
from pathlib import Path

CONTRACT = {"hidden_size": 5120, "feature_size": 25600, "block_size": 6,
            "capacity": 31, "rope_theta": 10000000, "shards": 3}

def load_manifest(path, verify_hashes=False):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or data.get("contract") != CONTRACT:
        raise ValueError("Unsupported model schema or backbone contract")
    for key in ("packages", "context_packages"):
        rows = data.get(key, [])
        if len(rows) != 3:
            raise ValueError(f"{key} must contain three shards")
        paths = []
        for row in rows:
            package = (path.parent / row["path"]).resolve()
            if not package.is_dir() or package.suffix != ".mlpackage":
                raise ValueError(f"Missing Core ML package: {package}")
            expected = row.get("files", {})
            if verify_hashes and not expected:
                raise ValueError(f"No file hashes for {package}")
            if verify_hashes:
                actual_files = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
                if actual_files != set(expected):
                    raise ValueError(f"Package file inventory differs: {package}")
                for name, digest in expected.items():
                    f = (package / name).resolve()
                    if not f.is_relative_to(package):
                        raise ValueError("Package hash path escapes package")
                    with f.open('rb') as stream:
                        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if actual != digest:
                        raise ValueError(f"Package hash mismatch: {f}")
            paths.append(str(package))
        data[key] = paths
    return data
