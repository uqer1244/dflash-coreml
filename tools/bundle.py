"""Build an allowlisted code-only source ZIP and checksums; never upload."""
import hashlib,json,zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
dist=root/'dist';dist.mkdir(exist_ok=True)
allowed_files={'README.md','LICENSE','NOTICE','CHANGELOG.md','pyproject.toml','MANIFEST.in','.gitignore'}
allowed_dirs={'src','native','examples','docs','tests','tools','.github','LICENSES'}
files=[]
for p in sorted(root.rglob('*')):
 if not p.is_file() or p.is_symlink() or p.name=='.DS_Store':continue
 rel=p.relative_to(root)
 if '__pycache__' in rel.parts or any(part.endswith('.egg-info') for part in rel.parts) or p.suffix in ('.pyc','.bin'):continue
 if (len(rel.parts)==1 and p.name in allowed_files) or rel.parts[0] in allowed_dirs:
  files.append(p)
archive=dist/'dflash-coreml-0.1.0a1-source.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,'dflash-coreml-0.1.0a1/'+str(p.relative_to(root)))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(dist/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
rows=[]
for p in sorted(dist.iterdir()):
 if p.suffix in ('.zip','.whl') or p.name=='SOURCE_MANIFEST.json':
  rows.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name)
(dist/'SHA256SUMS').write_text('\n'.join(rows)+'\n')
print(f'{len(files)} source files; archive {archive.name}, {archive.stat().st_size} bytes')
