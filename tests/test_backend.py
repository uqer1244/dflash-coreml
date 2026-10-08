import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dflash_coreml import CoreMLBackbone
from dflash_coreml.manifest import CONTRACT, load_manifest

class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        package = self.root/'dummy.mlpackage'
        package.mkdir()
        self.manifest = self.root/'manifest.json'
        row = {'path':package.name,'files':{}}
        self.manifest.write_text(json.dumps({'schema_version':1,'contract':CONTRACT,
                                            'packages':[row]*3,'context_packages':[row]*3}))
        self.bridge = self.root/'bridge'
        self.bridge.write_text(f'#!{sys.executable}\n'+'''import json,sys
config=json.load(open(sys.argv[1]))
print(json.dumps({'ready':True}),flush=True)
position=0
for line in sys.stdin:
 r=json.loads(line)
 if r['command']=='quit':break
 if r['command']=='reset':
  position=r['position'];print(json.dumps({'reset':True}),flush=True);continue
 position+=1
 print(json.dumps({'finite':True,'same_object_handoff':True,'context_only':r['command']=='commit','next_position':position}),flush=True)
''')
        self.bridge.chmod(0o755)
    def tearDown(self):
        self.temp.cleanup()
    def test_reset_commit_predict_and_close(self):
        with CoreMLBackbone(self.manifest,self.bridge,verify_hashes=False,timeout=3) as backend:
            backend.reset(30)
            h, rows = backend.forward(np.zeros((1,6,5120)),np.zeros((1,3,25600)))
            self.assertEqual(backend.offset,33)
            self.assertEqual([r['context_only'] for r in rows],[True,True,False])
            self.assertEqual(h.shape,(1,6,5120))
        self.assertIsNotNone(backend._proc.poll())
        self.assertFalse(Path(backend._temporary.name).exists())
        backend.close()
    def test_reject_invalid_inputs_before_state_update(self):
        with CoreMLBackbone(self.manifest,self.bridge,verify_hashes=False,timeout=3) as backend:
            backend.reset()
            with self.assertRaises(ValueError):
                backend.forward(np.full((1,6,5120),1e9),np.zeros((1,1,25600)))
            self.assertEqual(backend.offset,0)
            with self.assertRaises(ValueError):backend.reset(-1)
    def test_hashes_required_by_default(self):
        with self.assertRaisesRegex(ValueError,'No file hashes'):load_manifest(self.manifest,True)
    def test_hf_snapshot_manifest_and_weight_symlinks(self):
        import hashlib
        blobs=self.root/'blobs';blobs.mkdir()
        weight=blobs/'weight';weight.write_bytes(b'cached weight')
        package=self.root/'dummy.mlpackage'
        (package/'weight.bin').symlink_to(weight)
        data=json.loads(self.manifest.read_text())
        for key in ['packages','context_packages']:
            for row in data[key]:
                row['files']={'weight.bin':hashlib.sha256(weight.read_bytes()).hexdigest()}
        stored=blobs/'manifest';stored.write_text(json.dumps(data))
        self.manifest.unlink();self.manifest.symlink_to(stored)
        result=load_manifest(self.manifest,True)
        self.assertEqual(result['packages'],[str(package.resolve())]*3)
        weight.write_bytes(b'corrupted cache')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            load_manifest(self.manifest,True)

    def test_dead_bridge_cleans_up(self):
        self.bridge.write_text(f'#!{sys.executable}\n')
        with self.assertRaisesRegex(RuntimeError,'bridge exited'):
            CoreMLBackbone(self.manifest,self.bridge,verify_hashes=False,timeout=3)
    def test_timeout_terminates_process(self):
        self.bridge.write_text(f'#!{sys.executable}\nimport time\ntime.sleep(30)\n')
        with self.assertRaises(TimeoutError):
            CoreMLBackbone(self.manifest,self.bridge,verify_hashes=False,timeout=.1)

if __name__=='__main__':unittest.main()
