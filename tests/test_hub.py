import tempfile
import unittest
from pathlib import Path
from dflash_coreml.hub import _materialize_coreml_snapshot

class HubTests(unittest.TestCase):
    def test_materializes_hf_package_links_without_modifying_blobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);snapshot=root/'snapshots'/'revision'
            package=snapshot/'models'/'full-0.mlpackage';package.mkdir(parents=True)
            blob=root/'blobs'/'weight';blob.parent.mkdir();blob.write_bytes(b'original weights')
            weight=package/'Data'/'weight.bin';weight.parent.mkdir();weight.symlink_to(blob)
            manifest=root/'blobs'/'manifest';manifest.write_bytes(b'{}')
            (snapshot/'model-manifest.json').symlink_to(manifest)
            result=_materialize_coreml_snapshot(snapshot)
            prepared=result/weight.relative_to(snapshot)
            self.assertFalse(prepared.is_symlink())
            self.assertFalse((result/'model-manifest.json').is_symlink())
            self.assertEqual(prepared.read_bytes(),blob.read_bytes())
            self.assertEqual(_materialize_coreml_snapshot(snapshot),result)
            prepared.write_bytes(b'changed prepared copy')
            self.assertEqual(blob.read_bytes(),b'original weights')

if __name__=='__main__':unittest.main()
