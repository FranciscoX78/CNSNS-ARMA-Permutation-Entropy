import tempfile
import unittest
from pathlib import Path
from importlib.machinery import SourceFileLoader
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
prep=SourceFileLoader('prep',str(ROOT/'scripts/prepare_ims_set2.py')).load_module()
class PrepTests(unittest.TestCase):
    def test_requires_exactly_984_records(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'984'):
                prep.run(t,Path(t)/'out')
if __name__=='__main__':unittest.main()
