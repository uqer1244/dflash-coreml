import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dflash_coreml.generation import resolve_prefix

class PrefixTests(unittest.TestCase):
    def test_first_rejection(self):
        self.assertEqual(resolve_prefix([10,11],[12,13,14],8,{99}),([12],0,0))
    def test_partial_rejection(self):
        self.assertEqual(resolve_prefix([10,11],[10,12,13],8,{99}),([10,12],1,1))
    def test_full_acceptance_bonus(self):
        self.assertEqual(resolve_prefix([10,11],[10,11,12],8,{99}),([10,11,12],2,2))
    def test_eos_in_accepted_prefix(self):
        self.assertEqual(resolve_prefix([10,99],[10,99,12],8,{99}),([10,99],2,2))
    def test_token_limit_truncates_bonus(self):
        self.assertEqual(resolve_prefix([10,11],[10,11,12],2,{99}),([10,11],2,2))
    def test_eos_in_correction(self):
        self.assertEqual(resolve_prefix([10,11],[10,99,12],8,{99}),([10,99],1,1))
