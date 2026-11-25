import hashlib
import tempfile
import unittest
from pathlib import Path

from eval.review_prior_misses import checked_json, count_decisions


class LegacyReviewTests(unittest.TestCase):
    def test_count_supported_controls_and_each_decoy_separately(self):
        records = [
            {"label": 1, "condition": "prior", "flagged": {"old": False, "new": True}},
            {"label": 1, "condition": "absent", "flagged": {"old": True, "new": True}},
            {"label": 0, "condition": "prior", "flagged": {"old": False, "new": True}},
            {"label": 0, "condition": "absent", "flagged": {"old": False, "new": False}},
        ]
        result = count_decisions(records, ("old", "new"))
        self.assertEqual(result["old"]["prior_caught"], {"flagged": 0, "total": 1})
        self.assertEqual(result["new"]["prior_caught"], {"flagged": 1, "total": 1})
        self.assertEqual(result["new"]["supported_flagged"], {"flagged": 1, "total": 2})
        self.assertEqual(result["new"]["absent_caught"], {"flagged": 1, "total": 1})

    def test_exact_published_bytes_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.json"
            path.write_bytes(b'{"n": 1}')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(checked_json(path, expected), {"n": 1})
            path.write_bytes(b'{"n":1}')
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                checked_json(path, expected)
