import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mech_interp.snapshot import DEFAULT_MODEL, DEFAULT_REVISION, FILES, verify_model_source


class OfflineSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        # "git hash-object" for hello followed by a newline. LFS uses raw SHA-256.
        self.config = self.path / "config.json"
        self.config.write_bytes(b"hello\n")
        self.weights = self.path / "model.safetensors"
        self.weights.write_bytes(b"fixture")
        fingerprints = {
            self.config.name: (6, "ce013625030ba8dba906f756967f9e9ca394464a"),
            self.weights.name: (7, hashlib.sha256(b"fixture").hexdigest()),
        }
        mock_files = patch.dict(FILES, fingerprints, clear=True)
        mock_files.start()
        self.addCleanup(mock_files.stop)

    def verify(self):
        verify_model_source(DEFAULT_MODEL, DEFAULT_REVISION, self.path)

    def test_valid_snapshot_with_optional_hub_metadata(self):
        (self.path / "README.md").write_bytes(b"card")
        (self.path / ".cache").mkdir()
        self.verify()

    def test_modified_git_or_lfs_content_rejected_even_at_correct_length(self):
        for path in (self.config, self.weights):
            with self.subTest(path=path):
                content = path.read_bytes()
                path.write_bytes(b"x" * len(content))
                with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                    self.verify()
                path.write_bytes(content)

    def test_missing_or_override_files_rejected(self):
        extra = self.path / "adapter_config.json"
        extra.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unexpected files"):
            self.verify()
        extra.unlink()
        self.weights.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or incorrect size"):
            self.verify()

    def test_reject_unverified_local_identity_and_allow_hosted_models(self):
        for model_id, revision in ((DEFAULT_MODEL, "a" * 40), ("org/other", DEFAULT_REVISION)):
            with self.subTest(model_id=model_id), self.assertRaisesRegex(ValueError, "--model-path"):
                verify_model_source(model_id, revision, self.path)
        with self.assertRaisesRegex(ValueError, "Hub identifier"):
            verify_model_source(str(self.path), DEFAULT_REVISION, None)
        verify_model_source("org/other", "a" * 40, None)


if __name__ == "__main__":
    unittest.main()
