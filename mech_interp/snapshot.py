"""Verify private/offline copies against the upstream model revision.

The filenames, lengths, Git blob IDs, and LFS SHA-256 come from Qwen's public tree:
https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775
For regular files use the tree entry's oid; for model.safetensors use lfs.oid,
which hashes the model bytes rather than the Git LFS pointer text.
"""

import hashlib
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
FILES = {
    "config.json": (659, "0dbb161213629a23f0fc00ef286e6b1e366d180f"),
    "generation_config.json": (242, "dfc11073787daf1b0f9c0f1499487ab5f4c93738"),
    "merges.txt": (1671839, "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0"),
    "model.safetensors": (988097824, "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"),
    "tokenizer.json": (7031645, "443909a61d429dff23010e5bddd28ff530edda00"),
    "tokenizer_config.json": (7305, "07bfe0640cb5a0037f9322287fbfc682806cf672"),
    "vocab.json": (2776833, "4783fe10ac3adce15ac8f358ef5462739852c569"),
}
PASSIVE_FILES = {".git", ".cache", ".gitattributes", "LICENSE", "README.md"}


def verify_model_source(model_id, revision, model_path):
    """Require an authenticated offline snapshot or a hosted model identifier."""
    if model_path is None:
        if Path(model_id).exists():
            raise ValueError("Use a Hub identifier for --model and --model-path for local files")
        return
    if (model_id, revision) != (DEFAULT_MODEL, DEFAULT_REVISION):
        raise ValueError(
            f"--model-path supports only {DEFAULT_MODEL}@{DEFAULT_REVISION}; "
            "use a hosted --model and --revision for another model"
        )
    directory = Path(model_path)
    if not directory.is_dir():
        raise ValueError(f"Missing offline snapshot directory: {directory}")
    # Additional loader inputs (for example an adapter config) could override the pin.
    extra = {item.name for item in directory.iterdir()} - FILES.keys() - PASSIVE_FILES
    if extra:
        raise ValueError(f"Unexpected files in offline snapshot: {', '.join(sorted(extra))}")

    for filename, (size, expected) in FILES.items():
        path = directory / filename
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing or incorrect size for pinned file: {filename}")
        if filename == "model.safetensors":
            digest = hashlib.sha256()
        else:
            digest = hashlib.sha1(usedforsecurity=False)
            digest.update(f"blob {size}\0".encode("ascii"))
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Hash mismatch for pinned file: {filename}")
