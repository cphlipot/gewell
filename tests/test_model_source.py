"""Local source provenance is recorded without an allowlist."""
import hashlib
from tools.model_source import load_model


def test_metadata_is_informational(tmp_path):
    for content in (b'{"model":"fine-tune"}', b'{"model":"repacked"}'):
        (tmp_path / "config.json").write_bytes(content)
        model = load_model(tmp_path)
        assert model["snapshot"] == str(tmp_path.resolve())
        assert model["config_sha256"] == hashlib.sha256(content).hexdigest()
        assert model["revision"] == ""
