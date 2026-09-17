from __future__ import annotations

import copy
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bf16_artifact import ArtifactError, canonical_json_bytes, load_json_object
from tests.test_bf16_artifact import MODEL_REVISION
from tools.convert_bf16 import PreparedSource, Shard, convert, main as convert_main, parse_args as converter_args
from tools.serving_assets import (
    REQUIRED_ASSETS,
    publish_serving_assets,
    read_serving_assets,
    serving_metadata,
    validate_serving_assets,
    validate_serving_metadata,
)
from tools.verify_bf16 import validate_manifest
from tests.test_bf16_artifact import (
    SERVING_ASSET_BYTES,
    SERVING_ASSET_SHA256,
    _tiny_sources,
    _write_tiny_manifest_bundle,
)


class ServingAssetsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        for name, data in SERVING_ASSET_BYTES.items():
            (self.snapshot / name).write_bytes(data)
        self.output = self.root / "artifact"
        self.output.mkdir()

    def test_publishes_exact_files_and_deterministic_manifest_records(self) -> None:
        for name in ("tokenizer_config.json", "chat_template.jinja", "config.json", "generation_config.json"):
            (self.snapshot / name).write_bytes(b"unused source metadata")
        assets = read_serving_assets(self.snapshot)
        serving = publish_serving_assets(assets, self.output)
        self.assertEqual([item["file"] for item in serving["assets"]], ["tokenizer.json"])
        self.assertEqual(serving["repository"], "")
        self.assertEqual(serving["revision"], "")
        self.assertNotEqual(serving["revision"], MODEL_REVISION)
        self.assertEqual([item["file"] for item in serving["assets"]], list(REQUIRED_ASSETS))
        for record in serving["assets"]:
            name = record["file"]
            self.assertEqual(record, {"file": name, "byte_length": len(assets[name]),
                                      "sha256": SERVING_ASSET_SHA256[name]})
            self.assertEqual((self.output / name).read_bytes(), assets[name])
            self.assertFalse((self.output / name).is_symlink())
            self.assertNotEqual((self.output / name).stat().st_ino,
                                (self.snapshot / name).stat().st_ino)
        validate_serving_assets(self.output, serving)
        self.assertEqual(publish_serving_assets(assets, self.output), serving)
        self.assertEqual(set(path.name for path in self.output.iterdir()), set(REQUIRED_ASSETS))

    def test_reads_hub_source_symlinks_but_rejects_output_symlinks(self) -> None:
        name = REQUIRED_ASSETS[0]
        blob = self.snapshot / "blob"
        (self.snapshot / name).rename(blob)
        (self.snapshot / name).symlink_to(blob)
        assets = read_serving_assets(self.snapshot)
        (self.output / name).symlink_to(blob)
        with self.assertRaisesRegex(ArtifactError, "serving asset symlink"):
            publish_serving_assets(assets, self.output)
        with self.assertRaisesRegex(ArtifactError, "serving asset symlink"):
            validate_serving_assets(self.output, serving_metadata(assets))

    def test_rejects_missing_corrupt_or_conflicting_assets(self) -> None:
        name = REQUIRED_ASSETS[0]
        (self.snapshot / name).unlink()
        with self.assertRaisesRegex(ArtifactError, "cannot read serving asset"):
            read_serving_assets(self.snapshot)
        (self.snapshot / name).write_bytes(b"wrong bytes")
        self.assertEqual(read_serving_assets(self.snapshot)[name], b"wrong bytes")
        (self.output / name).write_bytes(b"existing")
        with self.assertRaisesRegex(ArtifactError, "refusing to replace"):
            publish_serving_assets(SERVING_ASSET_BYTES, self.output)
        self.assertEqual((self.output / name).read_bytes(), b"existing")
        self.assertEqual(len(list(self.output.iterdir())), 1)

    def test_manifest_inventory_is_strict_and_ordered(self) -> None:
        original = serving_metadata(SERVING_ASSET_BYTES)
        mutations = (
            lambda value: value.update(extra=True),
            lambda value: value.pop("repository"),
            lambda value: value.pop("revision"),
            lambda value: value.update(revision=None),
            lambda value: value["assets"].pop(),
            lambda value: value["assets"].append(value["assets"][0]),
            lambda value: value["assets"][0].update(file="config.json"),
            lambda value: value["assets"][0].update(file="../tokenizer.json"),
            lambda value: value["assets"][0].update(byte_length=True),
            lambda value: value["assets"][0].update(byte_length=0),
            lambda value: value["assets"][0].update(sha256="invalid"),
            lambda value: value["assets"][0].update(extra=True),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                value = copy.deepcopy(original)
                mutate(value)
                with self.assertRaises(ArtifactError):
                    validate_serving_metadata(value)

    def test_verifies_assets_against_manifest_after_copy(self) -> None:
        serving = publish_serving_assets(SERVING_ASSET_BYTES, self.output)
        target = self.output / REQUIRED_ASSETS[0]
        original = target.read_bytes()
        target.write_bytes(b"X" * len(original))
        with self.assertRaisesRegex(ArtifactError, "SHA-256 mismatch"):
            validate_serving_assets(self.output, serving)
        target.write_bytes(original + b"extra")
        with self.assertRaisesRegex(ArtifactError, "byte length mismatch"):
            validate_serving_assets(self.output, serving)
        target.unlink()
        with self.assertRaisesRegex(ArtifactError, "cannot read serving asset"):
            validate_serving_assets(self.output, serving)

    def test_regular_verifier_rejects_schema_4_and_modified_serving_bytes(self) -> None:
        artifact, manifest_path, manifest, verified, specs, contract = _write_tiny_manifest_bundle(self.output)
        manifest["schema_version"] = 4
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        with self.assertRaisesRegex(ArtifactError, "unsupported manifest schema"):
            validate_manifest(manifest_path, artifact, verified, specs, contract)
        manifest["schema_version"] = 6
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        (self.output / "tokenizer.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ArtifactError, "serving asset byte length mismatch"):
            validate_manifest(manifest_path, artifact, verified, specs, contract)

    def test_regular_verifier_requires_the_independent_serving_revision(self) -> None:
        artifact, manifest_path, original, verified, specs, contract = _write_tiny_manifest_bundle(self.output)
        for mutate in (
            lambda value: value["serving"].pop("revision"),
        ):
            with self.subTest(mutation=mutate):
                manifest = copy.deepcopy(original)
                mutate(manifest)
                manifest_path.write_bytes(canonical_json_bytes(manifest))
                with self.assertRaisesRegex(ArtifactError, "manifest serving"):
                    validate_manifest(manifest_path, artifact, verified, specs, contract)

    def test_cli_defaults_and_overrides_select_serving_snapshot_independently(self) -> None:
        self.assertEqual(converter_args(["--snapshot", str(self.snapshot), "--plan"]).serving_snapshot, None)
        self.assertEqual(converter_args(["--snapshot", str(self.snapshot), "--plan", "--serving-snapshot", str(self.snapshot)]).serving_snapshot,
                         self.snapshot)
        contract = object()
        for arguments, snapshot in (([], None),
                                    (["--serving-snapshot", str(self.snapshot)], self.snapshot)):
            with patch("tools.convert_bf16.load_model", return_value=contract), \
                    patch("tools.convert_bf16.convert", return_value=(self.output / "weights.gwt", self.output / "manifest.json")) as conversion, \
                    patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(convert_main(["--snapshot", str(self.snapshot), "--output", str(self.output), *arguments]), 0)
                self.assertEqual(conversion.call_args.kwargs["serving_snapshot"], snapshot)

    def test_converter_publishes_assets_and_failed_copy_cannot_publish_manifest(self) -> None:
        original_artifact, _, _, _, specs, contract = _write_tiny_manifest_bundle(self.output)
        # Source configuration and templates are not runtime bundle assets.
        (self.output / "tokenizer_config.json").write_bytes(b"old weight snapshot tokenizer config")
        (self.output / "chat_template.jinja").write_bytes(b"old weight snapshot chat template")
        sources, _ = _tiny_sources(self.output, specs)
        prepared = PreparedSource(
            snapshot=self.output, index_path=self.output / "model.safetensors.index.json",
            specs=specs, tensors=sources,
            shards=tuple(Shard(name, self.output / name, size, digest, (0, 0, size, 0))
                         for name, (size, digest) in {source.shard_name: (source.path.stat().st_size, "0" * 64) for source in sources}.items()),
        )
        destination = self.root / "converted"
        with patch("tools.convert_bf16.prepare_source", return_value=prepared), \
                patch("tools.convert_bf16.assert_source_unchanged"):
            artifact, manifest_path = convert(contract, destination, serving_snapshot=self.snapshot)
            manifest = load_json_object(manifest_path, canonical=True)
            self.assertTrue(artifact.is_file())
            self.assertEqual(manifest["schema_version"], 6)
            self.assertEqual(manifest["model"]["revision"], MODEL_REVISION)
            self.assertEqual(manifest["serving"]["revision"], "")
            self.assertEqual(artifact.read_bytes(), original_artifact.read_bytes())
            validate_serving_assets(destination, manifest["serving"])
            self.assertEqual({path.name for path in destination.iterdir()},
                             {"weights.gwt", "manifest.json", "tokenizer.json"})

            failed = self.root / "failed"
            with patch("tools.convert_bf16.publish_serving_assets", side_effect=OSError("copy failed")):
                with self.assertRaisesRegex(OSError, "copy failed"):
                    convert(contract, failed, serving_snapshot=self.snapshot)
            self.assertFalse((failed / "manifest.json").exists())
            self.assertFalse((failed / "weights.gwt").exists())
            self.assertFalse(list(failed.glob("*.partial")))


class PinnedServingAssetsTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("GEWELL_TEST_SERVING_SNAPSHOT"),
                         "set GEWELL_TEST_SERVING_SNAPSHOT to opt into copying pinned tokenizer assets")
    def test_real_pinned_assets_are_copied_and_verified_offline(self) -> None:
        snapshot = Path(os.environ["GEWELL_TEST_SERVING_SNAPSHOT"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            serving = publish_serving_assets(read_serving_assets(snapshot), output)
            validate_serving_assets(output, serving)
            self.assertEqual([asset["file"] for asset in serving["assets"]], ["tokenizer.json"])
            self.assertGreater(sum(item["byte_length"] for item in serving["assets"]), 30_000_000)
            for name in REQUIRED_ASSETS:
                self.assertFalse((output / name).is_symlink())
                self.assertEqual((output / name).read_bytes(), (snapshot / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
