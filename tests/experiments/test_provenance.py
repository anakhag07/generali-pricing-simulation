from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

import experiments.provenance as provenance
from experiments.provenance import array_sha256, file_record, file_sha256, write_provenance


def test_file_records_and_json_are_stable(tmp_path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("same bytes\n", encoding="utf-8")

    first = file_record(source)
    second = file_record(source)

    assert first == second
    assert first["sha256"] == file_sha256(source)

    output = write_provenance(
        tmp_path / "provenance.json",
        {"array": np.asarray([1, 2]), "source": first},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["array"] == [1, 2]
    assert payload["source"] == first


def test_array_hash_includes_shape_and_dtype() -> None:
    values = np.asarray([1, 2, 3, 4], dtype=np.int64)

    assert array_sha256(values) == array_sha256(values.copy())
    assert array_sha256(values) != array_sha256(values.reshape(2, 2))
    assert array_sha256(values) != array_sha256(values.astype(np.float64))


def test_source_hash_cache_detects_edits_replacements_and_tree_changes(tmp_path, monkeypatch) -> None:
    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    time.sleep(1.01)  # Exercise caching after the filesystem timestamp tick.
    calls = []
    original_hash = provenance.file_sha256

    def tracked_hash(path):
        calls.append(path)
        return original_hash(path)

    monkeypatch.setattr(provenance, "file_sha256", tracked_hash)
    first = provenance.source_hashes(tmp_path)
    assert provenance.source_hashes(tmp_path) == first
    assert calls == [source]

    original_stat = source.stat()
    source.write_text("value = 2\n")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    edited = provenance.source_hashes(tmp_path)
    assert edited != first
    assert len(calls) == 2

    replacement = tmp_path / "replacement.tmp"
    replacement.write_text("value = 3\n")
    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    replacement.replace(source)
    replaced = provenance.source_hashes(tmp_path)
    assert replaced != edited
    assert len(calls) == 3

    added = tmp_path / "extra.py"
    added.write_text("# new dependency\n")
    assert set(provenance.source_hashes(tmp_path)) == {"module.py", "extra.py"}
    source.unlink()
    assert set(provenance.source_hashes(tmp_path)) == {"extra.py"}


def test_recent_source_edits_with_identical_stat_fields_are_not_cached(tmp_path, monkeypatch) -> None:
    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    signature = provenance._source_signature(source)
    monkeypatch.setattr(provenance, "_source_signature", lambda path: signature)
    first = provenance.source_hashes(tmp_path)
    source.write_text("value = 2\n")
    assert provenance.source_hashes(tmp_path) != first


def test_source_edit_during_hashing_is_rejected_and_not_cached(tmp_path, monkeypatch) -> None:
    source = tmp_path / "module.py"
    source.write_text("before\n")
    original_hash = provenance.file_sha256

    def edit_during_hash(path):
        digest = original_hash(path)
        path.write_text("after\n")
        return digest

    monkeypatch.setattr(provenance, "file_sha256", edit_during_hash)
    with pytest.raises(ValueError, match="Source changed while hashing"):
        provenance.source_hashes(tmp_path)
    monkeypatch.setattr(provenance, "file_sha256", original_hash)
    assert provenance.source_hashes(tmp_path) == {"module.py": original_hash(source)}
