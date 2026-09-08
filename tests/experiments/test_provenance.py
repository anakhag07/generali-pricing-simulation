from __future__ import annotations

import json

import numpy as np

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
