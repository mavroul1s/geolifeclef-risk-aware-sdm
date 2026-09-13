import io
import json
from pathlib import Path
import zipfile

import pytest

from scripts.kaggle_artifacts import (
    KaggleReader,
    SafeKaggleError,
    V20_ALLOWLIST,
    decode_csv_prefix,
    file_summary,
    safe_name,
    sanitize_text,
)


def test_redaction_removes_signed_urls_and_known_credentials():
    rendered = sanitize_text(
        'failure https://storage.example/file?X-Goog-Signature=abc keyvalue api_key=other token=third',
        ('keyvalue',),
    )
    assert "https://" not in rendered
    assert "Signature" not in rendered
    assert "keyvalue" not in rendered
    assert "other" not in rendered
    assert "third" not in rendered


@pytest.mark.parametrize("name", ("../escape", "/absolute", "C:/escape", "a\\escape", "a/./b", "a/../b"))
def test_artifact_name_cannot_escape_destination(name):
    with pytest.raises(SafeKaggleError):
        safe_name(name)


def test_file_summary_contains_no_signed_url():
    summary = file_summary({"fileName": "some/path.npy", "url": "https://private.example/?signed=yes", "totalBytes": 12})
    assert summary == {"name": "some/path.npy", "bytes": 12}


def test_zip_prefix_decode_is_bounded():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.csv", "surveyId,speciesId\n1,23\n" + "2,99\n" * 100000)
    decoded = decode_csv_prefix(stream.getvalue(), 128)
    assert decoded.startswith(b"surveyId,speciesId\n1,23\n")
    assert len(decoded) == 128


def test_version_check_refuses_after_kernel_advances():
    client = KaggleReader.__new__(KaggleReader)
    client.json = lambda endpoint, args: {"metadata": {"currentVersionNumber": 21}}
    with pytest.raises(SafeKaggleError, match="differs"):
        client.output(version=20)


def test_version_check_refuses_push_race():
    client = KaggleReader.__new__(KaggleReader)
    calls = iter(({"metadata": {"currentVersionNumber": 20}}, {"files": []}, {"metadata": {"currentVersionNumber": 21}}))
    client.json = lambda endpoint, args: next(calls)
    with pytest.raises(SafeKaggleError, match="advanced"):
        client.output(version=20)


def test_version_checked_output_does_not_retrieve_all_files():
    client = KaggleReader.__new__(KaggleReader)
    output = {"files": [{"fileName": "artifact", "url": "https://signed.example"}]}
    calls = iter(({"metadata": {"currentVersionNumber": 20}}, output, {"metadata": {"currentVersionNumber": 20}}))
    client.json = lambda endpoint, args: next(calls)
    assert client.output(version=20) is output


def test_download_rejects_non_allowlisted_files_before_network(tmp_path):
    client = KaggleReader.__new__(KaggleReader)
    client.output = lambda: pytest.fail("Network listing must not run")
    with pytest.raises(SafeKaggleError, match="allowlisted"):
        client.download_v20(["../secret"], tmp_path)
    assert "challenger_2025_test_probabilities.npy" in V20_ALLOWLIST


class Response:
    status_code = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def iter_content(self, chunk_size):
        yield b"payload"


def test_failed_size_check_retains_no_partial_artifact(tmp_path):
    client = KaggleReader.__new__(KaggleReader)
    client.requests = type("FakeRequests", (), {"get": lambda *a, **k: Response()})()
    destination = tmp_path / "probabilities.npy"
    with pytest.raises(SafeKaggleError, match="byte count"):
        client.download_url("https://signed.example", destination, expected_bytes=20)
    assert not destination.exists()
    assert not destination.with_suffix(".npy.partial").exists()


def test_complete_download_reports_digest_without_url(tmp_path):
    client = KaggleReader.__new__(KaggleReader)
    client.requests = type("FakeRequests", (), {"get": lambda *a, **k: Response()})()
    destination = tmp_path / "probabilities.npy"
    report = client.download_url("https://signed.example", destination, expected_bytes=7)
    assert destination.read_bytes() == b"payload"
    assert report["bytes"] == 7
    assert len(report["sha256"]) == 64
    assert "https://" not in json.dumps(report)
