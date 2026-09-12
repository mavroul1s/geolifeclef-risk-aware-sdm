"""Read-only, bounded Kaggle access without exposing credentials or signed URLs.

Endpoints were checked against the installed Kaggle 1.6.17 client and live API.
The live API rejects its advertised kernelVersionNumber files argument. Output
access therefore checks currentVersionNumber before and after listing and
refuses historical downloads after the kernel has advanced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://www.kaggle.com/api/v1"
KERNEL = "con1los/geolifeclef-risk-aware-sdm-phase-1"
COMPETITION = "geolifeclef-2025"
V20_ROOT = "geolifeclef-risk-aware-sdm/artifacts/environmental_challenger/"
MODELS = ("reference_2025", "challenger_2025", "challenger_3407")
V20_ALLOWLIST = frozenset(
    ["data_manifest.json", "frozen_policy.json", "challenger_report.json", "GLC25_PA_submission.csv"]
    + [f"{m}_best.pt" for m in MODELS]
    + [f"{m}_{s}_probabilities.npy" for m in MODELS for s in ("calibration", "audit", "test")]
)


class SafeKaggleError(RuntimeError):
    """A diagnostic deliberately containing no server response or request URL."""


def sanitize_text(text: str, secrets: tuple[str, ...] = ()) -> str:
    for secret in sorted(filter(None, secrets), key=len, reverse=True):
        text = text.replace(secret, "[redacted]")
    text = re.sub(r"https?://[^\s<>\"']+", "[URL redacted]", text)
    text = re.sub(r"(?i)(authorization|api[_-]?key|password|token)\s*[:=]\s*[^\s,}]+", r"\1=[redacted]", text)
    return text


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or any(p in ("..", ".") for p in name.split("/")) or ":" in name:
        raise SafeKaggleError("Unsafe artifact filename rejected.")
    return name


def file_summary(item: dict) -> dict:
    return {"name": safe_name(item.get("fileName", item.get("name", ""))),
            "bytes": int(item.get("totalBytes", item.get("contentLength", item.get("size", 0))) or 0)}


class KaggleReader:
    def __init__(self):
        try:
            import requests
        except ImportError:
            sys.path.insert(0, str(ROOT / ".venv" / "kaggle-api"))
            import requests
        self.requests = requests
        self.session = requests.Session()
        if os.environ.get("KAGGLE_API_TOKEN"):
            token = os.environ["KAGGLE_API_TOKEN"]
            self.session.headers["Authorization"] = "Bearer " + token
            self.secrets = (token,)
        else:
            try:
                credential = json.loads((ROOT / "api_key" / "kaggle_2.json").read_text(encoding="utf-8-sig"))
                username, key = credential["username"], credential["key"]
                if not username or not key:
                    raise ValueError
            except Exception:
                raise SafeKaggleError("Authorized Kaggle credential is unavailable or incomplete.") from None
            self.session.auth = (username, key)
            self.secrets = (key,)

    def request(self, endpoint: str, *, params: dict | None = None, stream: bool = False):
        try:
            response = self.session.get(API_ROOT + endpoint, params=params, stream=stream, timeout=(30, 120))
        except Exception:
            raise SafeKaggleError("Kaggle request failed before an HTTP response.") from None
        if response.status_code != 200:
            code = response.status_code
            response.close()
            raise SafeKaggleError(f"Kaggle request failed (HTTP {code}).")
        return response

    def json(self, endpoint: str, params: dict | None = None):
        with self.request(endpoint, params=params) as response:
            try:
                return response.json()
            except Exception:
                raise SafeKaggleError("Kaggle did not return the expected JSON response.") from None

    def kernel_args(self, kernel: str) -> dict:
        parts = kernel.split("/")
        if len(parts) != 2 or not all(re.fullmatch(r"[A-Za-z0-9_-]+", p) for p in parts):
            raise SafeKaggleError("Kernel must be an owner/slug pair.")
        return {"userName": parts[0], "kernelSlug": parts[1]}

    def status(self, kernel: str = KERNEL) -> dict:
        raw = self.json("/kernels/status", self.kernel_args(kernel))
        return {"kernel": kernel, "status": sanitize_text(str(raw.get("status", "unknown")), self.secrets)}

    def files(self, kernel: str = KERNEL, version: int | None = None) -> list[dict]:
        return self.output(kernel, version)["files"]

    def competition_files(self) -> list[dict]:
        args = {"pageSize": 100}
        result = []
        for _ in range(100):
            raw = self.json(f"/competitions/data/list/{COMPETITION}", args)
            result.extend(raw.get("files", []))
            token = raw.get("nextPageToken")
            if not token:
                return result
            args["pageToken"] = token
        raise SafeKaggleError("Competition file pagination exceeded the bounded limit.")

    def output(self, kernel: str = KERNEL, version: int = 20) -> dict:
        # Fail closed if latest output no longer corresponds to the requested
        # version. Signed output URLs stay inside this process only.
        args = self.kernel_args(kernel)
        before = self.json("/kernels/pull", args)["metadata"].get("currentVersionNumber")
        if version is not None and before != version:
            raise SafeKaggleError("Latest kernel version differs from the requested version; refusing unpinned output.")
        output = self.json("/kernels/output", args)
        if output.get("nextPageToken"):
            raise SafeKaggleError("Output inventory is paginated; refusing an incomplete listing.")
        after = self.json("/kernels/pull", args)["metadata"].get("currentVersionNumber")
        if before != after:
            raise SafeKaggleError("Kernel version advanced while listing outputs; refusing unpinned output.")
        return output

    def download_url(self, url: str, destination: Path, expected_bytes: int = 0) -> dict:
        # A fresh unauthenticated request ensures Kaggle authorization is never
        # attached to the signed storage URL.
        temporary = destination.with_suffix(destination.suffix + ".partial")
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest, count = hashlib.sha256(), 0
        try:
            with self.requests.get(url, stream=True, timeout=(30, 180)) as response:
                if response.status_code != 200:
                    raise SafeKaggleError(f"Artifact download failed (HTTP {response.status_code}).")
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        count += len(chunk)
                        if expected_bytes and count > expected_bytes:
                            raise SafeKaggleError("Artifact exceeded the registered file size.")
                        handle.write(chunk)
                        digest.update(chunk)
            if expected_bytes and count != expected_bytes:
                raise SafeKaggleError("Artifact byte count differs from the registered file size.")
            temporary.replace(destination)
        except SafeKaggleError:
            temporary.unlink(missing_ok=True)
            raise
        except Exception:
            temporary.unlink(missing_ok=True)
            raise SafeKaggleError("Artifact download failed; no partial artifact retained.") from None
        return {"file": destination.name, "bytes": count, "sha256": digest.hexdigest()}

    def download_v20(self, names: list[str], destination: Path) -> list[dict]:
        if not names or any(n not in V20_ALLOWLIST for n in names):
            raise SafeKaggleError("Only exact allowlisted v20 artifact filenames may be downloaded.")
        output = self.output()
        by_name = {f["fileName"]: f for f in output["files"]}
        reports = []
        for name in names:
            entry = by_name.get(V20_ROOT + name)
            if entry is None:
                raise SafeKaggleError(f"Required v20 artifact is absent: {name}.")
            reports.append(self.download_url(entry["url"], destination / name, file_summary(entry)["bytes"]))
        (destination / "download_manifest.json").write_text(json.dumps({"kernel": KERNEL, "version": 20, "artifacts": reports}, indent=2), encoding="utf-8")
        return reports

    def peek_csv(self, name: str, max_bytes: int = 1024 * 1024) -> dict:
        safe_name(name)
        if not name.lower().endswith(".csv"):
            raise SafeKaggleError("Schema peeking supports CSV files only.")
        from urllib.parse import quote
        with self.request(f"/competitions/data/download/{COMPETITION}/{quote(name, safe='/')}", stream=True) as response:
            prefix = bytearray()
            for chunk in response.iter_content(64 * 1024):
                prefix.extend(chunk[:max_bytes - len(prefix)])
                if len(prefix) >= max_bytes:
                    break
        data = decode_csv_prefix(bytes(prefix), max_bytes)
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace"))))
        if len(rows) < 2:
            raise SafeKaggleError("CSV prefix contains too few rows to establish its schema.")
        return {"file": name, "columns": rows[0], "sample_rows": rows[1:4], "downloaded_prefix_bytes": len(prefix)}


def decode_csv_prefix(data: bytes, max_bytes: int) -> bytes:
    """Decode the first file in an ordinary streamed ZIP without fetching all of it."""
    if not data.startswith(b"PK\x03\x04"):
        return data
    if len(data) < 30:
        raise SafeKaggleError("Incomplete ZIP header in bounded CSV prefix.")
    _, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", data[:30])
    if flags & 1:
        raise SafeKaggleError("Encrypted competition files are unsupported.")
    payload = data[30 + name_len + extra_len:]
    if method == 0:
        return payload[:max_bytes]
    if method == 8:
        return zlib.decompressobj(-15).decompress(payload, max_bytes)
    raise SafeKaggleError("Unsupported ZIP compression in CSV prefix.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "files", "competition-files", "schema", "download-v20", "log"))
    parser.add_argument("--kernel", default=KERNEL)
    parser.add_argument("--version", type=int, default=20)
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "v20_frozen")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    client = KaggleReader()
    if args.action == "status":
        result = client.status(args.kernel)
    elif args.action == "files":
        result = [file_summary(f) for f in client.files(args.kernel, args.version)]
    elif args.action == "competition-files":
        result = [file_summary(f) for f in client.competition_files()]
    elif args.action == "schema":
        result = [client.peek_csv(name) for name in args.file]
    elif args.action == "download-v20":
        result = client.download_v20(args.file, args.output)
    else:
        result = {"log": sanitize_text(str(client.output(args.kernel, args.version).get("log", "")), client.secrets)}
    rendered = json.dumps(result, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafeKaggleError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
    except Exception:
        print("ERROR: Unexpected Kaggle access failure; raw details withheld to protect credentials and signed URLs.", file=sys.stderr)
        raise SystemExit(2)
