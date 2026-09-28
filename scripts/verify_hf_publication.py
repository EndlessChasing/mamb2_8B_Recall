#!/usr/bin/env python3
"""Verify a public Hugging Face publication against its local checksum manifest.

This checks distribution integrity, not model quality. Every file is downloaded
anonymously at a pinned commit and hashed, including checksum_manifest.json.
Uses only Python's standard library, the hf CLI, and curl.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from urllib.parse import quote


MANIFEST = "checksum_manifest.json"
FILE_LIMIT = 10_000_000


def run(command, timeout=180):
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=timeout, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-1500:]
        raise ValueError(f"{Path(command[0]).name} failed ({result.returncode}): {detail}")
    return result.stdout


def hf_json(hf, *args):
    return json.loads(run([hf, *args, "--format", "json"]))


def identities(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_id": hashlib.sha1(
                f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()}


def validate_name(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"Invalid manifest path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Unsafe or noncanonical manifest path: {name!r}")
    return path


def load_staging(staging):
    """Hash every entry and require the manifest to describe the exact package."""
    if not staging.is_dir():
        raise ValueError(f"Staging directory does not exist: {staging}")
    manifest_path = staging / MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"Missing regular {MANIFEST}")
    if manifest_path.stat().st_size >= FILE_LIMIT:
        raise ValueError("Manifest exceeds the supported size limit")
    raw_manifest = manifest_path.read_bytes()
    document = json.loads(raw_manifest)
    rows = document.get("files") if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain a nonempty files array")
    entries = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Manifest entries must be objects")
        name = row.get("path")
        validate_name(name)
        size, digest = row.get("bytes"), row.get("sha256")
        if name == MANIFEST or name in entries:
            raise ValueError(f"Duplicate or self-referential manifest entry: {name}")
        if type(size) is not int or not 0 <= size < FILE_LIMIT:
            raise ValueError(f"File must be smaller than 10 MB: {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"Invalid SHA-256: {name}")
        local = staging / name
        if not local.is_file() or local.is_symlink():
            raise ValueError(f"Missing regular file: {name}")
        if not local.resolve().is_relative_to(staging.resolve()):
            raise ValueError(f"File escapes staging: {name}")
        if local.stat().st_size != size:
            raise ValueError(f"Local size mismatch: {name}")
        actual = identities(local.read_bytes())
        if actual["bytes"] != size or actual["sha256"] != digest:
            raise ValueError(f"Local bytes changed: {name}")
        entries[name] = {"path": name, **actual, "local_manifest_verified": True}
    entries[MANIFEST] = {"path": MANIFEST, **identities(raw_manifest),
                         "local_manifest_verified": "manifest_itself_hashed"}
    inventory = set()
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Staging contains a symlink: {path.relative_to(staging)}")
        if path.is_file():
            inventory.add(path.relative_to(staging).as_posix())
        elif not path.is_dir():
            raise ValueError(f"Staging contains a special file: {path}")
    if inventory != set(entries):
        raise ValueError(f"Local inventory differs: {sorted(inventory ^ set(entries))}")
    return dict(sorted(entries.items()))


def verify_remote(rows, expected):
    if not isinstance(rows, list):
        raise ValueError("Unexpected hf models list response")
    remote = {}
    for row in rows:
        if "blob_id" not in row:
            if "tree_id" in row:
                continue
            raise ValueError(f"Unexpected repository entry: {row.get('path')}")
        name = row["path"]
        if name in remote:
            raise ValueError(f"Duplicate remote file: {name}")
        remote[name] = row
    if set(remote) != set(expected):
        raise ValueError(f"Remote inventory differs: {sorted(set(remote) ^ set(expected))}")
    verified = []
    for name, entry in expected.items():
        row = remote[name]
        if row.get("size") != entry["bytes"]:
            raise ValueError(f"Remote size mismatch: {name}")
        lfs = row.get("lfs")
        if lfs:
            if lfs.get("size") != entry["bytes"] or lfs.get("sha256") != entry["sha256"]:
                raise ValueError(f"Remote LFS digest mismatch: {name}")
            method = "server_lfs_sha256"
        else:
            if row["blob_id"] != entry["git_blob_id"]:
                raise ValueError(f"Remote Git blob mismatch: {name}")
            method = "server_git_blob_with_local_sha256"
        verified.append({**entry, "server_verification": method,
                         "server_blob_id": row["blob_id"],
                         "server_size": row["size"]})
    return verified


def anonymous_check(repo, revision, entry):
    url = (f"https://huggingface.co/{repo}/resolve/{revision}/"
           + quote(entry["path"], safe="/"))
    with tempfile.TemporaryDirectory(prefix="mamba-recall-hf-verify-") as tmp:
        target = Path(tmp) / "body"
        # --disable is first: ignore curlrc credentials/settings. No token,
        # cookies, netrc, or Authorization header is supplied to this request.
        status = run([
            "curl", "--disable", "--fail", "--location", "--silent", "--show-error",
            "--proto", "=https", "--proto-redir", "=https", "--max-redirs", "5",
            "--connect-timeout", "20", "--max-time", "60", "--retry", "2",
            "--retry-max-time", "120", "--max-filesize", str(32 << 20),
            "--output", str(target), "--write-out", "%{http_code}", "--url", url,
        ])
        if status.strip() != "200":
            raise ValueError(f"Anonymous GET was not HTTP 200: {entry['path']}")
        actual = identities(target.read_bytes())
    if actual["bytes"] != entry["bytes"] or actual["sha256"] != entry["sha256"]:
        raise ValueError(f"Anonymous downloaded bytes differ: {entry['path']}")
    return {"path": entry["path"], "url": url, "http_status": 200,
            "bytes": actual["bytes"], "sha256": actual["sha256"],
            "full_anonymous_body_downloaded": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Hugging Face owner/model")
    parser.add_argument("--staging", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New JSON report path outside staging")
    parser.add_argument("--hf", default="hf", help="Path to hf executable")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        raise ValueError("--repo must be an owner/model repository ID")
    if args.output:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError(f"Refusing to overwrite report: {args.output}")
        if args.output.resolve().is_relative_to(args.staging.resolve()):
            raise ValueError("--output must be outside staging")
        if not args.output.parent.is_dir():
            raise ValueError(f"Report parent directory does not exist: {args.output.parent}")
    expected = load_staging(args.staging)
    info = hf_json(args.hf, "models", "info", args.repo, "--expand", "sha,private")
    revision = info.get("sha")
    if info.get("private") is not False or not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Expected a public repository with a pinned Git commit")
    rows = hf_json(args.hf, "models", "list", args.repo, "--recursive", "--revision", revision)
    verified = verify_remote(rows, expected)
    print(f"Pinned {revision}: {len(verified)} server identities match; downloading anonymously.", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=4) as pool:
        public = list(pool.map(lambda entry: anonymous_check(args.repo, revision, entry), verified))
    # Catch local edits made during a long network check.
    if load_staging(args.staging) != expected:
        raise ValueError("Staging changed during verification")
    report = {"complete": True, "verified_at": datetime.now(timezone.utc).isoformat(),
              "repo": args.repo, "url": f"https://huggingface.co/{args.repo}",
              "revision": revision,
              "pinned_url": f"https://huggingface.co/{args.repo}/tree/{revision}",
              "public": True, "exact_local_inventory": True, "exact_remote_inventory": True,
              "manifest_sha256": expected[MANIFEST]["sha256"],
              "files_verified": len(verified),
              "total_repo_file_bytes": sum(x["bytes"] for x in verified),
              "all_files_downloaded_anonymously": True,
              "files": verified, "anonymous_download_checks": public,
              "scope": "Public distribution integrity at a pinned commit; no new GPU quality evaluation."}
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(serialized)
        print(json.dumps({key: value for key, value in report.items()
                          if key not in ("files", "anonymous_download_checks")}, indent=2))
    else:
        print(serialized, end="")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
