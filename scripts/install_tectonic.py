"""Install the checksum-pinned Tectonic compiler into the workspace."""
from __future__ import annotations

import hashlib
import io
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALL_ROOT = PROJECT_ROOT / "tools" / "tectonic"
VERSION = "0.16.9"
RELEASE_ROOT = (
    "https://github.com/tectonic-typesetting/tectonic/releases/download/"
    "tectonic%400.16.9"
)

ASSETS = {
    ("Windows", "x86_64"): (
        "tectonic-0.16.9-x86_64-pc-windows-msvc.zip",
        "131a24604785a9600989a3d91225f597df52ac06f00aeffe86fd529f99ee5cdd",
        "tectonic.exe",
    ),
    ("Linux", "x86_64"): (
        "tectonic-0.16.9-x86_64-unknown-linux-gnu.tar.gz",
        "f3c825128095dc3399ea11c08c18035b33050a216930c295c79e8eb11bd21de4",
        "tectonic",
    ),
    ("Darwin", "x86_64"): (
        "tectonic-0.16.9-x86_64-apple-darwin.tar.gz",
        "79d8839fa3594bfea9b2bf2ac0a0455bcc4d0de956a5e5c403107e9a72f79e86",
        "tectonic",
    ),
    ("Darwin", "aarch64"): (
        "tectonic-0.16.9-aarch64-apple-darwin.tar.gz",
        "edb67c61aba768289f6da441c9e6f523cfaff4f8b2a5708523ef29c543f8e88e",
        "tectonic",
    ),
}


def _normalized_machine(machine: str) -> str:
    lowered = machine.lower()
    if lowered in {"amd64", "x86_64"}:
        return "x86_64"
    if lowered in {"arm64", "aarch64"}:
        return "aarch64"
    return lowered


def _select_asset(
    system: str | None = None,
    machine: str | None = None,
) -> tuple[str, str, str]:
    key = (
        system or platform.system(),
        _normalized_machine(machine or platform.machine()),
    )
    if key not in ASSETS:
        raise RuntimeError(
            f"No pinned Tectonic {VERSION} binary is configured for {key}."
        )
    return ASSETS[key]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _extract(payload: bytes, asset_name: str, executable_name: str) -> Path:
    INSTALL_ROOT.mkdir(parents=True, exist_ok=True)
    destination = INSTALL_ROOT / executable_name
    if asset_name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            member = next(
                name
                for name in archive.namelist()
                if Path(name).name == executable_name
            )
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    else:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            member = next(
                item
                for item in archive.getmembers()
                if Path(item.name).name == executable_name and item.isfile()
            )
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not extract {executable_name}")
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    if os.name != "nt":
        destination.chmod(0o755)
    return destination


def main() -> None:
    asset_name, expected_digest, executable_name = _select_asset()
    executable = INSTALL_ROOT / executable_name
    if executable.exists():
        result = subprocess.run(
            [str(executable), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        version_text = result.stdout.strip() or result.stderr.strip()
        if VERSION not in version_text:
            raise RuntimeError(
                f"Unexpected existing Tectonic version: {version_text}"
            )
        print(f"Using existing {version_text} at {executable}")
        return

    url = f"{RELEASE_ROOT}/{asset_name}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "uav-dynamic-routing-artifact-builder"},
    )
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    actual_digest = _sha256(payload)
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"Tectonic checksum mismatch: {actual_digest} != {expected_digest}"
        )
    executable = _extract(payload, asset_name, executable_name)
    result = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip() or result.stderr.strip())
    print(f"Installed verified compiler at {executable}")


if __name__ == "__main__":
    main()
