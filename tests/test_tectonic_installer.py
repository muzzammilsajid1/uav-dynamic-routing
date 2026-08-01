from __future__ import annotations

import pytest

from scripts.install_tectonic import _normalized_machine, _select_asset


def test_windows_asset_is_checksum_pinned() -> None:
    name, digest, executable = _select_asset("Windows", "AMD64")
    assert name == "tectonic-0.16.9-x86_64-pc-windows-msvc.zip"
    assert len(digest) == 64
    assert executable == "tectonic.exe"


def test_machine_aliases_are_normalized() -> None:
    assert _normalized_machine("arm64") == "aarch64"
    assert _normalized_machine("AMD64") == "x86_64"


def test_unsupported_platform_fails_explicitly() -> None:
    with pytest.raises(RuntimeError):
        _select_asset("Plan9", "mips")
