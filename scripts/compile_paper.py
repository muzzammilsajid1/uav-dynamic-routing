"""Compile and verify the current research manuscript with pinned Tectonic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = PROJECT_ROOT / "paper_latex_v2"
GENERATED_ROOT = PAPER_ROOT / "generated"
BUILD_ROOT = PROJECT_ROOT / "tmp" / "pdfs" / "paper-build"
FINAL_ROOT = PROJECT_ROOT / "output" / "pdf"
INTEGRITY_REPORT = PROJECT_ROOT / "evaluation" / "results" / "integrity_report.json"
LOCAL_TECTONIC_CANDIDATES = (
    PROJECT_ROOT / "tools" / "tectonic" / "tectonic.exe",
    PROJECT_ROOT / "tools" / "tectonic" / "tectonic",
)
FINAL_PDF_NAME = "uav_dynamic_routing_research_paper.pdf"

REQUIRED_GENERATED_FRAGMENTS = (
    "abstract_results.tex",
    "results_narrative.tex",
    "conclusion_results.tex",
    "generalization_table.tex",
    "scaling_table.tex",
    "realism_table.tex",
    "ablation_table.tex",
    "adaptability_table.tex",
)
PLACEHOLDER_MARKERS = (
    "Generated after final experiments",
    "generated after the complete experiment suite finishes",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_integrity_report() -> dict:
    if not INTEGRITY_REPORT.exists():
        raise RuntimeError(
            "Final artifact integrity report is missing. Run the complete research "
            "pipeline before compiling a release PDF."
        )
    report = json.loads(INTEGRITY_REPORT.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError(
            f"Artifact integrity status is {report.get('status')!r}, not 'passed'."
        )
    return report


def _assert_final_fragments() -> None:
    missing = [
        name for name in REQUIRED_GENERATED_FRAGMENTS
        if not (GENERATED_ROOT / name).exists()
    ]
    if missing:
        raise RuntimeError(f"Missing generated manuscript fragments: {missing}")

    incomplete: list[str] = []
    for name in REQUIRED_GENERATED_FRAGMENTS:
        content = (GENERATED_ROOT / name).read_text(encoding="utf-8")
        if not content.strip() or any(marker in content for marker in PLACEHOLDER_MARKERS):
            incomplete.append(name)
    if incomplete:
        raise RuntimeError(
            "Generated manuscript fragments still contain placeholders: "
            f"{incomplete}"
        )


def _find_tectonic() -> Path:
    for candidate in LOCAL_TECTONIC_CANDIDATES:
        if candidate.exists():
            return candidate
    executable = shutil.which("tectonic")
    if executable:
        return Path(executable)
    raise RuntimeError(
        "Tectonic was not found. Install the pinned workspace-local compiler at "
        "tools/tectonic/ by running scripts/install_tectonic.py."
    )


def _verify_pdf(pdf_path: Path, require_final: bool) -> tuple[int, str]:
    payload = pdf_path.read_bytes()
    if not payload.startswith(b"%PDF-") or len(payload) < 10_000:
        raise RuntimeError(f"Compiler output is not a plausible PDF: {pdf_path}")

    log_path = BUILD_ROOT / "main.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    page_matches = re.findall(
        r"Output written on .+? \((\d+) pages?,",
        log_text,
        flags=re.IGNORECASE,
    )
    if not page_matches:
        raise RuntimeError("Could not verify the compiled PDF page count.")
    page_count = int(page_matches[-1])
    if page_count < 1:
        raise RuntimeError("Compiled PDF contains no pages.")

    fatal_patterns = [r"LaTeX Error:"]
    if require_final:
        fatal_patterns.extend(
            [
                r"There were undefined references",
                r"Citation .* undefined",
                r"Reference .* undefined",
            ]
        )
    failures = [
        pattern for pattern in fatal_patterns
        if re.search(pattern, log_text, flags=re.IGNORECASE)
    ]
    if failures:
        raise RuntimeError(f"PDF log failed validation checks: {failures}")

    if require_final:
        source_text = "\n".join(
            (GENERATED_ROOT / name).read_text(encoding="utf-8")
            for name in REQUIRED_GENERATED_FRAGMENTS
        )
        if any(marker in source_text for marker in PLACEHOLDER_MARKERS):
            raise RuntimeError("Release PDF source still contains placeholder prose.")

    return page_count, _sha256(pdf_path)


def _write_manifest(
    *,
    output_path: Path,
    engine: Path,
    engine_version: str,
    page_count: int,
    pdf_sha256: str,
    integrity_report: dict | None,
) -> None:
    manifest = {
        "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(PAPER_ROOT / "main.tex"),
        "output": str(output_path),
        "engine": str(engine),
        "engine_version": engine_version,
        "page_count": page_count,
        "pdf_sha256": pdf_sha256,
        "integrity_report_sha256": (
            _sha256(INTEGRITY_REPORT) if integrity_report is not None else None
        ),
        "artifact_integrity_status": (
            integrity_report.get("status") if integrity_report is not None else None
        ),
    }
    (FINAL_ROOT / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Compile a draft without requiring final experiment artifacts.",
    )
    args = parser.parse_args()
    require_final = not args.allow_incomplete

    integrity_report = None
    if require_final:
        integrity_report = _load_integrity_report()
        _assert_final_fragments()

    engine = _find_tectonic()
    version_result = subprocess.run(
        [str(engine), "--version"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    engine_version = version_result.stdout.strip() or version_result.stderr.strip()

    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    cache_root = PROJECT_ROOT / "tools" / "tectonic" / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["TECTONIC_CACHE_DIR"] = str(cache_root)

    command = [
        str(engine),
        "main.tex",
        "--outdir",
        str(BUILD_ROOT),
        "--keep-logs",
        "--keep-intermediates",
        "--print",
        "--color",
        "never",
    ]
    print(f"> {' '.join(command)}", flush=True)
    subprocess.run(
        command,
        cwd=PAPER_ROOT,
        env=environment,
        check=True,
    )

    built_pdf = BUILD_ROOT / "main.pdf"
    page_count, pdf_sha256 = _verify_pdf(built_pdf, require_final)
    if require_final:
        output_path = FINAL_ROOT / FINAL_PDF_NAME
        shutil.copy2(built_pdf, output_path)
        shutil.copy2(built_pdf, PAPER_ROOT / "main.pdf")
        _write_manifest(
            output_path=output_path,
            engine=engine,
            engine_version=engine_version,
            page_count=page_count,
            pdf_sha256=pdf_sha256,
            integrity_report=integrity_report,
        )
    else:
        output_path = BUILD_ROOT / "main.pdf"

    print(
        f"Verified {page_count}-page PDF: {output_path} "
        f"(sha256={pdf_sha256})"
    )


if __name__ == "__main__":
    main()
