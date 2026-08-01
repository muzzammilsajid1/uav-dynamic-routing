"""Validate LaTeX inputs, citations, labels, and references without a TeX engine."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=PROJECT_ROOT / "paper_latex_v2",
    )
    args = parser.parse_args()
    main_path = args.paper_dir / "main.tex"
    pending = [main_path]
    sources: dict[Path, str] = {}

    while pending:
        path = pending.pop()
        if path in sources:
            continue
        if not path.exists():
            raise FileNotFoundError(f"Missing LaTeX input: {path}")
        text = path.read_text(encoding="utf-8")
        sources[path] = text
        for input_name in re.findall(r"\\input\{([^}]+)\}", text):
            input_path = args.paper_dir / input_name
            if input_path.suffix != ".tex":
                input_path = input_path.with_suffix(".tex")
            pending.append(input_path)

    combined = "\n".join(sources.values())
    bibliography = (args.paper_dir / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(
        re.findall(
            r"@\w+\s*\{\s*([^,\s]+)",
            bibliography,
            flags=re.IGNORECASE,
        )
    )
    citation_keys: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", combined):
        citation_keys.update(key.strip() for key in group.split(","))
    missing_citations = sorted(citation_keys - bib_keys)
    if missing_citations:
        raise RuntimeError(f"Missing bibliography keys: {missing_citations}")

    labels = re.findall(r"\\label\{([^}]+)\}", combined)
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate LaTeX labels: {duplicates}")
    references = set(
        re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", combined)
    )
    missing_labels = sorted(references - set(labels))
    if missing_labels:
        raise RuntimeError(f"Missing referenced labels: {missing_labels}")

    print(
        f"Validated {len(sources)} LaTeX files, {len(citation_keys)} citations, "
        f"{len(labels)} labels, and {len(references)} references"
    )


if __name__ == "__main__":
    main()
