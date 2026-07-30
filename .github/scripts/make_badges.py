from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


SITE_BADGES = Path("site/badges")
SITE_BADGES.mkdir(parents=True, exist_ok=True)


def color_for_percent(value: float) -> str:
    if value >= 95:
        return "brightgreen"
    if value >= 90:
        return "green"
    if value >= 80:
        return "yellowgreen"
    if value >= 70:
        return "yellow"
    if value >= 60:
        return "orange"
    return "red"


def color_for_pylint(value: float) -> str:
    if value >= 9.5:
        return "brightgreen"
    if value >= 9.0:
        return "green"
    if value >= 8.0:
        return "yellowgreen"
    if value >= 7.0:
        return "yellow"
    if value >= 6.0:
        return "orange"
    return "red"


def write_shields_badge(path: Path, label: str, message: str, color: str) -> None:
    payload = {
        "schemaVersion": 1,
        "label": label,
        "message": message,
        "color": color,
    }

    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_pylint_badge() -> None:
    report_path = Path("pylint-report.txt")

    if not report_path.exists():
        write_shields_badge(
            SITE_BADGES / "pylint.json",
            label="pylint",
            message="missing",
            color="lightgrey",
        )
        return

    text = report_path.read_text(encoding="utf-8", errors="replace")

    match = re.search(r"rated at\s+(-?\d+(?:\.\d+)?)/10", text)

    if not match:
        write_shields_badge(
            SITE_BADGES / "pylint.json",
            label="pylint",
            message="unknown",
            color="lightgrey",
        )
        return

    score = float(match.group(1))

    write_shields_badge(
        SITE_BADGES / "pylint.json",
        label="pylint",
        message=f"{score:.2f}/10",
        color=color_for_pylint(score),
    )


def make_coverage_badge() -> None:
    coverage_path = Path("coverage.xml")

    if not coverage_path.exists():
        write_shields_badge(
            SITE_BADGES / "coverage.json",
            label="coverage",
            message="missing",
            color="lightgrey",
        )
        return

    root = ET.parse(coverage_path).getroot()

    line_rate = root.attrib.get("line-rate")

    if line_rate is None:
        write_shields_badge(
            SITE_BADGES / "coverage.json",
            label="coverage",
            message="unknown",
            color="lightgrey",
        )
        return

    coverage = float(line_rate) * 100.0

    write_shields_badge(
        SITE_BADGES / "coverage.json",
        label="coverage",
        message=f"{coverage:.1f}%",
        color=color_for_percent(coverage),
    )


def main() -> None:
    make_pylint_badge()
    make_coverage_badge()


if __name__ == "__main__":
    main()
