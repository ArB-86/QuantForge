from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ReportGenerator:
    def __init__(
        self,
        run_dir: str | Path,
        config: dict[str, Any],
        metrics: dict[str, Any],
        artifacts: dict[str, Any],
    ):
        self.run_dir = Path(run_dir)
        self.config = config
        self.metrics = metrics
        self.artifacts = artifacts

    def generate(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

        md = self._markdown()
        html = self._html(md)

        (self.run_dir / "report.md").write_text(
            md,
            encoding="utf-8",
        )

        (self.run_dir / "report.html").write_text(
            html,
            encoding="utf-8",
        )

        summary = {
            "generated_at": datetime.utcnow().isoformat(),
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }

        with open(
            self.run_dir / "summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=4)

    def _markdown(self) -> str:
        lines = []

        lines.append("# QuantForge Research Report\n")

        lines.append("## Experiment\n")

        for k, v in sorted(self.config.items()):
            lines.append(f"- **{k}** : {v}")

        lines.append("\n## Metrics\n")

        for k, v in sorted(self.metrics.items()):
            lines.append(f"- **{k}** : {v}")

        lines.append("\n## Artifacts\n")

        for k, v in sorted(self.artifacts.items()):
            lines.append(f"- **{k}** : {v}")

        return "\n".join(lines)

    def _html(self, markdown: str) -> str:
        body = (
            markdown.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>\n")
        )

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>QuantForge Report</title>

<style>
body{{
font-family:Arial,Helvetica,sans-serif;
margin:40px;
background:#ffffff;
color:#222;
}}

h1{{
border-bottom:2px solid #ddd;
padding-bottom:10px;
}}

code{{
background:#f3f3f3;
padding:2px 4px;
}}

</style>

</head>

<body>

{body}

</body>

</html>
"""
