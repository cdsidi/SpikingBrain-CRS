"""评估报告生成器。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class ReportGenerator:
    """生成评估 JSON/Markdown/日志产物。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, filename: str, data: Dict) -> Path:
        """保存 JSON 汇总。"""
        p = self.output_dir / filename
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def save_markdown(self, filename: str, title: str, sections: Dict[str, Dict | str | float | int]) -> Path:
        """保存 Markdown 报告。"""
        lines = [f"# {title}", ""]
        for k, v in sections.items():
            lines.append(f"## {k}")
            if isinstance(v, dict):
                lines.append("```json")
                lines.append(json.dumps(v, ensure_ascii=False, indent=2))
                lines.append("```")
            else:
                lines.append(str(v))
            lines.append("")
        p = self.output_dir / filename
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def append_log(self, filename: str, message: str) -> Path:
        """追加日志行。"""
        p = self.output_dir / filename
        with p.open("a", encoding="utf-8") as f:
            f.write(message.rstrip("\n") + "\n")
        return p

