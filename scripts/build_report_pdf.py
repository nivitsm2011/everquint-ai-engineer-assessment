"""Export reports/REPORT.md to reports/REPORT.pdf (Markdown -> HTML -> headless Edge/Chrome print).

Mermaid blocks (which a PDF cannot render) are replaced by the pre-drawn architecture image.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]
CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1d2733; }
h1 { font-size: 21pt; border-bottom: 2px solid #44546a; padding-bottom: 4px; }
h2 { font-size: 15pt; margin-top: 22px; color: #1f3a5f; border-bottom: 1px solid #c9d3df; }
h3 { font-size: 12pt; margin-top: 16px; color: #1f3a5f; }
table { border-collapse: collapse; margin: 8px 0 12px; font-size: 9.2pt; width: 100%; }
th, td { border: 1px solid #b7c2cf; padding: 3px 6px; vertical-align: top; }
th { background: #e8eef5; text-align: left; }
code { background: #f0f3f6; padding: 0 3px; font-size: 9pt; }
pre { background: #f0f3f6; padding: 8px; overflow-x: auto; font-size: 8.5pt; }
img { max-width: 100%; }
tr, img { page-break-inside: avoid; }
"""


def main() -> None:
    """Build the PDF (default: REPORT.md; pass another file name in reports/ to export it)."""
    source = REPORTS / (sys.argv[1] if len(sys.argv) > 1 else "REPORT.md")
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"```mermaid.*?```", "![Architecture](figures/architecture.png)", text, flags=re.S)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    head = f"<meta charset='utf-8'><style>{CSS}</style>"
    html = f"<!doctype html><html><head>{head}</head><body>{body}</body></html>"
    html_path = REPORTS / "_report_build.html"
    html_path.write_text(html, encoding="utf-8")
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        sys.exit("No Edge or Chrome found for PDF export.")
    pdf = source.with_suffix(".pdf")
    subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}",
            html_path.as_uri(),
        ],
        check=True,
        timeout=180,
    )
    html_path.unlink()
    sys.stdout.write(f"wrote {pdf} ({pdf.stat().st_size // 1024} KB)\n")


if __name__ == "__main__":
    main()
