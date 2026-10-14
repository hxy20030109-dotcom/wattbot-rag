"""Build a local checklist for human downloads; makes no network requests."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data/competition/metadata.csv"
PDF_DIR = ROOT / "data/pdfs"
OUT_DIR = ROOT / "data/processed"


def cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def main() -> None:
    with METADATA.open(encoding="utf-8-sig", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    rows = []
    for meta in metadata:
        doc_id, url = meta["id"], meta["url"].strip()
        if Path(doc_id).name != doc_id or any(char in doc_id for char in '/\\:'):
            raise ValueError(f"Invalid document ID: {doc_id!r}")
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unexpected source URL for {doc_id}")
        pdf = PDF_DIR / f"{doc_id}.pdf"
        cached = False
        if pdf.is_file():
            with pdf.open("rb") as handle:
                cached = handle.read(5) == b"%PDF-"
        rows.append({
            "status": "cached" if cached else "pending",
            "doc_id": doc_id,
            "save_as": f"{doc_id}.pdf",
            "title": meta["title"],
            "url": url,
            "url_filename_hint": unquote(parsed.path.rsplit("/", 1)[-1]),
        })
    pending = [row for row in rows if row["status"] == "pending"]
    cached = [row for row in rows if row["status"] == "cached"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "download_checklist.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# WattBot PDF 人工下载清单", "",
        f"总计 {len(rows)} 篇；已缓存 {len(cached)} 篇；待下载 {len(pending)} 篇。", "",
        "这份清单只生成本地链接，不会自动访问或下载文档。", "",
        "1. 按顺序逐篇打开原始链接，保持 URL 中的版本号（例如 v2）。",
        "2. 在浏览器中下载 PDF。可以保留浏览器默认文件名，统一存到 Downloads；下载完一批后由助手核对、复制归档。",
        "3. 如自行保存到 data/pdfs，请使用表格中的归档文件名。不要覆盖已有文件。",
        "4. 链接失效、要求付费或出现访问限制时，记录文档 ID 并跳过，之后核查；不要自行换成最新版本。",
        "5. 这里的缓存状态来自生成时的本地文件；下载后重新生成清单以更新。", "",
        f"归档目录：`{PDF_DIR}`", "",
        f"metadata.csv SHA-256：`{hashlib.sha256(METADATA.read_bytes()).hexdigest()}`", "",
        "## 待下载", "",
        "| 序号 | 文档 ID / 归档文件名 | 标题 | 固定来源链接 |",
        "| ---: | --- | --- | --- |",
    ]
    for index, row in enumerate(pending, start=1):
        lines.append(f"| {index} | `{row['save_as']}` | {cell(row['title'])} | [打开原始链接](<{row['url']}>) |")
    lines.extend(["", "## 已有缓存", ""])
    lines.extend(f"- `{row['save_as']}` — {cell(row['title'])}" for row in cached)
    (OUT_DIR / "download_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"total={len(rows)} cached={len(cached)} pending={len(pending)}")
    print(OUT_DIR / "download_checklist.md")


if __name__ == "__main__":
    main()
