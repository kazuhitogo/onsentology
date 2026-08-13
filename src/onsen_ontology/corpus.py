"""比較実験用のコーパス構築（Phase 7）。

「同じ一次情報をトリプルに整理する価値はあるのか」を測るには、**オントロジーと同じ
情報源**を生テキストのまま検索できる状態に置く必要がある。このモジュールは
``docs/`` が出典として記録している URL 群を取得し、Markdown に落とす。

コーパスの定義（案A）
    **``docs/*.md`` に出典として書かれている URL の全体**をコーパスとする。
    人が「これは入れる／外す」と選ばない。選ぶと「都合の悪い情報源を外した」という
    批判が立つし、実際に選別の誘惑が働く。URL の抽出は :func:`source_urls` が機械的に行う。

この定義だと**コーパスは汚染されたまま入る**。蔵王温泉観光協会と湯の花茶屋 新左衛門の湯の
公式ページには平成26年改訂前の効能表記（虚弱児童・慢性婦人病など）と通俗表現（美人の湯）が
載っている。それは藁人形ではなく、生テキストで検索するとはそういうことである。

全チャンクに**出典URLと取得日**を持たせるのは、RAG 側にも公平に出典を語らせるためである。
オントロジー側は ``onsen:sourceUrl`` を持っているので、片方だけ出典を言えない状態で
比較すると条件の差が出典の有無にすり替わる。

取得したファイルは git 管理外（``corpus/``）。第三者の著作物なので再配布はせず、
``uv run onsen corpus build`` で誰でも同じものを組み直せるようにしてある。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

#: コーパスの既定の置き場所。git 管理外。
DEFAULT_CORPUS_DIR = Path("corpus")

#: 出典URLを拾う対象。リポジトリが「調査した」と記録している文書だけを見る。
DEFAULT_DOCS_DIR = Path("docs")

#: 取得時の User-Agent。素性を明かしておく（研究目的・少量アクセス）。
USER_AGENT = "onsen-ontology-corpus-builder/0.1 (+https://github.com/kazuhitogo/onsentology)"

#: 連続アクセスの間隔（秒）。相手のサーバに負荷をかけない。
FETCH_INTERVAL_SEC = 1.0

FETCH_TIMEOUT_SEC = 60

#: URL の末尾に紛れ込む記号・全角文字を落とすためのパターン。
#: Markdown の表や文中に書かれた URL は「`」「）」「、」「。」などで終わることがある。
_URL_PATTERN = re.compile(r"https?://[^\s<>\"'`\)\]｜|、。]+")
_URL_TRAILING = re.compile(r"[.,;:！？!?]+$")


def source_urls(docs_dir: Path | str = DEFAULT_DOCS_DIR) -> list[str]:
    """``docs/*.md`` に出典として現れる URL を重複なく集める。

    人の選別を挟まないことが要件なので、フィルタは「http(s) であること」だけにしてある。
    """
    directory = Path(docs_dir)
    found: dict[str, None] = {}
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for raw in _URL_PATTERN.findall(text):
            url = _URL_TRAILING.sub("", raw)
            found.setdefault(url, None)
    return list(found)


# --------------------------------------------------------------------------
# HTML → テキスト
# --------------------------------------------------------------------------

#: 中身を捨てるタグ。
_SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}

#: 改行を入れるタグ。
_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "td", "th", "section", "article", "header", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol", "dl", "dt", "dd", "blockquote",
}


class _TextExtractor(HTMLParser):
    """HTML から本文テキストと ``<title>`` を取り出す。

    依存を増やしたくないので標準ライブラリだけで書いている。整形の忠実さより、
    **BM25 の索引に載る語が落ちないこと**を優先する（表の中の pH の値など）。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data.strip()
            return
        if self._skip_depth:
            return
        if data.strip():
            self.parts.append(data)

    def text(self) -> str:
        return normalize_whitespace("".join(self.parts))


def normalize_whitespace(text: str) -> str:
    """行内の空白を畳み、空行の連続を1つにする。"""
    lines = [re.sub(r"[ \t\u3000\xa0]+", " ", line).strip() for line in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def html_to_text(html: str) -> tuple[str, str]:
    """HTML を ``(本文, タイトル)`` にする。"""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text(), parser.title


def decode_bytes(raw: bytes, content_type: str = "") -> str:
    """文字コードを推定してデコードする。

    温泉施設の公式ページには Shift_JIS のものが現に残っている（箱根温泉旅館協同組合）。
    ヘッダの charset → HTML の meta → 候補を順に試す、の順で決める。
    """
    candidates: list[str] = []
    header_charset = re.search(r"charset=([\w\-]+)", content_type, re.IGNORECASE)
    if header_charset:
        candidates.append(header_charset.group(1))
    head = raw[:4096].decode("ascii", errors="ignore")
    meta_charset = re.search(r"charset=[\"']?([\w\-]+)", head, re.IGNORECASE)
    if meta_charset:
        candidates.append(meta_charset.group(1))
    candidates += ["utf-8", "cp932", "euc-jp"]

    for name in candidates:
        # iso-8859-1 は日本語ページでも既定値として返ってくるので信用しない
        if name.lower().replace("_", "-") in {"iso-8859-1", "latin-1", "us-ascii"}:
            continue
        try:
            return raw.decode(name)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def pdf_to_text(raw: bytes) -> str:
    """PDF をテキストにする（``pdftotext -layout``）。

    ``-layout`` を付けるのは、鉱泉分析法指針の第1-3表のような**表**を読める形で残すため。
    列が崩れると「酸性泉 水素イオン 1mg」の対応が失われる。
    """
    if shutil.which("pdftotext") is None:
        raise RuntimeError("pdftotext が無い（poppler-utils を入れる）")
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "in.pdf"
        pdf_path.write_bytes(raw)
        completed = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True,
            check=True,
        )
    return normalize_whitespace(completed.stdout.decode("utf-8", errors="replace"))


# --------------------------------------------------------------------------
# 取得
# --------------------------------------------------------------------------


@dataclass
class FetchResult:
    """1 URL の取得結果。失敗も理由つきで残す（欠損を欠損として持つ）。"""

    url: str
    status: str  # "ok" / "error"
    content_type: str = ""
    title: str = ""
    text: str = ""
    bytes: int = 0
    error: str = ""
    retrieved_at: str = ""


def fetch(url: str, *, timeout: int = FETCH_TIMEOUT_SEC) -> FetchResult:
    """1 URL を取得してテキスト化する。リダイレクトは追う。"""
    retrieved_at = datetime.now(UTC).strftime("%Y-%m-%d")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
            final_url = response.geturl()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        return FetchResult(url=url, status="error", error=str(exc), retrieved_at=retrieved_at)

    try:
        if "pdf" in content_type.lower() or final_url.lower().endswith(".pdf"):
            text, title = pdf_to_text(raw), ""
            content_type = content_type or "application/pdf"
        else:
            text, title = html_to_text(decode_bytes(raw, content_type))
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        return FetchResult(
            url=url,
            status="error",
            content_type=content_type,
            bytes=len(raw),
            error=f"本文の抽出に失敗: {exc}",
            retrieved_at=retrieved_at,
        )

    if not text.strip():
        return FetchResult(
            url=url,
            status="error",
            content_type=content_type,
            bytes=len(raw),
            error="本文が空（JavaScript で描画されるページの可能性）",
            retrieved_at=retrieved_at,
        )
    return FetchResult(
        url=url,
        status="ok",
        content_type=content_type,
        title=title,
        text=text,
        bytes=len(raw),
        retrieved_at=retrieved_at,
    )


def slugify(url: str) -> str:
    """URL からファイル名を作る。ホスト名とパスを残して人が読めるようにする。"""
    stripped = re.sub(r"^https?://", "", url).rstrip("/")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stripped).strip("-").lower()
    return slug[:120] or "document"


def write_document(result: FetchResult, directory: Path) -> Path:
    """取得結果を出典つき Markdown として書き出す。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slugify(result.url)}.md"
    front = [
        "---",
        f"source_url: {result.url}",
        f"retrieved_at: {result.retrieved_at}",
        f"content_type: {result.content_type}",
        f"title: {result.title}",
        "---",
        "",
    ]
    path.write_text("\n".join(front) + result.text + "\n", encoding="utf-8")
    return path


@dataclass
class BuildReport:
    """コーパス構築の結果。"""

    written: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    total_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "取得成功": len(self.written),
            "取得失敗": len(self.failed),
            "総文字数": self.total_chars,
            "失敗した出典": self.failed,
        }


def build_corpus(
    *,
    docs_dir: Path | str = DEFAULT_DOCS_DIR,
    out_dir: Path | str = DEFAULT_CORPUS_DIR,
    urls: list[str] | None = None,
    interval: float = FETCH_INTERVAL_SEC,
    progress: bool = False,
    fetcher: Any = fetch,
) -> BuildReport:
    """出典URLを順に取得して ``out_dir`` に Markdown を書く。

    失敗は ``manifest.json`` に理由つきで残す。取れなかったものを黙って落とすと
    「コーパスに何が入っているか」が分からなくなる。
    """
    directory = Path(out_dir)
    targets = urls if urls is not None else source_urls(docs_dir)
    report = BuildReport()
    manifest: list[dict[str, Any]] = []

    for index, url in enumerate(targets, start=1):
        if progress:
            print(f"[{index}/{len(targets)}] {url}", flush=True)
        result = fetcher(url)
        entry: dict[str, Any] = {
            "source_url": url,
            "retrieved_at": result.retrieved_at,
            "status": result.status,
            "content_type": result.content_type,
            "chars": len(result.text),
        }
        if result.status == "ok":
            path = write_document(result, directory)
            entry["file"] = path.name
            report.written.append(path.name)
            report.total_chars += len(result.text)
        else:
            entry["error"] = result.error
            report.failed.append({"source_url": url, "error": result.error})
        manifest.append(entry)
        if interval and index < len(targets):
            time.sleep(interval)

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "docs_dir": str(docs_dir),
                "documents": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


__all__ = [
    "DEFAULT_CORPUS_DIR",
    "DEFAULT_DOCS_DIR",
    "BuildReport",
    "FetchResult",
    "build_corpus",
    "decode_bytes",
    "fetch",
    "html_to_text",
    "normalize_whitespace",
    "pdf_to_text",
    "slugify",
    "source_urls",
    "write_document",
]
