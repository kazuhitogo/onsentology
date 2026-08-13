"""コーパス構築のテスト（Phase 7）。

ネットワークには触らない。取得は :func:`onsen_ontology.corpus.fetch` を差し替えて検証する。
見るべきことは3つ。

1. **コーパスの定義が機械的である**こと。``docs/*.md`` の URL を人が選別せずに拾う
2. **全チャンクに出典URLと取得日が付く**こと。RAG 側にも公平に出典を語らせるための要件
3. **取れなかったものが理由つきで残る**こと。黙って落とすとコーパスの中身が分からなくなる
"""

from __future__ import annotations

import json
from pathlib import Path

from onsen_ontology import corpus


def test_source_urls_are_collected_from_docs() -> None:
    """出典URLは docs から機械的に拾う（人が選別しない）。"""
    urls = corpus.source_urls("docs")
    assert len(urls) > 40
    assert all(url.startswith("http") for url in urls)
    # 末尾に記号や全角文字が紛れ込んでいない
    assert all(not url.endswith(("`", "。", "、", ")", "」")) for url in urls)
    # 法令側と施設側の両方が入っている
    assert any("env.go.jp" in url for url in urls)
    assert any("onsen-kusatsu.com" in url for url in urls)
    # 重複しない
    assert len(urls) == len(set(urls))


def test_source_urls_ignores_prose_punctuation(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(
        "出典は https://example.org/a.html （2026-08-13取得）。\n"
        "| S1 | https://example.org/b.pdf | 説明 |\n"
        "`https://example.org/c/`（リンク切れ）\n",
        encoding="utf-8",
    )
    assert corpus.source_urls(docs) == [
        "https://example.org/a.html",
        "https://example.org/b.pdf",
        "https://example.org/c/",
    ]


def test_html_to_text_drops_script_and_keeps_numbers() -> None:
    html = """
    <html><head><title>御座之湯 泉質</title><style>.a{color:red}</style></head>
    <body><script>var x = 1;</script>
    <h2>泉質・効能</h2><table><tr><td>湯畑源泉</td><td>pH2.08</td></tr></table>
    </body></html>
    """
    text, title = corpus.html_to_text(html)
    assert title == "御座之湯 泉質"
    assert "pH2.08" in text
    assert "湯畑源泉" in text
    # スクリプト・スタイルの中身は索引に載せない
    assert "var x" not in text
    assert "color:red" not in text


def test_decode_bytes_handles_shift_jis() -> None:
    """箱根温泉旅館協同組合のページは Shift_JIS で、ヘッダは iso-8859-1 を返す。"""
    raw = "温泉法".encode("cp932")
    assert corpus.decode_bytes(raw, "text/html; charset=iso-8859-1") == "温泉法"
    assert corpus.decode_bytes("温泉法".encode(), "text/html; charset=utf-8") == "温泉法"


def test_slugify_keeps_host_and_path() -> None:
    slug = corpus.slugify("http://onsen-kusatsu.com/gozanoyu/hot-spring/")
    assert slug == "onsen-kusatsu.com-gozanoyu-hot-spring"


def test_written_document_carries_provenance(tmp_path: Path) -> None:
    """全文書に出典URLと取得日を持たせる（比較の前提）。"""
    result = corpus.FetchResult(
        url="https://example.org/a.html",
        status="ok",
        content_type="text/html",
        title="題",
        text="本文じゃ",
        retrieved_at="2026-08-13",
    )
    path = corpus.write_document(result, tmp_path)
    body = path.read_text(encoding="utf-8")
    assert body.startswith("---\n")
    assert "source_url: https://example.org/a.html" in body
    assert "retrieved_at: 2026-08-13" in body
    assert body.rstrip().endswith("本文じゃ")


def test_build_corpus_records_failures(tmp_path: Path) -> None:
    """取れなかった出典は理由つきで manifest に残る。"""

    def fake_fetch(url: str) -> corpus.FetchResult:
        if "bad" in url:
            return corpus.FetchResult(
                url=url, status="error", error="本文が空", retrieved_at="2026-08-13"
            )
        return corpus.FetchResult(
            url=url, status="ok", text="湯じゃ", retrieved_at="2026-08-13"
        )

    report = corpus.build_corpus(
        urls=["https://example.org/good", "https://example.org/bad"],
        out_dir=tmp_path,
        interval=0,
        fetcher=fake_fetch,
    )
    assert len(report.written) == 1
    assert report.failed == [{"source_url": "https://example.org/bad", "error": "本文が空"}]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    statuses = {entry["source_url"]: entry["status"] for entry in manifest["documents"]}
    assert statuses == {"https://example.org/good": "ok", "https://example.org/bad": "error"}
