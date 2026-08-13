"""生テキスト検索（BM25）のテスト（Phase 7）。

条件 D の対抗馬としての公平性を担保するためのテストである。特に見るのは2点。

1. **数値が索引から落ちないこと**。「pH2.08 を答えられるか」という問いが検索の失敗に
   すり替わってはならない
2. **無いものを無いと返すこと**。一致する語が1つも無ければ件数0を返す。ただし文字bigramでは
   「秋保温泉」の「温泉」が他の文書に一致するので、収録していない温泉地でも隣接情報が返る。
   これは生テキスト検索の性質そのもので、閾値で隠さない（取れ高を細工しない）

コーパス本体（``corpus/``）は git 管理外なので、ここでは合成した小さなコーパスで検証する。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from onsen_ontology import retrieval

_DOC_A = """---
source_url: http://onsen-kusatsu.com/gozanoyu/hot-spring/
retrieved_at: 2026-08-13
content_type: text/html
title: 御座之湯 泉質
---
泉質・効能 源泉は「湯畑源泉」と「万代源泉」の二種類。

湯畑源泉 pH2.08 硫黄泉。万代源泉 pH1.7 酸性塩化物硫酸塩温泉。
"""

_DOC_B = """---
source_url: https://www.env.go.jp/nature/onsen/pdf/2-5_p_11.pdf
retrieved_at: 2026-08-13
content_type: application/pdf
title:
---
飲用の方法及び注意 1回100mL〜150mL程度、1日の量はおよそ200mL〜500mLまでとすること。

pH3未満の温泉を飲用する場合は、真水で薄めてpH3以上とし、1回100mLまでとすること。
"""


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    (tmp_path / "gozanoyu.md").write_text(_DOC_A, encoding="utf-8")
    (tmp_path / "keiji-kijun.md").write_text(_DOC_B, encoding="utf-8")
    return tmp_path


def test_tokenize_keeps_numbers_and_ascii() -> None:
    tokens = retrieval.tokenize("湯畑源泉 pH2.08")
    assert "ph" in tokens
    assert "2.08" in tokens
    # 日本語は文字bigram
    assert "湯畑" in tokens
    assert "源泉" in tokens


def test_parse_document_reads_front_matter() -> None:
    meta, body = retrieval.parse_document(_DOC_A)
    assert meta["source_url"] == "http://onsen-kusatsu.com/gozanoyu/hot-spring/"
    assert meta["retrieved_at"] == "2026-08-13"
    assert body.lstrip().startswith("泉質・効能")


def test_split_chunks_respects_maximum() -> None:
    body = "\n\n".join(["あ" * 300 for _ in range(6)])
    chunks = retrieval.split_chunks(body, target=400, maximum=700)
    assert chunks
    assert all(len(chunk) <= 700 for chunk in chunks)
    # 段落境界の無い長い塊も上限で切る
    long_chunks = retrieval.split_chunks("い" * 2000, target=400, maximum=700)
    assert all(len(chunk) <= 700 for chunk in long_chunks)
    assert "".join(long_chunks) == "い" * 2000


def test_every_chunk_carries_provenance(corpus_dir: Path) -> None:
    """出典URLと取得日は全チャンクに付く。RAG 側も出典を語れなければ比較が成立しない。"""
    chunks = retrieval.load_chunks(corpus_dir)
    assert chunks
    for chunk in chunks:
        assert chunk.source_url.startswith("http")
        assert chunk.retrieved_at == "2026-08-13"
        assert chunk.chunk_id.startswith(chunk.document_id)
        assert "出典URL" in chunk.to_dict()


def test_search_finds_the_published_value(corpus_dir: Path) -> None:
    tools = retrieval.DocumentSearchTools(corpus_dir=corpus_dir)
    result = tools.search_documents("湯畑源泉 pH")
    assert result["件数"] >= 1
    top = result["結果"][0]
    assert "2.08" in top["本文"]
    assert top["出典URL"].startswith("http")


def test_search_returns_neighbours_for_absent_topics(corpus_dir: Path) -> None:
    """コーパスに無い温泉地を引くと、**隣接する何か**が返る。

    「秋保温泉」は文字bigramで「温泉」が一致するので、収録していない温泉地でもヒットが出る。
    これは実装の不備ではなく生テキスト検索の性質そのもので、条件 D の「無いと言えない」という
    弱点の正体である（形態素解析器に替えても「温泉」で引っかかることは変わらない）。
    ツールの側で件数を隠したり閾値で切ったりはしない。**取れ高を細工せずに測る**ためである。
    """
    tools = retrieval.DocumentSearchTools(corpus_dir=corpus_dir)
    result = tools.search_documents("秋保温泉")
    assert result["件数"] > 0
    assert all("秋保" not in hit["本文"] for hit in result["結果"])
    # 一致する語が1つも無ければ0件を返す（そこは黙って埋めない）
    empty = tools.search_documents("zzzqqq")
    assert empty["件数"] == 0
    assert "見つからなかった" in empty["注記"]


def test_fetch_document_returns_a_window(corpus_dir: Path) -> None:
    tools = retrieval.DocumentSearchTools(corpus_dir=corpus_dir)
    fetched = tools.fetch_document("keiji-kijun")
    assert "100mL" in fetched["本文"]
    assert fetched["出典URL"].endswith("2-5_p_11.pdf")
    assert fetched["続きの offset"] is None
    # chunk_id を渡しても文書として引ける
    assert tools.fetch_document("keiji-kijun#0")["document_id"] == "keiji-kijun"
    assert "error" in tools.fetch_document("no-such-document")


def test_fetch_document_pages_through_long_text(tmp_path: Path) -> None:
    body = "---\nsource_url: https://example.org/a\nretrieved_at: 2026-08-13\n---\n" + (
        "湯" * (retrieval.FETCH_WINDOW_CHARS * 2)
    )
    (tmp_path / "long.md").write_text(body, encoding="utf-8")
    tools = retrieval.DocumentSearchTools(corpus_dir=tmp_path)
    first = tools.fetch_document("long")
    assert len(first["本文"]) == retrieval.FETCH_WINDOW_CHARS
    assert first["続きの offset"] == retrieval.FETCH_WINDOW_CHARS
    second = tools.fetch_document("long", offset=first["続きの offset"])
    assert second["offset"] == retrieval.FETCH_WINDOW_CHARS
    assert second["本文"]
