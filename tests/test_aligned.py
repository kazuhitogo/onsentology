"""「揃えた生ドキュメント」（比較条件 F）のテスト。

この条件は「Ontology と同じ情報を、同じ形に揃えた文書にしたら、生テキスト検索でも
届くのか」を測るためのものである。実験として成立させるために守らなければならない性質を
テストで固定する。

1. **5類型すべての実施の有無が、全施設について同じ形で書かれている**（横断して数え上げられる）
2. **推論値とヒューリスティックが混ざっていない**（答えを文書に焼き込んでいない）
3. **未公表を未公表と書いている**（「書いていない」と「無い」を区別する）
4. **出典URLと取得日を持つ**（条件 D と同じく出典を語れる）
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Graph

from onsen_ontology import aligned
from onsen_ontology.retrieval import DocumentIndex, parse_document


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory, raw_graph: Graph) -> Path:
    directory = tmp_path_factory.mktemp("aligned")
    aligned.build_aligned_corpus(out_dir=directory, graph=raw_graph)
    return directory


def _docs(corpus: Path) -> dict[str, str]:
    return {path.stem: path.read_text(encoding="utf-8") for path in corpus.glob("*.md")}


def test_every_facility_states_all_five_treatments(corpus: Path) -> None:
    """全施設の文書に、掲示義務の5類型が同じ順序・同じ形で並ぶ。

    条件 D が Q9（加水も加温も消毒もしていない施設）で落ちたのは、**「無い」の書き方が
    施設ごとに違い、4項目を同じ形で並べたページがほとんど無い**ためだった。
    揃えた文書ではそこを揃える。揃っていなければ実験の前提が崩れる。
    """
    facilities = {k: v for k, v in _docs(corpus).items() if k.startswith("facility-")}
    assert len(facilities) == 15
    for name, body in facilities.items():
        for caption in ("加水", "加温", "循環（ろ過含む）", "消毒処理", "入浴剤添加"):
            line = next(
                (row for row in body.splitlines() if row.startswith(f"- {caption}:")), None
            )
            assert line is not None, f"{name} に {caption} の行が無い"
            assert any(
                state in line
                for state in ("実施している", "実施していない", "掲示を確認できていない")
            ), f"{name} の {caption} が実施の有無を書いていない: {line}"


def test_inferred_values_and_heuristics_are_not_baked_in(corpus: Path) -> None:
    """推論値とヒューリスティックを文書に入れない。

    ``isUnmodifiedSupply``（法定4類型すべて非実施 → 無加工供給）や ``recommendedAfter``
    （仕上げ湯。法令に根拠のない経験則）を書いてしまうと、**答えを文書に置いた**ことになり
    「揃えれば届くのか」を測れない。
    """
    blob = "\n".join(_docs(corpus).values())
    for banned in ("無加工供給", "皮膚刺激", "仕上げ湯", "美肌", "美人の湯", "うちみ", "虚弱児童"):
        assert banned not in blob, f"揃えた文書に {banned} が混ざっている"


def test_unpublished_values_are_written_as_unpublished(corpus: Path) -> None:
    """未公表を未公表と書く。Q11（玉川は放射能泉か）の前提になる。"""
    tamagawa = next(v for k, v in _docs(corpus).items() if k.endswith("tamagawa"))
    assert "ラドン濃度は未公表" in tamagawa
    assert "未公表" in tamagawa
    # 収録していない温泉地は、そもそも文書が無い（Q10 の前提）
    assert all("秋保" not in body for body in _docs(corpus).values())


def test_coverage_document_declares_the_universe(corpus: Path) -> None:
    """母集合を文書に書く。「この15施設で全部」と言える文書が無いと数え上げられない。"""
    index = _docs(corpus)["index-coverage"]
    assert "15施設についてのみ記述している" in index
    assert "登別温泉 さぎり湯" in index
    assert "長湯温泉 御前湯（竹田市営）" in index


def test_quality_documents_carry_criteria_and_indications(corpus: Path) -> None:
    """泉質文書は判定基準と適応症・禁忌症を条文の表記で持つ。"""
    acidic = _docs(corpus)["quality-AcidicSpring"]
    assert "水素イオン" in acidic
    assert "1.0" in acidic
    assert "アトピー性皮膚炎" in acidic
    assert "皮膚又は粘膜の過敏な人" in acidic
    radioactive = _docs(corpus)["quality-RadioactiveSpring"]
    assert "ラドン" in radioactive


def test_every_document_carries_provenance(corpus: Path) -> None:
    """条件 D と同じく、全文書が出典URLと取得日を持つ。"""
    for name, body in _docs(corpus).items():
        meta, _ = parse_document(body)
        assert meta.get("source_url", "").startswith("http"), name
        assert meta.get("retrieved_at"), name


def test_aligned_corpus_is_searchable(corpus: Path) -> None:
    """BM25 で引ける形になっている（条件 D と同じ検索レイヤを使う）。"""
    index = DocumentIndex.from_directory(corpus)
    assert len(index) >= 29
    hits = index.search("加水 加温 循環 消毒 実施していない", top_k=3)
    assert hits
    assert any("さぎり湯" in chunk.text for chunk, _ in hits)
