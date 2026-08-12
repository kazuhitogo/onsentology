"""図解（DOT 出力）のテスト。

図は記事に載せるものなので、**何が実線で何が点線になるか**が壊れたら落ちるようにしておく。
点線は「推論で増えた辺」を意味する。ここが狂うと、書いた事実と導いた事実の区別が図から失われる。
"""

from __future__ import annotations

import pytest

from onsen_ontology.visualize import VIEWS, build_view


def test_views_are_declared() -> None:
    assert VIEWS == ("schema", "facility", "quality")


def test_unknown_view_is_rejected() -> None:
    with pytest.raises(ValueError, match="未知の切り口"):
        build_view("nonexistent")


def test_facility_view_contains_the_facts_and_marks_inference() -> None:
    """施設のサブグラフに、書いた事実と導いた事実の両方が出る。"""
    dot = build_view("facility", name="御座之湯")

    # 書いた事実（実線）
    assert "御座之湯" in dot
    assert "湯畑源泉" in dot and "万代源泉" in dot  # 1施設が2源泉を使う例
    assert "2.08" in dot  # 湯畑源泉の pH
    assert "泉質名の表記から読み取った泉質" in dot  # hasQualityFromName

    # 導いた事実（点線）: プロパティチェーンによる施設→適応症の畳み込み
    chain_edges = [
        line
        for line in dot.splitlines()
        if "施設の浴用適応症" in line and "style=dashed" in line
    ]
    assert chain_edges, "プロパティチェーンの推論辺が点線で描かれていない"


def test_quality_view_shows_r7_inference() -> None:
    """泉質の図に R7（禁忌症を持つ泉質 → 皮膚刺激が強い）が点線で出る。"""
    dot = build_view("quality")
    assert "酸性泉" in dot and "硫黄泉" in dot
    irritating = [
        line for line in dot.splitlines() if "皮膚刺激が強い" in line and "style=dashed" in line
    ]
    assert len(irritating) == 2, "皮膚刺激が強いのは酸性泉と硫黄泉の2泉質だけ"


def test_reified_treatment_is_not_marked_as_inferred() -> None:
    """湯使い宣言の中間ノード（空白ノード）は、書いた事実なので実線で描く。

    空白ノードは読み直すと ID が変わるため、素のグラフとの差分では推論扱いになってしまう。
    それを避けていることの確認。
    """
    dot = build_view("facility", name="大滝乃湯")
    treatment_edges = [
        line for line in dot.splitlines() if "湯使い" in line or "実施の有無" in line
    ]
    assert treatment_edges
    assert not [line for line in treatment_edges if "style=dashed" in line]


def test_schema_view_has_classes_and_property_hierarchy() -> None:
    dot = build_view("schema")
    assert "温泉施設" in dot and "源泉" in dot
    assert "subPropertyOf" in dot or "泉質" in dot
