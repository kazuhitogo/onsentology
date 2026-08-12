"""pytest 共通フィクスチャ。

推論済みグラフの生成は OWL 2 RL の演繹閉包に 25 秒前後かかるため、セッション単位で共有する。
"""

from __future__ import annotations

import pytest
from rdflib import Graph

from onsen_ontology.env import load_dotenv
from onsen_ontology.graph import load_graph, load_inferred_graph

# 実 Bedrock を呼ぶオプトインテスト（ONSEN_TEST_BEDROCK=1）でも CLI と同じ設定を使う。
load_dotenv()


@pytest.fixture(scope="session")
def raw_graph() -> Graph:
    """推論なしのグラフ。"""
    return load_graph()


@pytest.fixture(scope="session")
def graph() -> Graph:
    """推論済みグラフ（キャッシュを使う）。"""
    return load_inferred_graph()
