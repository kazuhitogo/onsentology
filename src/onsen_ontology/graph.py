"""グラフの読み込み。

TTL は4ファイルに分かれている。

- ``onsen_ontology.ttl``   : スキーマ（TBox）。クラス、プロパティ、分類区分の個体。
- ``onsen_knowledge.ttl``  : 法定知識（規範）。掲示用泉質10種、適応症・禁忌症、利用プロトコル。
- ``onsen_instances.ttl``  : 実データ（ABox）。温泉地・源泉・施設・浴槽。
- ``onsen_heuristics.ttl`` : 独自ヒューリスティック。口語表現、条文表記の言い換え、相談の意図。

分けている理由は、出典と更新サイクルが違うため。法定知識の出典は環境省の通知（改訂は数年に一度）、
実データの出典は各施設の公式サイト（いつ変わるか分からない）、ヒューリスティックには**出典が無い**。
スキーマだけを他プロジェクトに再利用することもできる。
"""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph

from .namespaces import OID, ONSEN

#: リポジトリルート（src/onsen_ontology/graph.py から2つ上）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY_DIR = PROJECT_ROOT / "ontology"

SCHEMA_FILE = ONTOLOGY_DIR / "onsen_ontology.ttl"
KNOWLEDGE_FILE = ONTOLOGY_DIR / "onsen_knowledge.ttl"
INSTANCES_FILE = ONTOLOGY_DIR / "onsen_instances.ttl"
HEURISTICS_FILE = ONTOLOGY_DIR / "onsen_heuristics.ttl"

DEFAULT_FILES = (SCHEMA_FILE, KNOWLEDGE_FILE, INSTANCES_FILE, HEURISTICS_FILE)

#: 推論済みグラフのキャッシュ。OWL 2 RL の演繹閉包に 25 秒前後かかるため、
#: TTL とルール定義が変わっていなければ再利用する。
CACHE_FILE = PROJECT_ROOT / ".cache" / "inferred.ttl"


def load_graph(files: tuple[Path, ...] = DEFAULT_FILES) -> Graph:
    """TTL を読み込んで 1 つの Graph にまとめる。推論はしない。"""
    graph = Graph()
    graph.bind("onsen", ONSEN)
    graph.bind("oid", OID)
    for path in files:
        graph.parse(path, format="turtle")
    return graph


def _newest_input_mtime() -> float:
    """TTL と推論ルール定義のうち最も新しい更新時刻。"""
    watched = [*DEFAULT_FILES, Path(__file__).with_name("reasoning.py")]
    return max(path.stat().st_mtime for path in watched)


def load_inferred_graph(*, use_cache: bool = True) -> Graph:
    """推論適用済みのグラフを返す。

    ``use_cache=True`` のとき、TTL と ``reasoning.py`` のどちらも更新されていなければ
    ``.cache/inferred.ttl`` を読み込む。CLI やエージェントの起動を速くするためのもので、
    推論結果そのものは変わらない。
    """
    from .reasoning import apply_reasoning

    if use_cache and CACHE_FILE.exists() and CACHE_FILE.stat().st_mtime >= _newest_input_mtime():
        graph = Graph()
        graph.bind("onsen", ONSEN)
        graph.bind("oid", OID)
        graph.parse(CACHE_FILE, format="turtle")
        return graph

    graph = load_graph()
    apply_reasoning(graph)
    if use_cache:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        graph.serialize(destination=CACHE_FILE, format="turtle")
    return graph


__all__ = [
    "CACHE_FILE",
    "DEFAULT_FILES",
    "INSTANCES_FILE",
    "KNOWLEDGE_FILE",
    "ONTOLOGY_DIR",
    "PROJECT_ROOT",
    "SCHEMA_FILE",
    "load_graph",
    "load_inferred_graph",
]
