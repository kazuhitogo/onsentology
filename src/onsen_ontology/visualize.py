"""グラフの一部を DOT 形式で書き出す（図解用）。

4,176トリプルを丸ごと描いても読めないので、**見たい切り口だけを SPARQL CONSTRUCT で
切り出して**描く。切り出し方そのものが「このオントロジーで何が言えるか」の説明になる。

用意した切り口は3つ。

- ``schema``   : スキーマ（TBox）。クラスと、クラス間をつなぐプロパティの定義
- ``facility`` : 1施設のサブグラフ。施設 → 源泉 → 泉質 → 適応症／禁忌症と湯使いの宣言
- ``quality``  : 掲示用泉質10種と適応症・禁忌症の二部グラフ

推論で増えたトリプルは点線で描く。**どこまでが書いたもので、どこからが導かれたものか**が
図で分かるようにするためである。
"""

from __future__ import annotations

import html
from collections.abc import Iterable

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from .graph import load_graph, load_inferred_graph
from .namespaces import ONSEN

#: 図に出さないプロパティ。説明文・出典・ラベルは線が多すぎて読めなくなる。
#: ``owl:sameAs`` は OWL 2 RL が全ノードに反射律で付ける自己ループなので落とす。
NOISY = {
    RDFS.comment,
    RDFS.label,
    OWL.sameAs,
    ONSEN.dataStatus,
    ONSEN.legalBasis,
    ONSEN.heuristicRule,
    ONSEN.colloquialExpression,
    ONSEN.nonStandardParaphrase,
    URIRef("http://purl.org/dc/terms/source"),
    URIRef("http://purl.org/dc/terms/title"),
    URIRef("http://purl.org/dc/terms/created"),
}

#: 図に出さないノード（推論器が付ける最上位クラス）
NOISY_NODES = {OWL.Thing, RDFS.Resource, OWL.NamedIndividual}

def _facility_triples(graph: Graph, name: str) -> list[tuple[object, object, object]]:
    """施設を起点に、源泉→泉質→適応症／禁忌症と湯使い宣言まで辿って集める。

    同じことを SPARQL の CONSTRUCT でも書けるが、入れ子 OPTIONAL を含む形は rdflib の
    エンジンで 90 秒かかった（結果は同じ）。**素直に辿るほうが速い**ので走査にしている。
    グラフ探索の一部は SPARQL より手続きで書いたほうがよいという実例。
    """
    facility = next(
        (
            candidate
            for candidate in graph.subjects(RDF.type, ONSEN.Facility)
            if name in str(graph.value(candidate, RDFS.label) or "")
        ),
        None,
    )
    if facility is None:
        raise ValueError(f"施設が見つからない: {name}")

    triples: list[tuple[object, object, object]] = [
        (facility, p, o) for p, o in graph.predicate_objects(facility)
    ]
    for source in graph.objects(facility, ONSEN.hasSpringSource):
        triples += [(source, p, o) for p, o in graph.predicate_objects(source)]
        for quality in graph.objects(source, ONSEN.hasQuality):
            for predicate in (ONSEN.hasBathingIndication, ONSEN.hasBathingContraindication):
                triples += [(quality, predicate, o) for o in graph.objects(quality, predicate)]
    for declaration in graph.objects(facility, ONSEN.declaresTreatment):
        triples += [(declaration, p, o) for p, o in graph.predicate_objects(declaration)]
    return triples


Q_SCHEMA = """
CONSTRUCT {
    ?class a owl:Class ; rdfs:label ?classLabel ; rdfs:subClassOf ?super .
    ?prop rdfs:domain ?domain ; rdfs:range ?range ; rdfs:label ?propLabel .
    ?prop rdfs:subPropertyOf ?superProp .
}
WHERE {
    { ?class a owl:Class . OPTIONAL { ?class rdfs:label ?classLabel }
      OPTIONAL { ?class rdfs:subClassOf ?super } }
    UNION
    { ?prop a owl:ObjectProperty ; rdfs:domain ?domain ; rdfs:range ?range .
      OPTIONAL { ?prop rdfs:label ?propLabel }
      OPTIONAL { ?prop rdfs:subPropertyOf ?superProp } }
}
"""

Q_QUALITY = """
CONSTRUCT {
    ?quality a ?type ; rdfs:label ?qLabel ;
             onsen:hasBathingIndication ?indication ;
             onsen:hasBathingContraindication ?contra ;
             onsen:recommendedAfter ?after ;
             onsen:isSkinIrritating ?irritating .
    ?indication rdfs:label ?iLabel .
    ?contra rdfs:label ?cLabel .
}
WHERE {
    ?quality a ?type ; rdfs:label ?qLabel .
    ?type rdfs:subClassOf* onsen:SpringQuality .
    OPTIONAL { ?quality onsen:hasBathingIndication ?indication . ?indication rdfs:label ?iLabel }
    OPTIONAL { ?quality onsen:hasBathingContraindication ?contra . ?contra rdfs:label ?cLabel }
    OPTIONAL { ?quality onsen:recommendedAfter ?after }
    OPTIONAL { ?quality onsen:isSkinIrritating ?irritating }
}
"""

VIEWS = ("schema", "facility", "quality")


def _label(graph: Graph, node: object) -> str:
    """ノードの表示名。ラベルがあれば日本語ラベル、無ければ URI の末尾。"""
    if isinstance(node, Literal):
        text = str(node)
        return text if len(text) <= 28 else text[:27] + "…"
    if isinstance(node, BNode):
        return "（中間ノード）"
    if isinstance(node, URIRef):
        label = graph.value(node, RDFS.label)
        if label is not None:
            return str(label)
        text = str(node)
        return text.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    return str(node)


def _node_id(node: object) -> str:
    return f'"{html.escape(str(node))}"'


def _style(graph: Graph, node: object) -> str:
    """クラスは箱、個体は丸角、リテラルは平箱。種類ごとに色を変える。"""
    if isinstance(node, Literal):
        return 'shape=box, style="filled", fillcolor="#f5f5f5", fontcolor="#333333"'
    if isinstance(node, BNode):
        return 'shape=diamond, style="filled", fillcolor="#fff3e0"'
    types = set(graph.objects(node, RDF.type))
    if OWL.Class in types:
        return 'shape=box, style="filled,rounded", fillcolor="#e3f2fd"'
    if ONSEN.Facility in types:
        return 'shape=box, style="filled,rounded,bold", fillcolor="#e8f5e9"'
    if ONSEN.SpringSource in types:
        return 'shape=box, style="filled,rounded", fillcolor="#fffde7"'
    if ONSEN.HealthCondition in types:
        return 'shape=ellipse, style="filled", fillcolor="#fce4ec"'
    if any(t in types for t in (ONSEN.SaltSpringQuality, ONSEN.SpecialComponentSpringQuality)):
        return 'shape=box, style="filled,rounded", fillcolor="#ede7f6"'
    return 'shape=box, style="filled,rounded", fillcolor="#ffffff"'


def to_dot(
    triples: Iterable[tuple[object, object, object]],
    *,
    labels: Graph,
    inferred: set[tuple[object, object, object]] | None = None,
    title: str = "",
) -> str:
    """トリプル列を DOT にする。``inferred`` に含まれる辺は点線で描く。"""
    inferred = inferred or set()
    lines = [
        "digraph onsen {",
        '  graph [rankdir=LR, fontname="Noto Sans CJK JP", labelloc=t, '
        f'label="{html.escape(title)}", fontsize=18];',
        '  node [fontname="Noto Sans CJK JP", fontsize=11];',
        '  edge [fontname="Noto Sans CJK JP", fontsize=9, color="#555555"];',
    ]
    seen: set[str] = set()
    for subject, predicate, obj in triples:
        for node in (subject, obj):
            node_id = _node_id(node)
            if node_id not in seen:
                seen.add(node_id)
                lines.append(
                    f"  {node_id} [label=\"{html.escape(_label(labels, node))}\", "
                    f"{_style(labels, node)}];"
                )
        style = ' style=dashed, color="#c62828"' if (subject, predicate, obj) in inferred else ""
        lines.append(
            f"  {_node_id(subject)} -> {_node_id(obj)} "
            f'[label="{html.escape(_label(labels, predicate))}"{style}];'
        )
    lines.append("}")
    return "\n".join(lines)


def build_view(view: str, *, name: str = "", mark_inferred: bool = True) -> str:
    """切り口を1つ選んで DOT を返す。"""
    if view not in VIEWS:
        raise ValueError(f"未知の切り口: {view}（{', '.join(VIEWS)} のいずれか）")

    inferred_graph = load_inferred_graph()

    if view == "facility":
        triples = _facility_triples(inferred_graph, name)
        title = f"施設のサブグラフ: {name}"
    else:
        from rdflib.plugins.sparql import prepareQuery

        from .queries import PREFIXES

        query, title = {
            "schema": (Q_SCHEMA, "スキーマ（TBox）: クラスとプロパティ"),
            "quality": (Q_QUALITY, "掲示用泉質と適応症・禁忌症"),
        }[view]
        triples = list(inferred_graph.query(prepareQuery(PREFIXES + query)))

    triples = [
        (s, p, o)
        for s, p, o in triples
        if p not in NOISY and s not in NOISY_NODES and o not in NOISY_NODES
    ]

    marked: set[tuple[object, object, object]] = set()
    if mark_inferred:
        raw = load_graph()
        # 空白ノード（湯使い宣言の中間ノードなど）は読み直すと ID が変わるので、
        # 素のグラフとの差分では「推論で増えた」と誤判定される。BNode を含む辺は除く。
        marked = {
            (s, p, o)
            for s, p, o in triples
            if (s, p, o) not in raw and not isinstance(s, BNode) and not isinstance(o, BNode)
        }

    return to_dot(triples, labels=inferred_graph, inferred=marked, title=title)


__all__ = ["VIEWS", "build_view", "to_dot"]
