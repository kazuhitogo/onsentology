"""推論エンジン。

2種類の推論を組み合わせている。

1. **OWL 2 RL 推論（owlrl）** — プロパティチェーン公理、逆プロパティ、対称プロパティ、
   クラス階層の伝播。宣言的に書けるものはここに任せる。
2. **SPARQL ルール推論** — 数値比較・否定の集約・算術。OWL では書けない部分。

「OWL でどこまで書けて、どこから書けないか」の境界がそのまま2種類の分かれ目になっている。
たとえば「pH 8.5 以上ならアルカリ性」は OWL 2 のデータ範囲では表現しづらく（`xsd:minInclusive`
ファセットで書けても owlrl が完全にサポートしない）、「加水も循環も消毒もしていない」は
閉世界の否定であって OWL の開世界仮説では原理的に書けない。

各ルールは :class:`Rule` として根拠（法令の条項 or 独自ヒューリスティック）を必ず持つ。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import owlrl
from rdflib import Graph

PREFIXES = """
PREFIX onsen: <https://example.org/onsen#>
PREFIX oid:   <https://example.org/onsen/id/>
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
"""


@dataclass(frozen=True)
class Rule:
    """SPARQL による推論ルール1件。"""

    id: str
    label: str
    basis: str
    """根拠。法令の条項、または "heuristic:" で始まる独自ルールの説明。"""
    sparql: str
    is_heuristic: bool = field(default=False)

    @property
    def query(self) -> str:
        return PREFIXES + self.sparql


# --------------------------------------------------------------------------
# R1〜R3: 分類3軸の導出
#
# 閾値を Python にハードコードせず、オントロジー側の onsen:minPH / onsen:maxPH 等の
# 宣言を読んで判定する。指針が改訂されて境界値が変わっても TTL を直すだけで済む。
# 区分の片側しか境界を持たない場合（酸性は上限のみ、アルカリ性は下限のみ）に対応するため
# OPTIONAL + !bound() を使っている。
# --------------------------------------------------------------------------

RULE_LIQUIDITY = Rule(
    id="R1",
    label="pH から液性区分を導出",
    basis="鉱泉分析法指針 1-2(2)",
    sparql="""
INSERT { ?source onsen:liquidityClass ?class }
WHERE {
    ?source onsen:pH ?ph .
    ?class a onsen:LiquidityClass .
    OPTIONAL { ?class onsen:minPH ?min }
    OPTIONAL { ?class onsen:maxPH ?max }
    FILTER ( (!bound(?min) || ?ph >= ?min) && (!bound(?max) || ?ph < ?max) )
}
""",
)

RULE_TEMPERATURE = Rule(
    id="R2",
    label="源泉温度から泉温区分を導出",
    basis="鉱泉分析法指針 1-2(1)",
    sparql="""
INSERT { ?source onsen:temperatureClass ?class }
WHERE {
    ?source onsen:sourceTemperature ?temp .
    ?class a onsen:TemperatureClass .
    OPTIONAL { ?class onsen:minTemperature ?min }
    OPTIONAL { ?class onsen:maxTemperature ?max }
    FILTER ( (!bound(?min) || ?temp >= ?min) && (!bound(?max) || ?temp < ?max) )
}
""",
)

RULE_OSMOTIC = Rule(
    id="R3",
    label="溶存物質から浸透圧区分を導出",
    basis="鉱泉分析法指針 1-2(3)",
    sparql="""
INSERT { ?source onsen:osmoticClass ?class }
WHERE {
    ?source onsen:dissolvedSolids ?ds .
    ?class a onsen:OsmoticPressureClass .
    OPTIONAL { ?class onsen:minDissolvedSolids ?min }
    OPTIONAL { ?class onsen:maxDissolvedSolids ?max }
    FILTER ( (!bound(?min) || ?ds >= ?min) && (!bound(?max) || ?ds < ?max) )
}
""",
)

# --------------------------------------------------------------------------
# R4: 泉質判定（成分値からの逆算）
#
# 成分値が揃っている源泉に対して、掲示用泉質の判定基準（onsen:criterionProperty /
# criterionMinValue）を適用する。データセット内でこれが発火するのは大涌谷温泉のみ。
# ほとんどの施設は成分値を Web 公開していないため、実務では掲示泉質名の転記に頼るしかない。
# --------------------------------------------------------------------------

RULE_QUALITY_FROM_COMPONENTS = Rule(
    id="R4",
    label="成分値から掲示用泉質を判定（下限値型）",
    basis="鉱泉分析法指針 1-3、第1-3表",
    sparql="""
INSERT { ?source onsen:hasQuality ?quality }
WHERE {
    ?quality onsen:criterionProperty ?prop ;
             onsen:criterionMinValue ?min .
    ?source a onsen:SpringSource ;
            ?prop ?value .
    FILTER ( ?value >= ?min )
    # 塩類泉は陰イオン主成分の一致も必要
    OPTIONAL { ?quality onsen:criterionMainAnion ?requiredAnion }
    OPTIONAL { ?source onsen:mainAnion ?actualAnion }
    FILTER ( !bound(?requiredAnion) || ?requiredAnion = ?actualAnion )
}
""",
)

RULE_ALKALINE_SIMPLE = Rule(
    id="R5",
    label="単純温泉かつ現地pH8.5以上 → アルカリ性単純温泉",
    basis="鉱泉分析法指針 1-3(2)",
    sparql="""
INSERT { ?source onsen:isAlkalineSimpleSpring true }
WHERE {
    ?p rdfs:subPropertyOf* onsen:hasQuality .
    ?source ?p onsen:SimpleSpring ;
            onsen:pH ?ph .
    FILTER ( ?ph >= 8.5 )
}
""",
)

# --------------------------------------------------------------------------
# R6: 無加工供給（閉世界の否定の集約）
#
# 「加水・循環ろ過・消毒・入浴剤のいずれも実施していない」を導出する。
# OWL の開世界仮説では「宣言がない」と「実施していない」を区別できないため、
# 4類型すべてについて isApplied=false の明示的な宣言があることを要求する。
# 加温は条件に含めない（白骨温泉 泡の湯のように源泉がぬるいため加温する例があり、
# 加温を否定条件に入れると低温泉の無加工供給を表現できなくなる）。
# --------------------------------------------------------------------------

RULE_UNMODIFIED_SUPPLY = Rule(
    id="R6",
    label="法定4類型すべて非実施 → 無加工供給",
    basis="温泉法施行規則第10条第2項",
    sparql="""
INSERT { ?facility onsen:isUnmodifiedSupply true }
WHERE {
    ?facility a onsen:Facility .
    ?facility onsen:declaresTreatment ?d1, ?d2, ?d3, ?d4 .
    ?d1 onsen:treatmentType onsen:AddingWater   ; onsen:isApplied false .
    ?d2 onsen:treatmentType onsen:Recirculation ; onsen:isApplied false .
    ?d3 onsen:treatmentType onsen:Disinfection  ; onsen:isApplied false .
    ?d4 onsen:treatmentType onsen:BathAdditive  ; onsen:isApplied false .
}
""",
)

# --------------------------------------------------------------------------
# R7: 皮膚刺激の強い泉質
#
# 掲示基準 2.(2)①エ(ア) が「刺激の強い泉質（例えば酸性泉や硫黄泉等）」と例示し、
# かつ泉質別禁忌症が定められているのが酸性泉と硫黄泉の2泉質のみであることを利用する。
# 「浴用の泉質別禁忌症を持つ泉質」という宣言的な条件で書けるので、泉質名を直書きしない。
# --------------------------------------------------------------------------

RULE_SKIN_IRRITATING = Rule(
    id="R7",
    label="浴用の泉質別禁忌症を持つ泉質 → 皮膚刺激が強い",
    basis="掲示基準 2.(2)①エ(ア)、2.(1)② 泉質別禁忌症",
    sparql="""
INSERT { ?quality onsen:isSkinIrritating true }
WHERE {
    ?quality a/rdfs:subClassOf* onsen:SpringQuality ;
             onsen:hasBathingContraindication ?condition .
}
""",
)

SPARQL_RULES: tuple[Rule, ...] = (
    RULE_LIQUIDITY,
    RULE_TEMPERATURE,
    RULE_OSMOTIC,
    RULE_QUALITY_FROM_COMPONENTS,
    RULE_ALKALINE_SIMPLE,
    RULE_UNMODIFIED_SUPPLY,
    RULE_SKIN_IRRITATING,
)


#: OWL 閉包を繰り返す上限。owlrl の1回の expand は不動点に達しないことがあるため。
MAX_OWL_PASSES = 5


def apply_owl_reasoning(graph: Graph, *, max_passes: int = MAX_OWL_PASSES) -> int:
    """OWL 2 RL の演繹閉包を計算する。追加されたトリプル数を返す。

    ここで効いているのは主に次の4つ。

    - ``owl:propertyChainAxiom`` — 施設→源泉→泉質、施設→源泉→泉質→浴用適応症 の畳み込み
    - ``owl:inverseOf`` — hasSpringSource ↔ isSourceOf
    - ``owl:SymmetricProperty`` — incompatibleWith
    - ``rdfs:subClassOf`` — 塩類泉／特殊成分を含む療養泉 → 掲示用泉質

    ``expand`` を1回呼ぶだけでは不動点に達しないことがある（このグラフでは2回目に149トリプル
    増える）。推論結果をキャッシュする以上、何回流しても同じ結果になる性質は必要なので、
    追加が止まるまで繰り返す。
    """
    total = 0
    for _ in range(max_passes):
        before = len(graph)
        owlrl.DeductiveClosure(
            owlrl.OWLRL_Semantics,
            axiomatic_triples=False,
            datatype_axioms=False,
        ).expand(graph)
        added = len(graph) - before
        total += added
        if added == 0:
            break
    return total


def apply_sparql_rules(graph: Graph, rules: tuple[Rule, ...] = SPARQL_RULES) -> dict[str, int]:
    """SPARQL ルールを順に適用し、ルールIDごとの追加トリプル数を返す。"""
    added: dict[str, int] = {}
    for rule in rules:
        before = len(graph)
        graph.update(rule.query)
        added[rule.id] = len(graph) - before
    return added


def apply_reasoning(graph: Graph, *, owl: bool = True) -> dict[str, int]:
    """推論を全部適用する。

    **SPARQL ルールを先に、OWL 推論を後に走らせる。** この順序には理由が2つある。

    1. SPARQL ルールは OWL の導出結果に依存しない。クラス階層が必要な R7 は
       ``?quality a/rdfs:subClassOf* onsen:SpringQuality`` とプロパティパスで書いており、
       プロパティ階層が必要な R5 も ``?p rdfs:subPropertyOf* onsen:hasQuality`` と書いている。
       どちらも推論器なしで階層をたどれる（実データの泉質は多くが
       ``onsen:hasQualityFromName`` = hasQuality の下位プロパティで主張されている）。
    2. 逆順（OWL → SPARQL）にすると、SPARQL が追加した ``onsen:liquidityClass`` 等に対して
       OWL のドメイン・レンジ推論が未適用のまま残り、パイプライン全体が不動点にならない。
       つまり2回流すとトリプルが増える。冪等でない推論はキャッシュと相性が悪い。

    :return: ``{"R1": 追加数, ..., "OWL": 追加数}``
    """
    result: dict[str, int] = {}
    result.update(apply_sparql_rules(graph))
    if owl:
        result["OWL"] = apply_owl_reasoning(graph)
    return result


__all__ = [
    "MAX_OWL_PASSES",
    "PREFIXES",
    "SPARQL_RULES",
    "Rule",
    "apply_owl_reasoning",
    "apply_reasoning",
    "apply_sparql_rules",
]
