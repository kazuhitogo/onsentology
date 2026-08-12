"""SPARQL 検索レイヤ。

エージェントのツールとして呼ばれることを前提に、戻り値はすべて JSON 化できる
``dict`` / ``list`` にしている。ここで返さなかった情報は LLM も知らないままになるので、
「値が無い」ことと「未確認である」ことを区別して返すのが重要。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from .namespaces import ONSEN
from .reasoning import PREFIXES

# --------------------------------------------------------------------------
# 症状の口語表現
#
# 掲示基準の適応症は「疲労回復」「末梢循環障害」のような硬い用語で書かれている。
# 利用者が使う口語からの橋渡しは onsen:colloquialExpression としてグラフ側に宣言してある
# （法令に根拠のない独自ヒューリスティックなので onsen:heuristicRule 注記が付いている）。
# Python の辞書で持たないのは、閾値をグラフに置くのと同じ理由である。
# --------------------------------------------------------------------------

_colloquial_cache: dict[int, dict[URIRef, tuple[str, ...]]] = {}


def colloquial_expressions(graph: Graph) -> dict[URIRef, tuple[str, ...]]:
    """``症状・病態 -> 口語表現`` の対応をグラフから読む。"""
    key = id(graph)
    cached = _colloquial_cache.get(key)
    if cached is None:
        table: dict[URIRef, list[str]] = {}
        for condition, word in graph.subject_objects(ONSEN.colloquialExpression):
            table.setdefault(condition, []).append(str(word))
        cached = {uri: tuple(sorted(words)) for uri, words in table.items()}
        _colloquial_cache[key] = cached
    return cached


def paraphrase_table(graph: Graph) -> dict[str, str]:
    """``条文表記でない言い換え -> 条文表記`` の対応をグラフから読む。

    検算（:mod:`onsen_ontology.verify`）が「掲示基準の表記に直せ」と指摘するために使う。
    """
    table: dict[str, str] = {}
    for condition, word in graph.subject_objects(ONSEN.nonStandardParaphrase):
        label = graph.value(condition, RDFS.label)
        if label is not None:
            table[str(word)] = str(label)
    return table


def _label(graph: Graph, uri: URIRef | None) -> str | None:
    if uri is None:
        return None
    value = graph.value(uri, URIRef("http://www.w3.org/2000/01/rdf-schema#label"))
    return str(value) if value is not None else str(uri)


def _lit(value: Any) -> Any:
    """rdflib のリテラルを素の Python 値に落とす。

    ``xsd:decimal`` は ``decimal.Decimal`` になり ``json.dumps`` が通らないので float にする。
    ツールの戻り値を LLM に渡す都合上、ここで JSON 化可能にしておく必要がある。
    """
    if isinstance(value, Literal):
        native = value.toPython()
        if isinstance(native, Decimal):
            return float(native)
        if isinstance(native, date):
            return native.isoformat()
        return native
    if isinstance(value, URIRef):
        return str(value)
    return value


# --------------------------------------------------------------------------
# 一覧・詳細
# --------------------------------------------------------------------------

Q_LIST_FACILITIES = (
    PREFIXES
    + """
SELECT ?facility ?name ?area ?address
WHERE {
    ?facility a onsen:Facility ; rdfs:label ?name .
    OPTIONAL { ?facility onsen:locatedInArea ?areaUri . ?areaUri rdfs:label ?area }
    OPTIONAL { ?facility onsen:address ?address }
}
ORDER BY ?name
"""
)


def list_facilities(graph: Graph) -> list[dict[str, Any]]:
    """登録されている全施設の一覧。"""
    return [
        {
            "uri": str(row.facility),
            "施設名": str(row.name),
            "温泉地": _lit(row.area),
            "所在地": _lit(row.address),
        }
        for row in graph.query(Q_LIST_FACILITIES)
    ]


Q_FACILITY_DETAIL = (
    PREFIXES
    + """
SELECT ?name ?area ?address ?claims ?unmodified ?drinking ?comment ?status
WHERE {
    ?facility rdfs:label ?name .
    OPTIONAL { ?facility onsen:locatedInArea ?areaUri . ?areaUri rdfs:label ?area }
    OPTIONAL { ?facility onsen:address ?address }
    OPTIONAL { ?facility onsen:claimsKakenagashi ?claims }
    OPTIONAL { ?facility onsen:isUnmodifiedSupply ?unmodified }
    OPTIONAL { ?facility onsen:drinkingPermitted ?drinking }
    OPTIONAL { ?facility rdfs:comment ?comment }
    OPTIONAL { ?facility onsen:dataStatus ?status }
}
"""
)

Q_FACILITY_SOURCES = (
    PREFIXES
    + """
SELECT DISTINCT ?source ?srcName ?displayed ?legacy ?pH ?temp ?liquidity ?tempClass ?osmotic ?status
WHERE {
    ?facility onsen:hasSpringSource ?source .
    ?source rdfs:label ?srcName .
    OPTIONAL { ?source onsen:displayedQualityName ?displayed }
    OPTIONAL { ?source onsen:legacyQualityName ?legacy }
    OPTIONAL { ?source onsen:pH ?pH }
    OPTIONAL { ?source onsen:sourceTemperature ?temp }
    OPTIONAL { ?source onsen:liquidityClass ?lc . ?lc rdfs:label ?liquidity }
    OPTIONAL { ?source onsen:temperatureClass ?tc . ?tc rdfs:label ?tempClass }
    OPTIONAL { ?source onsen:osmoticClass ?oc . ?oc rdfs:label ?osmotic }
    OPTIONAL { ?source onsen:dataStatus ?status }
}
ORDER BY ?srcName
"""
)

Q_SOURCE_QUALITIES = (
    PREFIXES
    + """
SELECT DISTINCT ?quality ?qName
WHERE {
    ?source onsen:hasQuality ?quality .
    ?quality rdfs:label ?qName .
}
ORDER BY ?qName
"""
)

Q_FACILITY_TREATMENTS = (
    PREFIXES
    + """
SELECT ?type ?typeName ?applied ?reason
WHERE {
    ?facility onsen:declaresTreatment ?decl .
    ?decl onsen:treatmentType ?type ; onsen:isApplied ?applied .
    ?type rdfs:label ?typeName .
    OPTIONAL { ?decl onsen:reason ?reason }
}
ORDER BY ?typeName
"""
)

Q_FACILITY_SOURCE_URLS = (
    PREFIXES
    + """
SELECT DISTINCT ?url
WHERE {
    { ?facility <http://purl.org/dc/terms/source> ?url }
    UNION
    { ?facility onsen:hasSpringSource ?src . ?src <http://purl.org/dc/terms/source> ?url }
}
"""
)


def describe_facility(graph: Graph, facility: str) -> dict[str, Any] | None:
    """施設1件の詳細。``facility`` は URI か施設名の部分一致。"""
    uri = resolve_facility(graph, facility)
    if uri is None:
        return None

    detail: dict[str, Any] = {"uri": str(uri)}
    for row in graph.query(Q_FACILITY_DETAIL, initBindings={"facility": uri}):
        detail.update(
            {
                "施設名": str(row.name),
                "温泉地": _lit(row.area),
                "所在地": _lit(row.address),
                "源泉かけ流しを自主表示": _lit(row.claims),
                "無加工供給（推論値）": _lit(row.unmodified),
                "飲用許可": _lit(row.drinking),
                "備考": _lit(row.comment),
                "データ状態": _lit(row.status),
            }
        )
        break

    sources: list[dict[str, Any]] = []
    for row in graph.query(Q_FACILITY_SOURCES, initBindings={"facility": uri}):
        qualities = [
            str(q.qName)
            for q in graph.query(Q_SOURCE_QUALITIES, initBindings={"source": row.source})
        ]
        sources.append(
            {
                "源泉名": str(row.srcName),
                "掲示泉質名": _lit(row.displayed),
                "旧泉質名": _lit(row.legacy),
                "掲示用泉質": qualities,
                "pH": _lit(row.pH),
                "源泉温度": _lit(row.temp),
                "液性区分": _lit(row.liquidity),
                "泉温区分": _lit(row.tempClass),
                "浸透圧区分": _lit(row.osmotic),
                "データ状態": _lit(row.status),
            }
        )
    detail["源泉"] = sources

    detail["湯使い"] = [
        {
            "類型": str(row.typeName),
            "実施": _lit(row.applied),
            "理由": _lit(row.reason),
        }
        for row in graph.query(Q_FACILITY_TREATMENTS, initBindings={"facility": uri})
    ]

    detail["出典"] = sorted(
        str(row.url) for row in graph.query(Q_FACILITY_SOURCE_URLS, initBindings={"facility": uri})
    )
    return {k: v for k, v in detail.items() if v is not None}


Q_ALL_FACILITY_LABELS = (
    PREFIXES
    + """
SELECT ?facility ?name
WHERE { ?facility a onsen:Facility ; rdfs:label ?name }
"""
)


def resolve_facility(graph: Graph, facility: str) -> URIRef | None:
    """URI 文字列または施設名から施設 URI を引く。

    LLM は「有馬 金の湯」のように語を組み合わせた名前を渡してくる。実際のラベルは
    「有馬本温泉 金の湯」なので単純な部分一致では外れる。そこで3段構えにした。

    1. ラベルの部分一致
    2. 空白で分割したトークンが**すべて**ラベルに含まれる
    3. トークンのどれかがラベルに含まれる（最も多く一致したものを採る）
    """
    if facility.startswith("http"):
        uri = URIRef(facility)
        return uri if (uri, None, None) in graph else None

    candidates = [(row.facility, str(row.name)) for row in graph.query(Q_ALL_FACILITY_LABELS)]

    exact = [(uri, name) for uri, name in candidates if facility in name]
    if exact:
        # 最短の名前を返す（「有馬」で2施設が当たった場合の安定化）
        return min(exact, key=lambda pair: len(pair[1]))[0]

    tokens = [t for t in facility.replace("　", " ").split() if t]
    if len(tokens) > 1:
        all_match = [(uri, name) for uri, name in candidates if all(t in name for t in tokens)]
        if all_match:
            return min(all_match, key=lambda pair: len(pair[1]))[0]

        scored = [
            (sum(1 for t in tokens if t in name), -len(name), uri)
            for uri, name in candidates
            if any(t in name for t in tokens)
        ]
        if scored:
            return max(scored)[2]
    return None


# --------------------------------------------------------------------------
# 症状からの検索
# --------------------------------------------------------------------------

Q_CONDITIONS_BY_LABEL = (
    PREFIXES
    + """
SELECT ?condition ?label
WHERE {
    ?condition a onsen:HealthCondition ; rdfs:label ?label .
    FILTER ( CONTAINS(?label, ?needle) )
}
"""
)


def resolve_conditions(graph: Graph, keyword: str) -> list[tuple[URIRef, str]]:
    """口語のキーワードから、掲示基準の用語に対応する症状・病態を引く。

    口語表現はグラフ側（``onsen:colloquialExpression``）に宣言してある。以前は Python の
    辞書で持っていたが、閾値をグラフに置くのと同じ理由でオントロジーに移した。
    口語表現での部分一致 → 見つからなければラベルの部分一致、の2段構え。
    """
    needles: list[str] = []
    for condition, words in colloquial_expressions(graph).items():
        if any(word in keyword for word in words):
            needles.append(str(graph.value(condition, RDFS.label)))
    if not needles:
        needles = [keyword]

    found: dict[URIRef, str] = {}
    for needle in needles:
        for row in graph.query(Q_CONDITIONS_BY_LABEL, initBindings={"needle": Literal(needle)}):
            found[row.condition] = str(row.label)
    return sorted(found.items(), key=lambda kv: kv[1])


Q_FACILITIES_BY_INDICATION = (
    PREFIXES
    + """
SELECT DISTINCT ?facility ?facName ?source ?srcName ?quality ?qName ?area
WHERE {
    ?quality onsen:hasBathingIndication ?condition ;
             rdfs:label ?qName .
    ?source onsen:hasQuality ?quality ; rdfs:label ?srcName .
    ?facility onsen:hasSpringSource ?source ; rdfs:label ?facName .
    OPTIONAL { ?facility onsen:locatedInArea ?areaUri . ?areaUri rdfs:label ?area }
}
ORDER BY ?facName
"""
)


def find_facilities_by_symptom(graph: Graph, keyword: str) -> dict[str, Any]:
    """症状の口語表現から、泉質別浴用適応症が一致する施設を探す。

    返り値には必ず「これは掲示基準上の泉質別適応症であって医学的推奨ではない」旨の
    注記を含める。LLM がこの注記を読んで回答に反映することを期待している。
    """
    conditions = resolve_conditions(graph, keyword)
    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for condition_uri, condition_label in conditions:
        for row in graph.query(
            Q_FACILITIES_BY_INDICATION, initBindings={"condition": condition_uri}
        ):
            # 1施設が同じ泉質の源泉を複数持つ場合（長湯 御前湯は3源泉すべて炭酸水素塩泉）に
            # 同一行が並ぶので、施設・泉質・適応症の組で重複を落とす。
            key = (str(row.facility), str(row.qName), condition_label)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                {
                    "施設名": str(row.facName),
                    "温泉地": _lit(row.area),
                    "源泉名": str(row.srcName),
                    "泉質": str(row.qName),
                    "適応症": condition_label,
                    "uri": str(row.facility),
                }
            )
    general = [
        label
        for _, label in [
            (uri, _label(graph, uri))
            for uri in graph.objects(
                ONSEN.TherapeuticSpringGeneralProfile, ONSEN.hasBathingIndication
            )
        ]
        if label
    ]
    return {
        "検索語": keyword,
        "対応づけた掲示基準の用語": [label for _, label in conditions],
        "泉質別適応症で一致した施設": matches,
        "療養泉の一般的適応症（泉質を問わず共通）": sorted(general),
        "注記": (
            "これは環境省「掲示基準」が定める泉質別適応症・一般的適応症の区分であり、"
            "医学的な治療推奨ではない。適応症でも病期や状態によっては悪化する場合があるため、"
            "掲示基準自体が専門的知識を有する医師の指示・指導のもとで行うことが望ましいとしている。"
        ),
    }


# --------------------------------------------------------------------------
# 泉質の詳細
# --------------------------------------------------------------------------

Q_QUALITY_DETAIL = (
    PREFIXES
    + """
SELECT ?quality ?name ?comment ?basis ?irritating ?critProp ?critMin ?critMax
WHERE {
    ?quality a/rdfs:subClassOf* onsen:SpringQuality ; rdfs:label ?name .
    FILTER ( CONTAINS(?name, ?needle) )
    OPTIONAL { ?quality rdfs:comment ?comment }
    OPTIONAL { ?quality onsen:legalBasis ?basis }
    OPTIONAL { ?quality onsen:isSkinIrritating ?irritating }
    OPTIONAL { ?quality onsen:criterionProperty ?critProp }
    OPTIONAL { ?quality onsen:criterionMinValue ?critMin }
    OPTIONAL { ?quality onsen:criterionMaxValue ?critMax }
}
"""
)


def describe_spring_quality(graph: Graph, name: str) -> dict[str, Any] | None:
    """掲示用泉質1件の詳細（判定基準・適応症・禁忌症・刺激の強さ）。"""
    rows = list(graph.query(Q_QUALITY_DETAIL, initBindings={"needle": Literal(name)}))
    if not rows:
        return None
    row = min(rows, key=lambda r: len(str(r.name)))
    quality = row.quality

    def labels(prop: URIRef) -> list[str]:
        return sorted(filter(None, (_label(graph, obj) for obj in graph.objects(quality, prop))))

    return {
        "uri": str(quality),
        "泉質名": str(row.name),
        "判定基準": {
            "対象成分": _label(graph, row.critProp) if row.critProp else None,
            "下限値": _lit(row.critMin),
            "上限値": _lit(row.critMax),
        },
        "法的根拠": _lit(row.basis),
        "解説": _lit(row.comment),
        "皮膚刺激が強い（推論値）": _lit(row.irritating),
        "浴用適応症": labels(ONSEN.hasBathingIndication),
        "飲用適応症": labels(ONSEN.hasDrinkingIndication),
        "泉質別浴用禁忌症": labels(ONSEN.hasBathingContraindication),
        "仕上げ湯として推奨される先行泉質": labels(ONSEN.recommendedAfter),
        "連続利用が非推奨の泉質": labels(ONSEN.incompatibleWith),
    }


# --------------------------------------------------------------------------
# 一般的適応症・禁忌症・プロトコル
# --------------------------------------------------------------------------


def general_indications(graph: Graph) -> dict[str, Any]:
    """療養泉の一般的適応症と温泉の一般的禁忌症（浴用）。"""
    profile = ONSEN.TherapeuticSpringGeneralProfile
    return {
        "療養泉の一般的適応症（浴用）": sorted(
            filter(
                None, (_label(graph, o) for o in graph.objects(profile, ONSEN.hasBathingIndication))
            )
        ),
        "温泉の一般的禁忌症（浴用）": sorted(
            filter(
                None,
                (
                    _label(graph, o)
                    for o in graph.objects(profile, ONSEN.hasBathingContraindication)
                ),
            )
        ),
        "法的根拠": str(graph.value(profile, ONSEN.legalBasis)),
    }


def bathing_protocol(graph: Graph) -> dict[str, Any]:
    """浴用の方法及び注意（法定の数値制約）。"""
    p = ONSEN.StandardBathingProtocol
    return {
        "1日入浴回数上限（開始後数日間）": _lit(graph.value(p, ONSEN.maxBathsPerDayInitial)),
        "1日入浴回数上限（慣れた後）": _lit(graph.value(p, ONSEN.maxBathsPerDayAdapted)),
        "1回入浴時間上限（初期・分）": _lit(graph.value(p, ONSEN.maxMinutesPerBathInitial)),
        "1回入浴時間上限（慣れた後・分）": _lit(graph.value(p, ONSEN.maxMinutesPerBathAdapted)),
        "入浴後安静時間（分）": _lit(graph.value(p, ONSEN.restMinutesAfterBath)),
        "高温浴注意の閾値（℃）": _lit(graph.value(p, ONSEN.highTemperatureCautionThreshold)),
        "湯あたり発現時期（日）": [
            _lit(graph.value(p, ONSEN.onsetDaysMin)),
            _lit(graph.value(p, ONSEN.onsetDaysMax)),
        ],
        "条文の要点": str(graph.value(p, URIRef("http://www.w3.org/2000/01/rdf-schema#comment"))),
        "法的根拠": str(graph.value(p, ONSEN.legalBasis)),
    }


def drinking_protocol(graph: Graph) -> dict[str, Any]:
    """飲用の方法及び注意（法定の数値制約）。"""
    p = ONSEN.StandardDrinkingProtocol
    return {
        "1回飲用量上限（mL）": _lit(graph.value(p, ONSEN.drinkingMlPerServingMax)),
        "1日飲用総量上限（mL）": _lit(graph.value(p, ONSEN.drinkingMlPerDayMax)),
        "飲用可能年齢の下限": _lit(graph.value(p, ONSEN.minAgeForDrinking)),
        "条文の要点": str(graph.value(p, URIRef("http://www.w3.org/2000/01/rdf-schema#comment"))),
        "法的根拠": str(graph.value(p, ONSEN.legalBasis)),
    }


# --------------------------------------------------------------------------
# 含有成分別禁忌症（飲用）の算術評価
# --------------------------------------------------------------------------

Q_COMPONENT_RULES = (
    PREFIXES
    + """
SELECT ?rule ?ruleName ?prop ?threshold
WHERE {
    ?rule a onsen:ComponentContraindicationRule ;
          rdfs:label ?ruleName ;
          onsen:componentProperty ?prop ;
          onsen:thresholdMilligram ?threshold .
}
"""
)


def evaluate_drinking_contraindications(graph: Graph, source: str) -> dict[str, Any]:
    """含有成分別禁忌症（飲用）の限界飲用量を計算する。

    限界飲用量 = (閾値 mg / 温泉1kg中の成分量 A mg) × 1000 mL。
    飲用の1日総量はおよそ200〜500mLまでと定められているため、
    算出値が 500mL 以上になる場合は禁忌症の掲示を要しない。
    """
    uri = URIRef(source) if source.startswith("http") else None
    if uri is None:
        for candidate in graph.subjects(RDF.type, ONSEN.SpringSource):
            label = _label(graph, candidate)
            if label and source in label:
                uri = candidate
                break
    if uri is None:
        return {"error": f"源泉が見つからない: {source}"}

    results: list[dict[str, Any]] = []
    for row in graph.query(Q_COMPONENT_RULES):
        amount = graph.value(uri, row.prop)
        if amount is None:
            results.append(
                {
                    "ルール": str(row.ruleName),
                    "判定": "成分量が未確認のため計算不能",
                }
            )
            continue
        a = float(amount.toPython())
        if a <= 0:
            continue
        limit_ml = (float(row.threshold.toPython()) / a) * 1000
        conditions = sorted(
            filter(
                None,
                (
                    _label(graph, o)
                    for o in graph.objects(row.rule, ONSEN.hasDrinkingContraindication)
                ),
            )
        )
        results.append(
            {
                "ルール": str(row.ruleName),
                "成分量(mg/kg)": a,
                "限界飲用量(mL/日)": round(limit_ml, 1),
                "掲示が必要": limit_ml < 500,
                "飲用禁忌症": conditions,
            }
        )
    return {
        "源泉": _label(graph, uri),
        "評価結果": results,
        "法的根拠": "掲示基準 2.(1)③",
        "注記": (
            "算出された限界値が500mL以上の場合、1日の飲用量（およそ200〜500mLまで）を"
            "超えているため禁忌症を掲示することを要しない。"
        ),
    }


__all__ = [
    "colloquial_expressions",
    "paraphrase_table",
    "bathing_protocol",
    "describe_facility",
    "describe_spring_quality",
    "drinking_protocol",
    "evaluate_drinking_contraindications",
    "find_facilities_by_symptom",
    "general_indications",
    "list_facilities",
    "resolve_conditions",
    "resolve_facility",
]
