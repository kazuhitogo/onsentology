"""巡浴プラン（俗に「はしご湯」）の生成と検証。

「はしご湯」は一次情報に出現しない俗語であり、公的な定義や規制は存在しない。
一方、制約の根拠は掲示基準 2.(2)① に明確に存在する。したがってここでは
**巡浴を「法定の浴用プロトコルを満たす訪問順序を決める問題」として定式化**する。

検証結果の警告は必ず ``severity`` を持つ。

- ``legal``     : 掲示基準の条文に根拠がある（入浴回数・入浴時間・安静時間・高温浴・湯あたり）
- ``heuristic`` : 本プロジェクト独自の経験的ルール（刺激の強い泉の連続、仕上げ湯）

この2つを混ぜないことが、この種のツールで最も重要な設計上の約束である。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS

from .namespaces import ONSEN
from .queries import _label, _lit, resolve_facility
from .reasoning import PREFIXES

#: 掲示基準 2.(2)①エ(ア) が想定する入浴後の安静時間（分）
DEFAULT_GAP_MINUTES = 30


@dataclass
class Step:
    """巡浴プランの1ステップ。"""

    facility: URIRef
    name: str
    area: str | None
    sources: list[str] = field(default_factory=list)
    qualities: list[str] = field(default_factory=list)
    is_irritating: bool = False
    is_finishing: bool = False
    is_alkaline_simple: bool = False
    max_source_temperature: float | None = None
    minutes: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "施設名": self.name,
            "温泉地": self.area,
            "源泉": self.sources,
            "泉質": self.qualities,
            "刺激の強い泉質を含む": self.is_irritating,
            "仕上げ湯候補": self.is_finishing,
            "アルカリ性単純温泉": self.is_alkaline_simple,
            "源泉温度(最高)": self.max_source_temperature,
            "入浴時間(分)": self.minutes,
            "uri": str(self.facility),
        }


Q_FACILITY_PROFILE = (
    PREFIXES
    + """
SELECT DISTINCT ?facility ?facName ?area ?srcName ?qName ?irritating ?temp ?alkalineSimple
WHERE {
    ?facility a onsen:Facility ; rdfs:label ?facName ;
              onsen:hasSpringSource ?source .
    ?source rdfs:label ?srcName .
    OPTIONAL { ?facility onsen:locatedInArea ?areaUri . ?areaUri rdfs:label ?area }
    OPTIONAL { ?source onsen:hasQuality ?quality .
               ?quality rdfs:label ?qName .
               OPTIONAL { ?quality onsen:isSkinIrritating ?irritating } }
    OPTIONAL { ?source onsen:sourceTemperature ?temp }
    OPTIONAL { ?source onsen:isAlkalineSimpleSpring ?alkalineSimple }
}
"""
)

Q_FINISHING_QUALITIES = (
    PREFIXES
    + """
SELECT DISTINCT ?quality ?qName
WHERE {
    ?quality onsen:recommendedAfter ?irritatingQuality ; rdfs:label ?qName .
    ?irritatingQuality onsen:isSkinIrritating true .
}
"""
)


def _finishing_quality_labels(graph: Graph) -> set[str]:
    """刺激の強い泉質のあとの仕上げ湯として推奨される泉質のラベル集合。"""
    return {str(row.qName) for row in graph.query(Q_FINISHING_QUALITIES)}


def load_steps(graph: Graph, minutes: int = 10) -> dict[URIRef, Step]:
    """全施設を :class:`Step` として読み出す（プラン生成の材料）。"""
    finishing = _finishing_quality_labels(graph)
    steps: dict[URIRef, Step] = {}
    for row in graph.query(Q_FACILITY_PROFILE):
        step = steps.get(row.facility)
        if step is None:
            step = Step(
                facility=row.facility,
                name=str(row.facName),
                area=_lit(row.area),
                minutes=minutes,
            )
            steps[row.facility] = step
        if row.srcName and str(row.srcName) not in step.sources:
            step.sources.append(str(row.srcName))
        if row.qName and str(row.qName) not in step.qualities:
            step.qualities.append(str(row.qName))
        if row.irritating is not None and bool(row.irritating.toPython()):
            step.is_irritating = True
        if row.alkalineSimple is not None and bool(row.alkalineSimple.toPython()):
            step.is_alkaline_simple = True
        if row.temp is not None:
            temp = float(row.temp.toPython())
            if step.max_source_temperature is None or temp > step.max_source_temperature:
                step.max_source_temperature = temp
    for step in steps.values():
        step.is_finishing = any(q in finishing for q in step.qualities)
    return steps


def _protocol(graph: Graph) -> dict[str, Any]:
    p = ONSEN.StandardBathingProtocol
    return {
        "max_baths_initial": int(graph.value(p, ONSEN.maxBathsPerDayInitial).toPython()),
        "max_baths_adapted": int(graph.value(p, ONSEN.maxBathsPerDayAdapted).toPython()),
        "max_minutes_initial": int(graph.value(p, ONSEN.maxMinutesPerBathInitial).toPython()),
        "max_minutes_adapted": int(graph.value(p, ONSEN.maxMinutesPerBathAdapted).toPython()),
        "rest_minutes": int(graph.value(p, ONSEN.restMinutesAfterBath).toPython()),
        "high_temp": float(graph.value(p, ONSEN.highTemperatureCautionThreshold).toPython()),
        "onset_days_min": int(graph.value(p, ONSEN.onsetDaysMin).toPython()),
        "onset_days_max": int(graph.value(p, ONSEN.onsetDaysMax).toPython()),
        "basis": str(graph.value(p, ONSEN.legalBasis)),
    }


def _warning(severity: str, message: str, basis: str) -> dict[str, str]:
    return {"severity": severity, "message": message, "basis": basis}


def validate_itinerary(
    graph: Graph,
    steps: list[Step],
    *,
    adapted: bool = False,
    gap_minutes: int = DEFAULT_GAP_MINUTES,
    consecutive_days: int = 1,
    high_temperature_caution: bool = False,
) -> list[dict[str, str]]:
    """巡浴プランを法定プロトコルと独自ヒューリスティックで検証する。

    :param adapted: 温泉に慣れている（掲示基準のいう「慣れてきたら」）かどうか。
        回数上限が 2→3 回、時間上限が 10→20 分に緩む。
    :param gap_minutes: 入浴と入浴の間隔（分）。
    :param consecutive_days: 連続して温泉療養する日数。湯あたりの発現時期の判定に使う。
    :param high_temperature_caution: 高齢者・高血圧症・心臓病・脳卒中経験者に該当するか。
        該当する場合、42℃以上の高温浴に対して警告を出す。
    """
    p = _protocol(graph)
    warnings: list[dict[str, str]] = []

    max_baths = p["max_baths_adapted"] if adapted else p["max_baths_initial"]
    max_minutes = p["max_minutes_adapted"] if adapted else p["max_minutes_initial"]

    # --- 法定プロトコル -------------------------------------------------
    if len(steps) > max_baths:
        warnings.append(
            _warning(
                "legal",
                f"1日{len(steps)}回の入浴は上限を超えている。"
                f"掲示基準は入浴開始後数日間は1日1〜2回、慣れてきたら2〜3回までとしている"
                f"（今回の上限: {max_baths}回）。",
                p["basis"] + " イ(ウ) 入浴回数",
            )
        )

    for step in steps:
        if step.minutes > max_minutes:
            warnings.append(
                _warning(
                    "legal",
                    f"{step.name} の入浴時間 {step.minutes}分 は上限を超えている。"
                    f"掲示基準は1回当たり初めは3〜10分程度、慣れてきたら15〜20分程度までとしている"
                    f"（今回の上限: {max_minutes}分）。",
                    p["basis"] + " イ(エ) 入浴時間",
                )
            )

    if gap_minutes < p["rest_minutes"] and len(steps) > 1:
        warnings.append(
            _warning(
                "legal",
                f"入浴の間隔が{gap_minutes}分しかない。掲示基準は入浴後に保温と"
                f"{p['rest_minutes']}分程度の安静を心がけることとしている。",
                p["basis"] + " エ(ア) 入浴後の注意",
            )
        )

    if high_temperature_caution:
        for step in steps:
            temp = step.max_source_temperature
            if temp is not None and temp >= p["high_temp"]:
                warnings.append(
                    _warning(
                        "legal",
                        f"{step.name} の源泉温度は{temp}℃。高齢者、高血圧症若しくは心臓病の人、"
                        f"脳卒中を経験した人は{p['high_temp']}℃以上の高温浴は避けることとされている"
                        f"（浴槽温度は源泉温度と異なる場合があるため現地の掲示を確認すること）。",
                        p["basis"] + " イ(ア) 入浴温度",
                    )
                )

    if len(steps) >= 2:
        warnings.append(
            _warning(
                "legal",
                f"入浴のたびにコップ一杯程度の水分補給を行うこと（今回は{len(steps) + 1}回分が目安）。"
                "脱水症状は入浴回数に比例して累積する。",
                p["basis"] + " ア(カ)・エ(イ) 水分補給",
            )
        )

    if consecutive_days >= p["onset_days_min"]:
        warnings.append(
            _warning(
                "legal",
                f"温泉療養開始後おおむね{p['onset_days_min']}日〜{p['onset_days_max']}日前後に、"
                "気分不快・不眠・消化器症状等の湯あたり症状や皮膚炎が現れることがある。"
                f"{consecutive_days}日目はその時期に入っている。"
                "症状が現れている間は入浴を中止するか回数を減らし、回復を待つこと。",
                p["basis"] + " オ 湯あたり",
            )
        )

    # --- 独自ヒューリスティック ------------------------------------------
    for prev, curr in zip(steps, steps[1:], strict=False):
        if prev.is_irritating and curr.is_irritating:
            warnings.append(
                _warning(
                    "heuristic",
                    f"{prev.name} → {curr.name} は刺激の強い泉質（酸性泉・硫黄泉）の連続になっている。"
                    "掲示基準は肌の弱い人はこれらの泉質のあと温水で洗い流した方がよいとしており、"
                    "皮膚への負担が累積しやすい。順序を入れ替えるか、間に別の泉質を挟むとよい。",
                    "heuristic: 掲示基準 2.(2)①エ(ア) の「刺激の強い泉質」記述から導出",
                )
            )

    if steps and steps[-1].is_irritating:
        finishing = sorted(_finishing_quality_labels(graph))
        warnings.append(
            _warning(
                "heuristic",
                f"最後の{steps[-1].name}が刺激の強い泉質で終わっている。"
                f"仕上げの湯として{'・'.join(finishing)}を最後に置く構成が伝統的に好まれる"
                "（四万温泉が「草津の仕上げ湯」と呼ばれてきた例がある）。",
                "heuristic: 積善館（四万温泉）の記述 + 塩化物泉・硫酸塩泉・炭酸水素塩泉の適応症「皮膚乾燥症」",
            )
        )

    return warnings


def plan_itinerary(
    graph: Graph,
    *,
    area: str | None = None,
    facilities: list[str] | None = None,
    max_baths: int | None = None,
    adapted: bool = False,
    minutes: int = 10,
    gap_minutes: int = DEFAULT_GAP_MINUTES,
    consecutive_days: int = 1,
    high_temperature_caution: bool = False,
) -> dict[str, Any]:
    """巡浴プランを生成して検証する。

    並べ替えの方針は「刺激の強い泉を先、仕上げ湯を後」。これは法令ではなく
    ``onsen:recommendedAfter``（独自ヒューリスティック）に基づく順序である。

    :param area: 温泉地名（部分一致）で候補を絞る。
    :param facilities: 施設名を明示的に指定する（この順序は尊重せず並べ替える）。
    """
    all_steps = load_steps(graph, minutes=minutes)

    candidates: list[Step]
    if facilities:
        candidates = []
        for name in facilities:
            uri = resolve_facility(graph, name)
            if uri is not None and uri in all_steps:
                candidates.append(all_steps[uri])
    elif area:
        candidates = [s for s in all_steps.values() if s.area and area in s.area]
    else:
        candidates = list(all_steps.values())

    if not candidates:
        return {
            "error": "候補となる施設が見つからない",
            "指定": {"温泉地": area, "施設": facilities},
        }

    def order_key(step: Step) -> tuple[int, str]:
        if step.is_irritating:
            rank = 0
        elif step.is_alkaline_simple:
            rank = 3
        elif step.is_finishing:
            rank = 2
        else:
            rank = 1
        return (rank, step.name)

    ordered = sorted(candidates, key=order_key)

    protocol = _protocol(graph)
    limit = max_baths or (
        protocol["max_baths_adapted"] if adapted else protocol["max_baths_initial"]
    )
    if len(ordered) > limit:
        # 上限まで絞る。先頭（刺激の強い湯）と末尾（仕上げ湯）を残すのが狙いなので、
        # 両端から交互に採る。
        picked: list[Step] = []
        head, tail = 0, len(ordered) - 1
        while len(picked) < limit and head <= tail:
            if len(picked) % 2 == 0:
                picked.append(ordered[head])
                head += 1
            else:
                picked.append(ordered[tail])
                tail -= 1
        selected = sorted(picked, key=order_key)
    else:
        selected = ordered

    warnings = validate_itinerary(
        graph,
        selected,
        adapted=adapted,
        gap_minutes=gap_minutes,
        consecutive_days=consecutive_days,
        high_temperature_caution=high_temperature_caution,
    )

    return {
        "プラン": [{"順序": i + 1, **step.to_dict()} for i, step in enumerate(selected)],
        "検討した候補数": len(candidates),
        "1日入浴回数の上限": limit,
        "入浴間隔(分)": gap_minutes,
        "警告": warnings,
        "並べ替えの根拠": (
            "刺激の強い泉質（酸性泉・硫黄泉）を先に、仕上げ湯（塩化物泉・硫酸塩泉・"
            "炭酸水素塩泉・アルカリ性単純温泉）を後に置いている。これは法令ではなく "
            "onsen:recommendedAfter という本プロジェクト独自のヒューリスティックである。"
        ),
        "注記": (
            "掲示基準は温泉療養について、十分な効用を得るには通常2〜3週間が適当であり、"
            "適応症でも病期や状態によっては悪化する場合があるため専門的知識を有する医師の"
            "指示・指導のもとに行うことが望ましいとしている。これは医療的助言ではない。"
        ),
    }


def describe_itinerary(graph: Graph, facility_names: list[str], **kwargs: Any) -> dict[str, Any]:
    """利用者が指定した順序をそのまま検証する（並べ替えない）。"""
    steps: list[Step] = []
    all_steps = load_steps(graph, minutes=kwargs.pop("minutes", 10))
    unknown: list[str] = []
    for name in facility_names:
        uri = resolve_facility(graph, name)
        if uri is None or uri not in all_steps:
            unknown.append(name)
            continue
        steps.append(all_steps[uri])
    result = {
        "指定した順序": [{"順序": i + 1, **s.to_dict()} for i, s in enumerate(steps)],
        "警告": validate_itinerary(graph, steps, **kwargs),
    }
    if unknown:
        result["見つからなかった施設"] = unknown
    return result


def facility_label(graph: Graph, uri: URIRef) -> str | None:
    """デバッグ用のラベル取得。"""
    value = graph.value(uri, RDFS.label)
    return str(value) if value else _label(graph, uri)


__all__ = [
    "DEFAULT_GAP_MINUTES",
    "Step",
    "describe_itinerary",
    "facility_label",
    "load_steps",
    "plan_itinerary",
    "validate_itinerary",
]
