"""回答の検算レイヤ。

LLM が組み立てた回答文を、**ツールが実際に返した値**と照合する。
温泉爺のシステムプロンプトは「ツールが返した値だけを使う」「適応症に触れるときは掲示基準に
基づく区分であることを明示する」「法定と経験則を混同しない」と定めているが、
プロンプトは守られる保証がない。守られたかどうかをオントロジー側から機械的に検査する。

検出するのは4種類。

``unsourced_quantity``
    pH・温度・時間・容量などの**単位付きの数値**が、ツール戻り値にも相談文にも現れない。
``unsourced_term``
    オントロジーの語彙（施設名・源泉名・泉質名・適応症名・禁忌症名）を、
    ツールが返していないのに使っている。
``folk_expression``
    「美人の湯」「保温効果」のような法定の記述に無い通俗的な効能表現。
``missing_disclosure``
    適応症・禁忌症に触れながら掲示基準への言及がない（規則2違反）、または
    経験則由来の警告を伝えながら「経験則」と前置きしていない（規則5違反）。
``missing_tool_call``
    相談の種類から呼ぶべきツールを呼んでいない。巡浴を頼まれて ``plan_itinerary`` を
    呼ばずに一般論を述べる回答は、**値の照合では引っかからない**。検算は「言ったこと」しか
    見ないので、「言わなかったこと」は相談文から別に判定する。
``paraphrased_term``
    掲示基準の条文表記を要約した言い換え（「病気の活動期（特に熱のあるとき）」→「急性疾患」）。
    対応表はグラフ側の ``onsen:nonStandardParaphrase`` にある。意味は近いので誤りではないが、
    条文の表記ではないため掲示基準の引用として扱えない。

数値と語彙の検査は「ツール戻り値に無い」ことしか見ない。**意味の正しさは検査しない**。
言い換え（「病気の活動期」→「急性疾患」）は語彙の検査では素通りするので、
条文表記そのままの使用を強制したい場合は別の手段が必要である。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rdflib import Graph

from . import queries
from .namespaces import ONSEN

if TYPE_CHECKING:
    from .agent import ToolCallLog

#: 単位付き数値のパターン。単位を要求するのは、掲示基準の条番号（「2.(2)①」）や
#: 箇条書きの番号を数値として拾わないため。
_QUANTITY_PATTERNS: list[tuple[str, str]] = [
    (r"pH\s*(?:約|およそ)?([0-9][0-9,]*(?:\.[0-9]+)?)", "pH"),
    (r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:℃|度)", "温度"),
    (r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*分", "時間"),
    (r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:mL|ml|リットル|L)", "容量"),
    (r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:mg|g/kg|mg/kg)", "成分量"),
    (r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*回", "回数"),
    (r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*日", "日数"),
]

#: 法定の記述に無い通俗的な効能表現。値は「なぜ問題か」。
#: 掲示基準が使う語は「適応症」であって「効能」ではない。
FOLK_EXPRESSIONS: dict[str, str] = {
    "美人の湯": "俗称。掲示基準にも鉱泉分析法指針にも無い",
    "美肌": "掲示基準の適応症に無い表現",
    "保温効果": "掲示基準に無い。塩化物泉の適応症は条文で列挙されている",
    "保湿": "掲示基準の適応症に無い表現",
    "デトックス": "法定の記述に無い",
    "ピーリング": "法定の記述に無い",
    "血行促進": "掲示基準の適応症に無い表現",
    "新陳代謝": "掲示基準の適応症に無い表現",
    "免疫力": "掲示基準の適応症に無い表現",
    "若返り": "法定の記述に無い",
    "アンチエイジング": "法定の記述に無い",
    "特効": "医学的断定に当たる（規則3違反）",
    "万病": "医学的断定に当たる（規則3違反）",
}

#: 適応症・禁忌症に触れていると判定する語。
_INDICATION_WORDS = ("適応症", "禁忌症", "禁忌")

#: 掲示基準への言及と認める語。
_LEGAL_SOURCE_WORDS = ("掲示基準", "掲示の基準", "環境省", "温泉法")

#: 経験則の前置きと認める語。
_HEURISTIC_WORDS = ("経験則", "わしの勘", "法令ではない")

#: 相談文のキーワード → 呼ぶべきツールの対応は**グラフ側**にある（``onsen:ConsultIntent`` の個体）。
#: 「言わなかったこと」は照合では出てこない。巡浴の相談で ``plan_itinerary`` を呼ばずに
#: 一般論だけ述べる回答は、値の照合には引っかからないまま規則5に反する。
_EXPECTATION_QUERY = """
SELECT ?intent ?label ?severity ?detail WHERE {
    ?intent a onsen:ConsultIntent ;
            rdfs:label ?label ;
            onsen:findingSeverity ?severity ;
            rdfs:comment ?detail .
}
"""

_AREA_QUERY = """
SELECT DISTINCT ?label WHERE { ?s a onsen:OnsenArea ; rdfs:label ?label }
"""

_area_cache: dict[int, tuple[str, ...]] = {}
_expectation_cache: dict[int, tuple[ConsultExpectation, ...]] = {}

_VOCABULARY_CLASSES: tuple[tuple[Any, str, str], ...] = (
    # (クラス, 区分名, 裏を取るツール名)
    (ONSEN.Facility, "施設", "describe_facility"),
    (ONSEN.SpringSource, "源泉", "describe_facility"),
    (ONSEN.SpringQuality, "泉質", "describe_spring_quality"),
    (ONSEN.HealthCondition, "症状", "search_by_symptom"),
)

_VOCABULARY_QUERY = """
SELECT DISTINCT ?label WHERE {
    ?s a ?type .
    ?type rdfs:subClassOf* ?class .
    ?s rdfs:label ?label .
}
"""

#: 語彙照合の下限文字数。1〜2文字の語は偶然一致しやすい。
_MIN_TERM_LENGTH = 3

_vocabulary_cache: dict[int, dict[str, tuple[str, str]]] = {}


@dataclass(frozen=True)
class ConsultExpectation:
    """相談の意図と、それに対して呼ぶべきツール。グラフから読む。"""

    label: str
    keywords: tuple[str, ...]
    tools: tuple[str, ...]
    #: 呼ばれていなかったときに検算役が代わりに呼ぶツール
    repair_tool: str
    severity: str
    detail: str

    def matches(self, question: str) -> bool:
        return any(word in question for word in self.keywords)


@dataclass(frozen=True)
class Finding:
    """検算で見つかった1件。"""

    kind: str
    severity: str  # "error"（ツール戻り値と矛盾する） / "warning"（表現の逸脱）
    text: str
    detail: str
    #: 裏を取るためのツール呼び出し。``(ツール名, 引数)``。無ければ None。
    hint: tuple[str, dict[str, str]] | None = None

    def format(self) -> str:
        line = f"[{self.severity}] {self.kind}: {self.text} — {self.detail}"
        if self.hint is not None:
            name, arguments = self.hint
            line += f"（裏を取るなら {name}({json.dumps(arguments, ensure_ascii=False)})）"
        return line


def ontology_vocabulary(graph: Graph) -> dict[str, tuple[str, str]]:
    """検算に使う語彙を集める。``ラベル -> (区分名, 裏を取るツール名)``。"""
    key = id(graph)
    cached = _vocabulary_cache.get(key)
    if cached is not None:
        return cached
    vocabulary: dict[str, tuple[str, str]] = {}
    for cls, category, tool in _VOCABULARY_CLASSES:
        for (label,) in graph.query(_VOCABULARY_QUERY, initBindings={"class": cls}):
            text = str(label).strip()
            if len(text) >= _MIN_TERM_LENGTH:
                vocabulary.setdefault(text, (category, tool))
    _vocabulary_cache[key] = vocabulary
    return vocabulary


def _numbers(text: str) -> set[float]:
    """文字列中の数値をすべて集める。桁区切りのカンマは外す。"""
    found: set[float] = set()
    for token in re.findall(r"[0-9][0-9,]*(?:\.[0-9]+)?", text):
        try:
            found.add(float(token.replace(",", "")))
        except ValueError:
            continue
    return found


def _serialize(tool_calls: list[ToolCallLog]) -> str:
    parts = []
    for call in tool_calls:
        parts.append(json.dumps(call.input, ensure_ascii=False, default=str))
        parts.append(json.dumps(call.output, ensure_ascii=False, default=str))
    return "\n".join(parts)


def _heuristic_warning_returned(tool_calls: list[ToolCallLog]) -> bool:
    """ツールが severity=heuristic の警告を返したか。"""
    for call in tool_calls:
        blob = json.dumps(call.output, ensure_ascii=False, default=str)
        if '"heuristic"' in blob:
            return True
    return False


#: 範囲表記の前半は単位に隣接しないため取り落とす。「10〜15分」を「10分〜15分」に正規化する。
_RANGE_PATTERN = re.compile(
    r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*[〜～~ー−–—]\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"(℃|度|分|mL|ml|リットル|L|mg|g/kg|mg/kg|回|日)"
)


def _expand_ranges(text: str) -> str:
    """「3〜10分」を「3分〜10分」に書き換える。"""
    return _RANGE_PATTERN.sub(r"\1\3〜\2\3", text)


def _check_quantities(answer: str, allowed: set[float]) -> list[Finding]:
    findings = []
    seen: set[tuple[str, float]] = set()
    expanded = _expand_ranges(answer)
    for pattern, kind in _QUANTITY_PATTERNS:
        for raw in re.findall(pattern, expanded):
            try:
                value = float(raw.replace(",", ""))
            except ValueError:
                continue
            if value in allowed or (kind, value) in seen:
                continue
            seen.add((kind, value))
            findings.append(
                Finding(
                    kind="unsourced_quantity",
                    severity="error",
                    text=f"{kind} {raw}",
                    detail="ツール戻り値にも相談文にも現れない数値",
                )
            )
    return findings


def _check_terms(
    answer: str, allowed_text: str, vocabulary: dict[str, tuple[str, str]]
) -> list[Finding]:
    """ツールが返していないオントロジー語彙の使用を検出する。

    指摘には**裏を取るためのツール呼び出し**を添える。削らせるより、引かせたほうがよい。
    """
    findings = []
    for term in sorted(vocabulary, key=len, reverse=True):
        if term in answer and term not in allowed_text:
            category, tool = vocabulary[term]
            argument = "keyword" if tool == "search_by_symptom" else "name"
            findings.append(
                Finding(
                    kind="unsourced_term",
                    severity="error",
                    text=term,
                    detail=f"オントロジーの{category}名だが、ツールはこの語を返していない",
                    hint=(tool, {argument: term}),
                )
            )
    return findings


def _check_folk_expressions(answer: str, tool_text: str) -> list[Finding]:
    """通俗表現を検出する。ツール戻り値に同じ語があれば出典ありとして通す。

    「保温」は掲示基準の浴用プロトコル（入浴後の保温及び30分程度の安静）に、
    「保湿・美肌効果」は四万温泉 積善館の公式記述に現れる。オントロジーが返した語なら
    出典があるので指摘しない。

    照合先はツール戻り値だけで、相談文は含めない。**利用者が「デトックス」と言ったことは、
    温泉爺がその語を使ってよい根拠にはならない**。「掲示基準には無い」と否定するために
    引用した場合も指摘は出るが、severity は warning なので読んで判断すればよい。
    """
    return [
        Finding(kind="folk_expression", severity="warning", text=expression, detail=reason)
        for expression, reason in FOLK_EXPRESSIONS.items()
        if expression in answer and expression not in tool_text
    ]


def _check_disclosure(answer: str, tool_calls: list[ToolCallLog]) -> list[Finding]:
    findings = []
    if any(word in answer for word in _INDICATION_WORDS) and not any(
        word in answer for word in _LEGAL_SOURCE_WORDS
    ):
        findings.append(
            Finding(
                kind="missing_disclosure",
                severity="warning",
                text="適応症・禁忌症",
                detail="掲示基準に基づく区分であることを明示していない（規則2）",
            )
        )
    if _heuristic_warning_returned(tool_calls) and not any(
        word in answer for word in _HEURISTIC_WORDS
    ):
        findings.append(
            Finding(
                kind="missing_disclosure",
                severity="warning",
                text="severity=heuristic の警告",
                detail="経験則であることを前置きしていない（規則5）",
            )
        )
    return findings


def consult_expectations(graph: Graph) -> tuple[ConsultExpectation, ...]:
    """相談の意図と呼ぶべきツールの対応をグラフから読む。

    以前は Python の定数表だった。語彙・口語表現・言い換えをグラフに置いておきながら
    ここだけコードに残っているのは筋が通らないので、``onsen:ConsultIntent`` の個体に移した。
    """
    key = id(graph)
    cached = _expectation_cache.get(key)
    if cached is not None:
        return cached
    prefixes = (
        "PREFIX onsen: <https://example.org/onsen#>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    )
    expectations = []
    for intent, label, severity, detail in graph.query(prefixes + _EXPECTATION_QUERY):
        keywords = tuple(sorted(str(w) for w in graph.objects(intent, ONSEN.intentKeyword)))
        tools = tuple(str(t) for t in graph.objects(intent, ONSEN.expectedTool))
        if not keywords or not tools:
            continue
        repair = graph.value(intent, ONSEN.repairTool)
        expectations.append(
            ConsultExpectation(
                label=str(label),
                keywords=keywords,
                tools=tuple(sorted(tools)),
                repair_tool=str(repair) if repair is not None else sorted(tools)[0],
                severity=str(severity),
                detail=str(detail),
            )
        )
    result = tuple(sorted(expectations, key=lambda e: e.label))
    _expectation_cache[key] = result
    return result


def _area_labels(graph: Graph) -> tuple[str, ...]:
    """温泉地のラベル。巡浴の裏取りに `area` を渡すために使う。"""
    key = id(graph)
    cached = _area_cache.get(key)
    if cached is None:
        prefixes = "PREFIX onsen: <https://example.org/onsen#>\n"
        cached = tuple(str(label).strip() for (label,) in graph.query(prefixes + _AREA_QUERY))
        _area_cache[key] = cached
    return cached


def _check_expected_tools(
    question: str, tool_calls: list[ToolCallLog], graph: Graph | None
) -> list[Finding]:
    """相談の種類から、呼ぶべきだったツールを呼んでいないことを検出する。"""
    if graph is None:
        return []
    called = {call.name for call in tool_calls}
    findings = []
    for expectation in consult_expectations(graph):
        if not expectation.matches(question) or called & set(expectation.tools):
            continue
        hint: tuple[str, dict[str, str]] = (expectation.repair_tool, {})
        if expectation.repair_tool == "plan_itinerary":
            area = next((name for name in _area_labels(graph) if name[:2] in question), "")
            if area:
                hint = ("plan_itinerary", {"area": area})
        findings.append(
            Finding(
                kind="missing_tool_call",
                severity=expectation.severity,
                text=f"{expectation.label}に {expectation.repair_tool} を呼んでいない",
                detail=expectation.detail,
                hint=hint,
            )
        )
    return findings


def _check_paraphrases(answer: str, tool_text: str, graph: Graph | None) -> list[Finding]:
    """条文表記の言い換えを検出する。

    第3話・第4話の時点では「病気の活動期（特に熱のあるとき）」→「急性疾患」のような
    言い換えを素通りさせていた。語彙の照合は「オントロジーに無い語」を探すので、
    **オントロジーに無い言い換えは引っかからない**という構造的な穴だった。
    そこで言い換えの側をグラフに宣言し（``onsen:nonStandardParaphrase``）、対応表として使う。

    条文表記そのものが回答に併記されている場合と、ツール戻り値に言い換え語が
    含まれている場合は指摘しない。
    """
    if graph is None:
        return []
    findings = []
    for paraphrase, official in sorted(queries.paraphrase_table(graph).items()):
        if paraphrase not in answer or paraphrase in tool_text:
            continue
        if official in answer:
            continue
        findings.append(
            Finding(
                kind="paraphrased_term",
                severity="warning",
                text=paraphrase,
                detail=f"掲示基準の条文表記は「{official}」。要約した言い換えは条文の引用にならない",
            )
        )
    return findings


def verify_answer(
    answer: str,
    tool_calls: list[ToolCallLog],
    *,
    graph: Graph | None = None,
    question: str = "",
) -> list[Finding]:
    """回答文をツール戻り値と照合する。問題が無ければ空リストを返す。"""
    tool_text = _serialize(tool_calls)
    allowed_text = f"{tool_text}\n{question}"
    allowed_numbers = _numbers(allowed_text)

    findings = _check_quantities(answer, allowed_numbers)
    if graph is not None:
        findings += _check_terms(answer, allowed_text, ontology_vocabulary(graph))
    findings += _check_folk_expressions(answer, tool_text)
    findings += _check_paraphrases(answer, tool_text, graph)
    findings += _check_disclosure(answer, tool_calls)
    findings += _check_expected_tools(question, tool_calls, graph)
    return findings


def format_findings(findings: list[Finding]) -> str:
    """検算結果を人が読める形にする。"""
    if not findings:
        return "検算: 指摘なし"
    lines = [f"検算: {len(findings)}件の指摘"]
    lines += [f"  {finding.format()}" for finding in findings]
    return "\n".join(lines)


def revision_request(findings: list[Finding]) -> str:
    """検算結果を LLM に差し戻すための指示文。"""
    lines = [
        "わしの検算役がおぬしの回答を照合したところ、次の点が引っかかった。",
        "指摘された箇所は、ツールで裏を取るか、取れなければ回答から削ること。",
        "削った箇所は「そこは現地の掲示を見んと分からんのう」と正直に書く。",
        "",
    ]
    lines += [f"- {finding.format()}" for finding in findings]
    lines += ["", "以上を直した回答を、もう一度はじめから書き直してくれ。"]
    return "\n".join(lines)


def repair_hints(findings: list[Finding]) -> list[tuple[str, dict[str, str]]]:
    """指摘から裏取りのツール呼び出しを取り出す（重複を除く）。"""
    hints: list[tuple[str, dict[str, str]]] = []
    for finding in findings:
        if finding.hint is not None and finding.hint not in hints:
            hints.append(finding.hint)
    return hints


def carry_over_note(findings: list[Finding]) -> str:
    """前のターンまでの指摘を、次のターンに持ち越す注意文にする。

    同じ通俗表現を繰り返すモデルへの対処。指摘そのものではなく「もう言うな」を渡す。
    """
    if not findings:
        return ""
    lines = ["【検算役からの申し送り】これまでの回答で次の点を指摘済みじゃ。繰り返さぬこと。"]
    seen: set[tuple[str, str]] = set()
    for finding in findings:
        key = (finding.kind, finding.text)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {finding.kind}: {finding.text}")
    return "\n".join(lines)


def summarize(findings: list[Finding]) -> dict[str, Any]:
    """種別ごとの件数。テストと記事の集計に使う。"""
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1
    return {"件数": len(findings), "種別別": counts}


__all__ = [
    "FOLK_EXPRESSIONS",
    "ConsultExpectation",
    "Finding",
    "consult_expectations",
    "carry_over_note",
    "format_findings",
    "ontology_vocabulary",
    "repair_hints",
    "revision_request",
    "summarize",
    "verify_answer",
]
