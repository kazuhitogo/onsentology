"""「揃えた生ドキュメント」の生成（比較条件 F）。

条件 D（Web から丸ごと取得した生テキスト）が負けた原因は、**情報が書かれていないこと**
ではなく **記述が揃っていないこと**だった（本文の発見1）。ならば「同じ情報を同じ形に
揃えた文書」を置けば、生テキスト検索でも届くのか。それを測るための条件が F である。

つくり方の縛り
    グラフに**明示的に書いてある事実だけ**を Markdown にする。推論で導いた値
    （``isUnmodifiedSupply``、``isSkinIrritating``、液性区分などの分類）と、
    法令に根拠のない独自ヒューリスティック（``recommendedAfter`` など）は**入れない**。
    入れると「答えを文書に書いておいた」ことになり、比較にならない。

    したがって施設の文書は、温泉法施行規則第10条第2項が掲示を義務づけている
    5類型それぞれについて「実施の有無とその理由」を必ず並べる。**書いていないのではなく
    「実施していない」と書く**のがこの条件の要点である。未確認の値は
    ``onsen:dataStatus`` の記述をそのまま載せ、「未公表である」ことを文書に書く。

    形式も揃える。同じ見出し、同じ順序、同じ単位。**人間向けの散文ではなく、
    横断して数え上げられる形**にする。

出典
    各文書の front matter に、その施設・泉質の出典URL（``dcterms:source``）を入れる。
    条件 D と同じく、出典を語れる状態を保つ。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rdflib import Graph, URIRef
from rdflib.namespace import DCTERMS, RDF, RDFS

from .graph import load_graph
from .namespaces import ONSEN

#: 生成先の既定ディレクトリ。git 管理外。
DEFAULT_ALIGNED_DIR = Path("corpus-aligned")

#: 掲示義務のある湯使い5類型。**順序を固定する**（横断して数えるため）。
_TREATMENT_ORDER: tuple[tuple[Any, str], ...] = (
    (ONSEN.AddingWater, "加水"),
    (ONSEN.Heating, "加温"),
    (ONSEN.Recirculation, "循環（ろ過含む）"),
    (ONSEN.Disinfection, "消毒処理"),
    (ONSEN.BathAdditive, "入浴剤添加"),
)


def _label(graph: Graph, uri: Any) -> str:
    value = graph.value(uri, RDFS.label)
    return str(value) if value is not None else str(uri).split("#")[-1].split("/")[-1]


#: 掲示用泉質10種。クラス階層を自分でたどる（推論器に依存しない）。
_QUALITY_QUERY = """
PREFIX onsen: <https://example.org/onsen#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?quality WHERE {
    ?quality a ?type .
    ?type rdfs:subClassOf* onsen:SpringQuality .
    ?quality rdfs:label ?label .
}
"""


def _display_qualities(graph: Graph) -> list[URIRef]:
    return sorted((row[0] for row in graph.query(_QUALITY_QUERY)), key=str)


def _front_matter(title: str, sources: list[str], retrieved_at: str) -> str:
    return (
        "---\n"
        f"source_url: {sources[0] if sources else '（出典未記録）'}\n"
        f"retrieved_at: {retrieved_at}\n"
        "content_type: text/markdown\n"
        f"title: {title}\n"
        "---\n\n"
    )


def render_facility(graph: Graph, facility: URIRef, retrieved_at: str) -> tuple[str, str]:
    """施設1件の文書。**5類型すべての実施の有無**を同じ形で並べる。"""
    name = _label(graph, facility)
    area = graph.value(facility, ONSEN.locatedInArea)
    sources = [str(s) for s in graph.objects(facility, DCTERMS.source)]
    lines = [f"# {name}", ""]
    lines.append(f"- 温泉地: {_label(graph, area) if area else '（記録なし）'}")
    address = graph.value(facility, ONSEN.address)
    if address:
        lines.append(f"- 所在地: {address}")
    claims = graph.value(facility, ONSEN.claimsKakenagashi)
    if claims is not None:
        label = "「源泉かけ流し」を自主表示している" if bool(claims) else "自主表示なし"
        lines.append(f"- かけ流しの自称: {label}（法令上の定義はない自主表示）")
    drinking = graph.value(facility, ONSEN.drinkingPermitted)
    if drinking is not None:
        lines.append(f"- 飲用の許可: {'あり' if bool(drinking) else 'なし'}")

    lines += ["", "## 源泉", ""]
    for source in sorted(graph.objects(facility, ONSEN.hasSpringSource), key=str):
        lines.append(f"### {_label(graph, source)}")
        displayed = graph.value(source, ONSEN.displayedQualityName)
        lines.append(f"- 掲示泉質名: {displayed if displayed else '未公表（確認できていない）'}")
        for prop, caption, unit in (
            (ONSEN.pH, "pH", ""),
            (ONSEN.springTemperature, "源泉温度", "℃"),
            (ONSEN.dissolvedSubstanceTotal, "溶存物質総量", "mg/kg"),
        ):
            value = graph.value(source, prop)
            lines.append(
                f"- {caption}: {value}{unit}" if value is not None
                else f"- {caption}: 未公表（この源泉について公表された値を確認できていない）"
            )
        status = graph.value(source, ONSEN.dataStatus)
        if status:
            lines.append(f"- データの状態: {' '.join(str(status).split())}")
        for src in sorted(str(s) for s in graph.objects(source, DCTERMS.source)):
            lines.append(f"- 出典: {src}")
        lines.append("")

    lines += ["## 湯使い（温泉法施行規則第10条第2項の掲示事項）", ""]
    declared: dict[Any, tuple[bool, str]] = {}
    for declaration in graph.objects(facility, ONSEN.declaresTreatment):
        kind = graph.value(declaration, ONSEN.treatmentType)
        applied = graph.value(declaration, ONSEN.isApplied)
        reason = graph.value(declaration, ONSEN.reason)
        if kind is not None and applied is not None:
            declared[kind] = (bool(applied), str(reason) if reason else "")
    for kind, caption in _TREATMENT_ORDER:
        if kind in declared:
            applied, reason = declared[kind]
            state = "実施している" if applied else "実施していない"
            lines.append(f"- {caption}: {state}。理由: {reason or '（理由の記載なし）'}")
        else:
            lines.append(f"- {caption}: 掲示を確認できていない（実施の有無は不明）")
    lines.append("")
    for src in sources:
        lines.append(f"- 出典: {src}")
    lines.append("")
    return f"facility-{str(facility).split('/')[-1]}", _front_matter(
        name, sources, retrieved_at
    ) + "\n".join(lines)


def render_quality(graph: Graph, quality: URIRef, retrieved_at: str) -> tuple[str, str]:
    """掲示用泉質1件の文書。判定基準と適応症・禁忌症を条文の表記で並べる。"""
    name = _label(graph, quality)
    sources = [str(s) for s in graph.objects(quality, DCTERMS.source)]
    basis = graph.value(quality, ONSEN.legalBasis)
    if not sources:
        # 法定知識の個体は dcterms:source を持たず onsen:legalBasis で根拠を示している。
        # 条件 D と同じく出典を語れる状態にするため、根拠にあたる通知の URL を補う。
        sources = [
            "https://www.env.go.jp/nature/onsen/pdf/2-5_p_14.pdf"
            if basis and "指針" in str(basis)
            else "https://www.env.go.jp/nature/onsen/pdf/2-5_p_11.pdf"
        ]
    lines = [f"# 掲示用泉質: {name}", ""]
    component = graph.value(quality, ONSEN.criterionProperty)
    minimum = graph.value(quality, ONSEN.criterionMinValue)
    maximum = graph.value(quality, ONSEN.criterionMaxValue)
    lines.append("## 判定基準（鉱泉分析法指針 第1-3表）")
    lines.append("")
    if component is not None:
        unit = graph.value(component, ONSEN.unit)
        caption = _label(graph, component)
        lines.append(f"- 対象成分: {caption}{f'（単位 {unit}）' if unit else ''}")
    if minimum is not None:
        lines.append(f"- 下限値: {minimum}（この値以上で該当する）")
    if maximum is not None:
        lines.append(f"- 上限値: {maximum}")
    if component is None and minimum is None:
        lines.append("- 成分による下限値の定めはない")
    if basis:
        lines.append(f"- 法的根拠: {basis}")
    comment = graph.value(quality, RDFS.comment)
    if comment:
        lines += ["", " ".join(str(comment).split())]

    for prop, caption in (
        (ONSEN.hasBathingIndication, "浴用適応症（掲示基準 3.(1)）"),
        (ONSEN.hasDrinkingIndication, "飲用適応症（掲示基準 3.(1)）"),
        (ONSEN.hasBathingContraindication, "泉質別浴用禁忌症（掲示基準 2.(1)②）"),
    ):
        values = sorted(_label(graph, o) for o in graph.objects(quality, prop))
        lines += ["", f"## {caption}", ""]
        if values:
            lines += [f"- {value}" for value in values]
        else:
            lines.append("- 定めはない（掲示基準に該当する記載がない）")
    lines.append("")
    for src in sources:
        lines.append(f"- 出典: {src}")
    lines.append("")
    return f"quality-{str(quality).split('#')[-1]}", _front_matter(
        f"掲示用泉質 {name}", sources, retrieved_at
    ) + "\n".join(lines)


def render_index(graph: Graph, retrieved_at: str) -> tuple[str, str]:
    """収録範囲を書いた文書。**母集合を文書に書く**のがこの条件の要点のひとつ。"""
    facilities = sorted(
        (_label(graph, f), _label(graph, graph.value(f, ONSEN.locatedInArea)))
        for f in graph.subjects(RDF.type, ONSEN.Facility)
    )
    lines = [
        "# 収録範囲（この文書集がカバーする施設の全体）",
        "",
        f"この文書集は次の{len(facilities)}施設についてのみ記述している。"
        "ここに無い温泉地・施設の値は確認できていない。",
        "",
    ]
    lines += [f"- {name}（{area}）" for name, area in facilities]
    lines += [
        "",
        "各施設の文書には、温泉法施行規則第10条第2項が掲示を義務づけている5類型"
        "（加水・加温・循環（ろ過含む）・消毒処理・入浴剤添加）について、"
        "実施の有無と理由を必ず記載している。**「記載がない」ことと「実施していない」ことは"
        "区別して書いてある。**",
        "",
    ]
    return "index-coverage", _front_matter(
        "収録範囲", ["https://github.com/kazuhitogo/onsentology"], retrieved_at
    ) + "\n".join(lines)


def render_protocols(graph: Graph, retrieved_at: str) -> list[tuple[str, str]]:
    """法定プロトコルと一般的適応症・禁忌症。既存の検索レイヤの戻り値を文章にする。"""
    from . import queries

    documents: list[tuple[str, str]] = []
    for slug, title, payload in (
        ("protocol-bathing", "浴用の方法及び注意（掲示基準 2.(2)①）", queries.bathing_protocol(graph)),
        ("protocol-drinking", "飲用の方法及び注意（掲示基準 2.(2)②）", queries.drinking_protocol(graph)),
        ("general-indications", "療養泉の一般的適応症・温泉の一般的禁忌症", queries.general_indications(graph)),
    ):
        lines = [f"# {title}", ""]
        lines += _flatten(payload)
        lines.append("")
        documents.append(
            (
                slug,
                _front_matter(
                    title,
                    ["https://www.env.go.jp/nature/onsen/pdf/2-5_p_11.pdf"],
                    retrieved_at,
                )
                + "\n".join(lines),
            )
        )
    return documents


def _flatten(value: Any, depth: int = 0) -> list[str]:
    """辞書・リストを箇条書きにする。"""
    indent = "  " * depth
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{indent}- {key}:")
                lines += _flatten(item, depth + 1)
            else:
                lines.append(f"{indent}- {key}: {item}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines += _flatten(item, depth)
            else:
                lines.append(f"{indent}- {item}")
    else:
        lines.append(f"{indent}- {value}")
    return lines


def build_aligned_corpus(
    *,
    out_dir: Path | str = DEFAULT_ALIGNED_DIR,
    graph: Graph | None = None,
) -> dict[str, Any]:
    """揃えた生ドキュメントを書き出す。**推論値とヒューリスティックは入れない。**"""
    # 推論前のグラフを使う。導出結果を文書に焼き込むと「答えを書いておいた」ことになる。
    graph = graph if graph is not None else load_graph()
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(UTC).strftime("%Y-%m-%d")

    documents: list[tuple[str, str]] = [render_index(graph, retrieved_at)]
    for facility in graph.subjects(RDF.type, ONSEN.Facility):
        documents.append(render_facility(graph, facility, retrieved_at))
    for quality in _display_qualities(graph):
        documents.append(render_quality(graph, quality, retrieved_at))
    documents += render_protocols(graph, retrieved_at)

    total = 0
    for slug, body in documents:
        (directory / f"{slug}.md").write_text(body, encoding="utf-8")
        total += len(body)
    return {"文書数": len(documents), "総文字数": total, "置き場所": str(directory)}


__all__ = [
    "DEFAULT_ALIGNED_DIR",
    "build_aligned_corpus",
    "render_facility",
    "render_index",
    "render_protocols",
    "render_quality",
]
