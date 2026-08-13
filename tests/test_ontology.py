"""オントロジー（TTL）の構造テスト。

「10泉質がちょうど10個あるか」のような、一次情報との対応が崩れたら気づきたい性質を検査する。
"""

from __future__ import annotations

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, RDFS, SKOS

from onsen_ontology.namespaces import ONSEN


def _qualities(graph: Graph) -> set[URIRef]:
    """掲示用泉質の個体（サブクラス経由も含む）。"""
    found = set(graph.subjects(RDF.type, ONSEN.SpringQuality))
    for subclass in graph.subjects(RDFS.subClassOf, ONSEN.SpringQuality):
        found |= set(graph.subjects(RDF.type, subclass))
    return found


def test_ttl_parses(raw_graph: Graph) -> None:
    assert len(raw_graph) > 1000


def test_display_qualities_are_exactly_ten(raw_graph: Graph) -> None:
    """掲示用泉質はちょうど10種類（鉱泉分析法指針 1-3、掲示基準 別表）。"""
    labels = {str(raw_graph.value(q, RDFS.label)) for q in _qualities(raw_graph)}
    assert labels == {
        "単純温泉",
        "塩化物泉",
        "炭酸水素塩泉",
        "硫酸塩泉",
        "二酸化炭素泉",
        "含鉄泉",
        "酸性泉",
        "含よう素泉",
        "硫黄泉",
        "放射能泉",
    }


def test_quality_criteria_match_primary_source(raw_graph: Graph) -> None:
    """判定基準の閾値が一次情報どおりか（指針 第1-3表）。"""
    expected = {
        ONSEN.CarbonDioxideSpring: (ONSEN.freeCarbonDioxide, 1000.0),
        ONSEN.IronSpring: (ONSEN.totalIron, 20.0),
        ONSEN.AcidicSpring: (ONSEN.hydrogenIon, 1.0),
        ONSEN.IodineSpring: (ONSEN.iodideIon, 10.0),
        ONSEN.SulfurSpring: (ONSEN.totalSulfur, 2.0),
        ONSEN.RadioactiveSpring: (ONSEN.radon, 8.25),
        ONSEN.ChlorideSpring: (ONSEN.dissolvedSolids, 1000.0),
        ONSEN.BicarbonateSpring: (ONSEN.dissolvedSolids, 1000.0),
        ONSEN.SulfateSpring: (ONSEN.dissolvedSolids, 1000.0),
    }
    for quality, (prop, threshold) in expected.items():
        assert raw_graph.value(quality, ONSEN.criterionProperty) == prop, quality
        assert float(raw_graph.value(quality, ONSEN.criterionMinValue).toPython()) == threshold

    # 単純温泉だけは上限値型（溶存物質1,000mg未満かつ泉温25℃以上）
    assert float(raw_graph.value(ONSEN.SimpleSpring, ONSEN.criterionMaxValue).toPython()) == 1000.0
    assert (
        float(raw_graph.value(ONSEN.SimpleSpring, ONSEN.criterionMinTemperature).toPython()) == 25.0
    )


def test_classification_boundaries(raw_graph: Graph) -> None:
    """3分類軸の境界値（指針 1-2）。"""
    assert float(raw_graph.value(ONSEN.Acidic, ONSEN.maxPH).toPython()) == 3.0
    assert float(raw_graph.value(ONSEN.Alkaline, ONSEN.minPH).toPython()) == 8.5
    assert float(raw_graph.value(ONSEN.Neutral, ONSEN.minPH).toPython()) == 6.0
    assert float(raw_graph.value(ONSEN.Neutral, ONSEN.maxPH).toPython()) == 7.5

    assert float(raw_graph.value(ONSEN.ColdMineralSpring, ONSEN.maxTemperature).toPython()) == 25.0
    assert (
        float(raw_graph.value(ONSEN.LowTemperatureSpring, ONSEN.maxTemperature).toPython()) == 34.0
    )
    assert (
        float(raw_graph.value(ONSEN.HighTemperatureSpring, ONSEN.minTemperature).toPython()) == 42.0
    )

    assert float(raw_graph.value(ONSEN.Hypotonic, ONSEN.maxDissolvedSolids).toPython()) == 8000.0
    assert float(raw_graph.value(ONSEN.Hypertonic, ONSEN.minDissolvedSolids).toPython()) == 10000.0


def test_low_temperature_spring_is_not_called_bion(raw_graph: Graph) -> None:
    """泉温25〜34℃の正式名称は「低温泉」。「微温泉」は別名（俗称）として扱う。"""
    assert str(raw_graph.value(ONSEN.LowTemperatureSpring, RDFS.label)) == "低温泉"
    alt = [str(o) for o in raw_graph.objects(ONSEN.LowTemperatureSpring, SKOS.altLabel)]
    assert "微温泉" in alt


def test_quality_specific_contraindications_only_two(raw_graph: Graph) -> None:
    """泉質別禁忌症（浴用）が定められているのは酸性泉と硫黄泉の2泉質のみ。"""
    with_contra = {
        q
        for q in _qualities(raw_graph)
        if list(raw_graph.objects(q, ONSEN.hasBathingContraindication))
    }
    assert with_contra == {ONSEN.AcidicSpring, ONSEN.SulfurSpring}


def test_drinking_indications_seven_qualities(raw_graph: Graph) -> None:
    """飲用適応症が定められているのは7泉質（単純温泉・酸性泉・放射能泉を除く）。"""
    with_drinking = {
        str(raw_graph.value(q, RDFS.label))
        for q in _qualities(raw_graph)
        if list(raw_graph.objects(q, ONSEN.hasDrinkingIndication))
    }
    assert with_drinking == {
        "塩化物泉",
        "炭酸水素塩泉",
        "硫酸塩泉",
        "二酸化炭素泉",
        "含鉄泉",
        "含よう素泉",
        "硫黄泉",
    }


def test_sulfate_bathing_indications_equal_chloride(raw_graph: Graph) -> None:
    """掲示基準は硫酸塩泉の浴用適応症を「塩化物泉に同じ」と規定している。"""
    chloride = set(raw_graph.objects(ONSEN.ChlorideSpring, ONSEN.hasBathingIndication))
    sulfate = set(raw_graph.objects(ONSEN.SulfateSpring, ONSEN.hasBathingIndication))
    assert chloride == sulfate


def test_general_indications_count(raw_graph: Graph) -> None:
    """療養泉の一般的適応症（浴用）が漏れていないか。"""
    profile = ONSEN.TherapeuticSpringGeneralProfile
    indications = list(raw_graph.objects(profile, ONSEN.hasBathingIndication))
    contraindications = list(raw_graph.objects(profile, ONSEN.hasBathingContraindication))
    assert len(indications) == 15
    assert len(contraindications) == 8


def test_water_treatment_types_are_five(raw_graph: Graph) -> None:
    """法定の湯使い類型は5つ（温泉法施行規則第10条第2項）。"""
    types = {
        str(raw_graph.value(t, RDFS.label))
        for t in raw_graph.subjects(RDF.type, ONSEN.WaterTreatment)
    }
    assert types == {"加水", "加温", "循環（ろ過）", "入浴剤添加", "消毒"}


def test_heuristic_rules_are_annotated(raw_graph: Graph) -> None:
    """法令に根拠のないプロパティには必ず onsen:heuristicRule 注記が付いている。"""
    for prop in (
        ONSEN.recommendedAfter,
        ONSEN.incompatibleWith,
        ONSEN.colloquialExpression,
        ONSEN.nonStandardParaphrase,
        ONSEN.ConsultIntent,
    ):
        note = raw_graph.value(prop, ONSEN.heuristicRule)
        assert isinstance(note, Literal), prop
        assert len(str(note)) > 30


def test_heuristics_live_in_their_own_file() -> None:
    """独自ヒューリスティックは法定知識・実データと別ファイルに置く。

    出典が無いものを、出典があるもの（環境省の通知・各施設の公表値）と混ぜない。
    """
    from onsen_ontology.graph import HEURISTICS_FILE, INSTANCES_FILE, KNOWLEDGE_FILE

    heuristics = Graph()
    heuristics.parse(HEURISTICS_FILE, format="turtle")
    assert list(heuristics.subject_objects(ONSEN.colloquialExpression))
    assert list(heuristics.subject_objects(ONSEN.nonStandardParaphrase))
    assert list(heuristics.subjects(RDF.type, ONSEN.ConsultIntent))

    for path in (KNOWLEDGE_FILE, INSTANCES_FILE):
        other = Graph()
        other.parse(path, format="turtle")
        for prop in (
            ONSEN.colloquialExpression,
            ONSEN.nonStandardParaphrase,
            ONSEN.intentKeyword,
        ):
            assert not list(other.subject_objects(prop)), f"{path.name} に {prop} がある"


def test_nonstatutory_indications_are_absent_from_the_legal_file() -> None:
    """「現行の掲示基準に無い効能表記」という宣言を、法定知識ファイルに対して検査する。

    この表（``onsen:NonStatutoryIndication``）はどの語を列挙するかが本オントロジーの判断なので
    ヒューリスティックのファイルに置いている。ただし「現行の掲示基準の適応症一覧に無い」という
    主張自体は機械的に検査できる。``onsen_knowledge.ttl`` にその語が現れたら、宣言が誤っているか
    法定知識の側が変わったかのどちらかなので、落として教える。

    逆向きの取りこぼしも防ぐ。列挙した語が1件も無ければ、検算は何も検出しない。
    """
    from onsen_ontology.graph import HEURISTICS_FILE, KNOWLEDGE_FILE

    heuristics = Graph()
    heuristics.parse(HEURISTICS_FILE, format="turtle")
    terms = [
        str(label)
        for subject in heuristics.subjects(RDF.type, ONSEN.NonStatutoryIndication)
        for label in heuristics.objects(subject, RDFS.label)
    ]
    assert len(terms) >= 5

    legal_text = KNOWLEDGE_FILE.read_text(encoding="utf-8")
    for term in terms:
        assert term not in legal_text, f"現行の掲示基準に現れる語を非法定として宣言している: {term}"


def test_every_source_has_provenance(raw_graph: Graph) -> None:
    """すべての源泉に出典URLが付いている。"""
    for source in raw_graph.subjects(RDF.type, ONSEN.SpringSource):
        assert list(raw_graph.objects(source, DCTERMS.source)), f"出典がない源泉: {source}"


def test_kakenagashi_claim_is_separate_from_unmodified(raw_graph: Graph) -> None:
    """「かけ流し」の自主表示と法定湯使いの否定は別プロパティである。

    有馬 金の湯は claimsKakenagashi=true だが、加水・加温・消毒を実施しているため
    無加工供給ではない。この反例が残っていることを保証する。
    """
    kinnoyu = URIRef("https://example.org/onsen/id/fac-arima-kinnoyu")
    assert raw_graph.value(kinnoyu, ONSEN.claimsKakenagashi) == Literal(True)
    applied_types = {
        raw_graph.value(decl, ONSEN.treatmentType)
        for decl in raw_graph.objects(kinnoyu, ONSEN.declaresTreatment)
        if raw_graph.value(decl, ONSEN.isApplied) == Literal(True)
    }
    assert ONSEN.AddingWater in applied_types
    assert ONSEN.Disinfection in applied_types
