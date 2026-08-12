"""推論エンジンのテスト。

数値分類ルールは、公式サイトが掲示泉質名に3分類軸を明記している源泉（ひょうたん温泉）で
推論結果と掲示内容を突き合わせて検算する。これが実質的な受け入れテストになる。
"""

from __future__ import annotations

from decimal import Decimal

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from onsen_ontology.namespaces import ONSEN
from onsen_ontology.reasoning import (
    RULE_QUALITY_FROM_COMPONENTS,
    apply_reasoning,
    apply_sparql_rules,
)

HYOTAN = URIRef("https://example.org/onsen/id/src-hyotan")
GERO = URIRef("https://example.org/onsen/id/src-gero-mori")
NAGAYU3 = URIRef("https://example.org/onsen/id/src-nagayu-3")
SHIRAHONE = URIRef("https://example.org/onsen/id/src-shirahone-awanoyu")
KUSATSU_YUBATAKE = URIRef("https://example.org/onsen/id/src-kusatsu-yubatake")
SAGIRIYU = URIRef("https://example.org/onsen/id/fac-sagiriyu")
GOZENYU = URIRef("https://example.org/onsen/id/fac-gozenyu")
KINNOYU = URIRef("https://example.org/onsen/id/fac-arima-kinnoyu")
HYOTAN_FAC = URIRef("https://example.org/onsen/id/fac-hyotan")


# --------------------------------------------------------------------------
# R1〜R3: 分類3軸
# --------------------------------------------------------------------------


def test_liquidity_class_matches_official_display(graph: Graph) -> None:
    """ひょうたん温泉は掲示名が「弱酸性低張性高温泉」。pH3.1 からの推論と一致するか。"""
    assert (HYOTAN, ONSEN.liquidityClass, ONSEN.WeaklyAcidic) in graph
    assert (HYOTAN, ONSEN.osmoticClass, ONSEN.Hypotonic) in graph
    assert (HYOTAN, ONSEN.temperatureClass, ONSEN.HighTemperatureSpring) in graph


def test_liquidity_boundaries(graph: Graph) -> None:
    """境界付近の値が正しい区分に落ちるか。"""
    # pH 9.2 → アルカリ性（8.5以上）
    assert (GERO, ONSEN.liquidityClass, ONSEN.Alkaline) in graph
    # pH 6.6 → 中性（6以上7.5未満）
    assert (NAGAYU3, ONSEN.liquidityClass, ONSEN.Neutral) in graph
    # pH 2.08 → 酸性（3未満）
    assert (KUSATSU_YUBATAKE, ONSEN.liquidityClass, ONSEN.Acidic) in graph


def test_temperature_class_boundaries(graph: Graph) -> None:
    """29.9℃ → 低温泉、37.3℃ → 温泉（泉温区分）。"""
    assert (NAGAYU3, ONSEN.temperatureClass, ONSEN.LowTemperatureSpring) in graph
    assert (SHIRAHONE, ONSEN.temperatureClass, ONSEN.MildTemperatureSpring) in graph


def test_classification_is_single_valued(graph: Graph) -> None:
    """区分は排他的でなければならない（境界の重複がないことの確認）。"""
    for source in graph.subjects(ONSEN.pH, None):
        classes = list(graph.objects(source, ONSEN.liquidityClass))
        assert len(set(classes)) <= 1, f"{source} に液性区分が複数: {classes}"
    for source in graph.subjects(ONSEN.sourceTemperature, None):
        classes = list(graph.objects(source, ONSEN.temperatureClass))
        assert len(set(classes)) <= 1, f"{source} に泉温区分が複数: {classes}"


def test_ph_does_not_imply_acidic_spring(graph: Graph) -> None:
    """液性が「酸性」であることと泉質が「酸性泉」であることは別。

    玉川温泉は pH1.2 だが、水素イオン濃度(mg/kg) が未公表なので**成分値からは**酸性泉を導けない。
    判定基準は pH ではなく水素イオン 1mg/kg 以上である。R4（成分値からの判定）は発火しない。

    酸性泉に分類されるのは掲示泉質名「酸性・含二酸化炭素・鉄（Ⅱ）－塩化物温泉」の表記からの
    読み取り（``onsen:hasQualityFromName``）による。根拠が違うので pH から導いたことにはならない。
    """
    tamagawa = URIRef("https://example.org/onsen/id/src-tamagawa")
    assert (tamagawa, ONSEN.liquidityClass, ONSEN.Acidic) in graph
    assert (
        tamagawa,
        ONSEN.displayedQualityName,
        Literal("酸性・含二酸化炭素・鉄（Ⅱ）－塩化物温泉"),
    ) in graph
    # 成分値が無いので R4 は玉川に対して何も導出しない
    assert graph.value(tamagawa, ONSEN.hydrogenIon) is None
    minimal = Graph()
    minimal.parse("ontology/onsen_ontology.ttl", format="turtle")
    minimal.parse("ontology/onsen_knowledge.ttl", format="turtle")
    minimal.parse("ontology/onsen_instances.ttl", format="turtle")
    apply_sparql_rules(minimal, rules=(RULE_QUALITY_FROM_COMPONENTS,))
    assert (tamagawa, ONSEN.hasQuality, ONSEN.AcidicSpring) not in minimal
    # 分類の根拠は掲示名の読み取りだけである
    assert (tamagawa, ONSEN.hasQualityFromName, ONSEN.AcidicSpring) in minimal


def test_quality_from_name_is_read_for_all_special_components(graph: Graph) -> None:
    """掲示泉質名に現れた特殊成分はすべて読み取る。現れないものは読み取らない。

    「酸性・含二酸化炭素・鉄（Ⅱ）－塩化物温泉」からは4つの掲示用泉質が読める
    （指針 1-3(3)3)・(4) の付記規則）。一方ラドン濃度は名称に現れず未公表なので、
    放射能泉に該当するかは分からない。分からないものは主張しない。
    """
    tamagawa = URIRef("https://example.org/onsen/id/src-tamagawa")
    assert set(graph.objects(tamagawa, ONSEN.hasQualityFromName)) == {
        ONSEN.AcidicSpring,
        ONSEN.CarbonDioxideSpring,
        ONSEN.IronSpring,
        ONSEN.ChlorideSpring,
    }
    assert (tamagawa, ONSEN.hasQuality, ONSEN.RadioactiveSpring) not in graph
    # 下位プロパティなので推論器が hasQuality を導出し、施設への畳み込みまで届く
    assert (tamagawa, ONSEN.hasQuality, ONSEN.AcidicSpring) in graph
    facility = URIRef("https://example.org/onsen/id/fac-tamagawa")
    assert (facility, ONSEN.facilityHasQuality, ONSEN.AcidicSpring) in graph


def test_quality_basis_is_declared_for_every_source(raw_graph: Graph) -> None:
    """泉質の判定根拠を型で分ける。実データは成分値由来と名称由来を混ぜない。

    ``onsen:hasQuality`` を直接主張してよいのは成分値から判定できる源泉だけで、
    データセットでは大涌谷温泉のみ（水素イオン1.27mg/kg）。それ以外はすべて
    掲示泉質名または旧泉質名の表記からの読み取りなので ``onsen:hasQualityFromName`` を使う。
    """
    assert (ONSEN.hasQualityFromName, RDFS.subPropertyOf, ONSEN.hasQuality) in raw_graph

    direct = {s for s, _ in raw_graph.subject_objects(ONSEN.hasQuality)}
    assert direct == {URIRef("https://example.org/onsen/id/src-owakudani")}
    for source in direct:
        assert raw_graph.value(source, ONSEN.hydrogenIon) is not None

    for source in raw_graph.subjects(RDF.type, ONSEN.SpringSource):
        if source in direct:
            continue
        if raw_graph.value(source, ONSEN.displayedQualityName) is None and (
            raw_graph.value(source, ONSEN.legacyQualityName) is None
        ):
            continue
        assert list(raw_graph.objects(source, ONSEN.hasQualityFromName)), source


def test_kobe_official_display_is_transcribed(graph: Graph) -> None:
    """有馬 炭酸泉源は神戸市の「温泉の掲示内容」から転記した唯一の完全な掲示である。

    掲示基準に沿った掲示そのもの（pH・泉温・分析年月日・分析者）が公表されている例。
    掲示の3分類軸（低張性、弱酸性、冷鉱泉）のうち、pH 4.3 から R1 が弱酸性を導く。
    """
    co2 = URIRef("https://example.org/onsen/id/src-arima-ginsen-co2")
    assert graph.value(co2, ONSEN.pH).toPython() == Decimal("4.3")
    assert graph.value(co2, ONSEN.sourceTemperature).toPython() == Decimal("18.3")
    assert str(graph.value(co2, ONSEN.analysisDate)) == "2019-10-25"
    assert (co2, ONSEN.liquidityClass, ONSEN.WeaklyAcidic) in graph
    assert (co2, ONSEN.measurementPoint, ONSEN.AtSourceOutlet) in graph


def test_sukayu_sources_differ_within_one_bathroom(graph: Graph) -> None:
    """酸ヶ湯 千人風呂は1浴室の中で源泉ごとに泉質が違う。

    環境省の国民保養温泉地計画から源泉温度を転記した。鹿の湯だけが硫黄泉ではない。
    """
    prefix = "https://example.org/onsen/id/"
    temperatures = {
        "src-sukayu-netsunoyu": Decimal("48.1"),
        "src-sukayu-shibunrokubun": Decimal("56.7"),
        "src-sukayu-shikanoyu": Decimal("67.6"),
        "src-sukayu-hienoyu-dai": Decimal("64.8"),
        "src-sukayu-hienoyu-sho": Decimal("69.7"),
    }
    for name, expected in temperatures.items():
        source = URIRef(prefix + name)
        assert graph.value(source, ONSEN.sourceTemperature).toPython() == expected, name
        # 42℃以上なのでいずれも高温泉に分類される
        assert (source, ONSEN.temperatureClass, ONSEN.HighTemperatureSpring) in graph

    shikanoyu = URIRef(prefix + "src-sukayu-shikanoyu")
    assert (shikanoyu, ONSEN.hasQuality, ONSEN.SulfurSpring) not in graph
    assert (shikanoyu, ONSEN.hasQuality, ONSEN.SulfateSpring) in graph
    netsunoyu = URIRef(prefix + "src-sukayu-netsunoyu")
    assert (netsunoyu, ONSEN.hasQuality, ONSEN.SulfurSpring) in graph


# --------------------------------------------------------------------------
# R4: 成分値からの泉質判定
# --------------------------------------------------------------------------


def test_quality_from_components_fires_on_synthetic_source() -> None:
    """成分値が揃っていれば泉質を導出できることを、合成データで確かめる。

    実データでは成分値が Web 公開されていない施設が多く、このルールはほとんど発火しない。
    ルール自体が壊れていないことをここで担保する。
    """
    graph = Graph()
    graph.parse("ontology/onsen_ontology.ttl", format="turtle")
    graph.parse("ontology/onsen_knowledge.ttl", format="turtle")
    source = URIRef("https://example.org/onsen/id/test-source")
    graph.add(
        (source, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), ONSEN.SpringSource)
    )
    graph.add((source, ONSEN.totalSulfur, Literal(2.5)))
    graph.add((source, ONSEN.hydrogenIon, Literal(1.4)))
    graph.add((source, ONSEN.dissolvedSolids, Literal(1500.0)))
    graph.add((source, ONSEN.mainAnion, Literal("SO4--")))
    graph.add((source, ONSEN.radon, Literal(3.0)))  # 8.25マッヘ未満 → 放射能泉ではない

    apply_sparql_rules(graph, rules=(RULE_QUALITY_FROM_COMPONENTS,))

    qualities = set(graph.objects(source, ONSEN.hasQuality))
    assert ONSEN.SulfurSpring in qualities
    assert ONSEN.AcidicSpring in qualities
    assert ONSEN.SulfateSpring in qualities
    assert ONSEN.RadioactiveSpring not in qualities
    # 陰イオン主成分が SO4 なので塩化物泉・炭酸水素塩泉にはならない
    assert ONSEN.ChlorideSpring not in qualities
    assert ONSEN.BicarbonateSpring not in qualities


def test_owakudani_meets_acidic_criterion(graph: Graph) -> None:
    """大涌谷温泉は水素イオン1.27mg/kg なので酸性泉の基準を満たす（1mg/kg以上）。"""
    owakudani = URIRef("https://example.org/onsen/id/src-owakudani")
    assert (owakudani, ONSEN.hasQuality, ONSEN.AcidicSpring) in graph
    assert (owakudani, ONSEN.osmoticClass, ONSEN.Hypotonic) in graph  # 1,013mg/kg < 8,000


def test_carbon_dioxide_threshold_not_met(graph: Graph) -> None:
    """長湯1号源泉は遊離二酸化炭素572mg/kg。二酸化炭素泉（1,000mg/kg以上）ではない。"""
    nagayu1 = URIRef("https://example.org/onsen/id/src-nagayu-1")
    assert (nagayu1, ONSEN.hasQuality, ONSEN.BicarbonateSpring) in graph
    assert (nagayu1, ONSEN.hasQuality, ONSEN.CarbonDioxideSpring) not in graph


# --------------------------------------------------------------------------
# R5: アルカリ性単純温泉
# --------------------------------------------------------------------------


def test_alkaline_simple_spring(graph: Graph) -> None:
    """下呂温泉は単純温泉かつ pH9.2 なのでアルカリ性単純温泉。"""
    assert (GERO, ONSEN.isAlkalineSimpleSpring, Literal(True)) in graph
    # 白骨は硫黄泉・炭酸水素塩泉であって単純温泉ではないので導出されない
    assert (SHIRAHONE, ONSEN.isAlkalineSimpleSpring, Literal(True)) not in graph


# --------------------------------------------------------------------------
# R6: 無加工供給（閉世界の否定）
# --------------------------------------------------------------------------


def test_unmodified_supply_requires_all_four_declarations(graph: Graph) -> None:
    """4類型すべてを否定している施設だけが無加工供給になる。"""
    assert (SAGIRIYU, ONSEN.isUnmodifiedSupply, Literal(True)) in graph
    assert (GOZENYU, ONSEN.isUnmodifiedSupply, Literal(True)) in graph


def test_kakenagashi_claim_does_not_imply_unmodified(graph: Graph) -> None:
    """有馬 金の湯は「かけ流し」を自主表示するが加水・加温・消毒を実施 → 無加工ではない。"""
    assert (KINNOYU, ONSEN.claimsKakenagashi, Literal(True)) in graph
    assert (KINNOYU, ONSEN.isUnmodifiedSupply, Literal(True)) not in graph


def test_missing_declaration_is_not_negation(graph: Graph) -> None:
    """宣言がないことと「実施していない」ことを混同しない。

    ひょうたん温泉は加水・加温を否定しているが循環ろ過・消毒の掲示が未確認なので、
    無加工供給は導出されない。開世界仮説を明示的に閉じている箇所。
    """
    assert (HYOTAN_FAC, ONSEN.claimsKakenagashi, Literal(True)) in graph
    assert (HYOTAN_FAC, ONSEN.isUnmodifiedSupply, Literal(True)) not in graph


# --------------------------------------------------------------------------
# R7 と OWL 推論
# --------------------------------------------------------------------------


def test_skin_irritating_derived_for_acidic_and_sulfur(graph: Graph) -> None:
    """皮膚刺激が強いと導出されるのは酸性泉と硫黄泉のみ。"""
    irritating = set(graph.subjects(ONSEN.isSkinIrritating, Literal(True)))
    assert irritating == {ONSEN.AcidicSpring, ONSEN.SulfurSpring}


def test_property_chain_facility_to_quality(graph: Graph) -> None:
    """OWL プロパティチェーンで施設→源泉→泉質が1ホップに畳まれている。"""
    qualities = set(graph.objects(HYOTAN_FAC, ONSEN.facilityHasQuality))
    assert ONSEN.ChlorideSpring in qualities


def test_property_chain_facility_to_indication(graph: Graph) -> None:
    """施設→源泉→泉質→浴用適応症の3段チェーン。"""
    indications = set(graph.objects(HYOTAN_FAC, ONSEN.facilityHasBathingIndication))
    assert ONSEN.DrySkin in indications  # 塩化物泉の浴用適応症


def test_inverse_property(graph: Graph) -> None:
    """hasSpringSource の逆プロパティ isSourceOf が導出されている。"""
    assert (HYOTAN, ONSEN.isSourceOf, HYOTAN_FAC) in graph


def test_symmetric_incompatible_with(graph: Graph) -> None:
    """incompatibleWith は対称プロパティ。片方向だけ書けば逆も導出される。"""
    assert (ONSEN.AcidicSpring, ONSEN.incompatibleWith, ONSEN.SulfurSpring) in graph
    assert (ONSEN.SulfurSpring, ONSEN.incompatibleWith, ONSEN.AcidicSpring) in graph


def test_subclass_propagation(graph: Graph) -> None:
    """塩類泉の個体が掲示用泉質としても型付けされている。"""
    types = set(
        graph.objects(
            ONSEN.ChlorideSpring, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        )
    )
    assert ONSEN.SpringQuality in types
    assert ONSEN.SaltSpringQuality in types


def test_reasoning_is_idempotent() -> None:
    """同じ推論を2回かけてもトリプルは増えない。"""
    from onsen_ontology.graph import load_graph

    graph = load_graph()
    apply_reasoning(graph)
    size = len(graph)
    added = apply_reasoning(graph)
    assert len(graph) == size
    assert all(count == 0 for count in added.values()), added
