"""SPARQL 検索レイヤと巡浴プランのテスト。"""

from __future__ import annotations

import json

from rdflib import Graph

from onsen_ontology import itinerary, queries

# --------------------------------------------------------------------------
# 検索
# --------------------------------------------------------------------------


def test_list_facilities(graph: Graph) -> None:
    facilities = queries.list_facilities(graph)
    assert len(facilities) >= 15
    names = {f["施設名"] for f in facilities}
    assert "有馬本温泉 金の湯" in names
    assert "登別温泉 さぎり湯" in names


def test_describe_facility_includes_treatments_with_reasons(graph: Graph) -> None:
    """湯使いは「その旨」と「その理由」の両方が返る（掲示の法定要件）。"""
    detail = queries.describe_facility(graph, "金の湯")
    assert detail is not None
    assert detail["施設名"] == "有馬本温泉 金の湯"
    treatments = {t["類型"]: t for t in detail["湯使い"]}
    assert treatments["加水"]["実施"] is True
    assert "水道水" in treatments["加水"]["理由"]
    assert treatments["循環（ろ過）"]["実施"] is False
    assert detail["出典"]


def test_describe_facility_is_json_serializable(graph: Graph) -> None:
    """ツールの戻り値は LLM に渡すため JSON 化できる必要がある（xsd:decimal 対策）。"""
    detail = queries.describe_facility(graph, "ひょうたん")
    assert detail is not None
    encoded = json.dumps(detail, ensure_ascii=False)
    assert "3.1" in encoded  # pH
    assert "100.4" in encoded  # 源泉温度


def test_describe_facility_reports_unconfirmed_data(graph: Graph) -> None:
    """未確認であることが呼び出し側に伝わる。"""
    detail = queries.describe_facility(graph, "金の湯")
    assert detail is not None
    assert detail["源泉"][0]["pH"] is None
    assert "未確認" in detail["源泉"][0]["データ状態"]


def test_resolve_facility_prefers_shortest_match(graph: Graph) -> None:
    uri = queries.resolve_facility(graph, "金の湯")
    assert uri is not None and str(uri).endswith("fac-arima-kinnoyu")
    assert queries.resolve_facility(graph, "存在しない温泉") is None


def test_resolve_facility_handles_multi_token_names(graph: Graph) -> None:
    """LLM が渡してくる「有馬 金の湯」のような組み合わせ名を解決できる。

    実際のラベルは「有馬本温泉 金の湯」なので単純な部分一致では外れる。
    """
    for query in ("有馬 金の湯", "有馬温泉 金の湯", "草津 大滝乃湯", "長湯温泉 御前湯"):
        assert queries.resolve_facility(graph, query) is not None, query
    assert str(queries.resolve_facility(graph, "有馬 金の湯")).endswith("fac-arima-kinnoyu")
    assert str(queries.resolve_facility(graph, "有馬 銀の湯")).endswith("fac-arima-ginnoyu")


def test_symptom_search_maps_colloquial_to_official_terms(graph: Graph) -> None:
    """口語から掲示基準の用語への対応づけ。"""
    result = queries.find_facilities_by_symptom(graph, "肌が荒れてガサガサする")
    assert "皮膚乾燥症" in result["対応づけた掲示基準の用語"]
    matched = result["泉質別適応症で一致した施設"]
    assert matched
    assert {m["泉質"] for m in matched} & {"塩化物泉", "硫酸塩泉", "炭酸水素塩泉"}
    assert "医学的な治療推奨ではない" in result["注記"]


def test_symptom_search_includes_general_indications(graph: Graph) -> None:
    result = queries.find_facilities_by_symptom(graph, "疲れが取れない")
    general = result["療養泉の一般的適応症（泉質を問わず共通）"]
    assert "疲労回復" in general


def test_symptom_search_deduplicates(graph: Graph) -> None:
    """長湯 御前湯は3源泉すべて炭酸水素塩泉なので、重複行が出ないこと。"""
    result = queries.find_facilities_by_symptom(graph, "肌の乾燥")
    keys = [(m["施設名"], m["泉質"], m["適応症"]) for m in result["泉質別適応症で一致した施設"]]
    assert len(keys) == len(set(keys))


def test_describe_spring_quality(graph: Graph) -> None:
    acidic = queries.describe_spring_quality(graph, "酸性泉")
    assert acidic is not None
    assert acidic["判定基準"]["下限値"] == 1.0
    assert acidic["皮膚刺激が強い（推論値）"] is True
    assert "アトピー性皮膚炎" in acidic["浴用適応症"]
    assert "高齢者の皮膚乾燥症" in acidic["泉質別浴用禁忌症"]
    assert acidic["飲用適応症"] == []


def test_describe_spring_quality_finishing_relation(graph: Graph) -> None:
    chloride = queries.describe_spring_quality(graph, "塩化物泉")
    assert chloride is not None
    assert set(chloride["仕上げ湯として推奨される先行泉質"]) == {"酸性泉", "硫黄泉"}


def test_general_indications_and_protocol(graph: Graph) -> None:
    general = queries.general_indications(graph)
    assert len(general["療養泉の一般的適応症（浴用）"]) == 15
    assert "活動性の結核" in general["温泉の一般的禁忌症（浴用）"]

    protocol = queries.bathing_protocol(graph)
    assert protocol["1日入浴回数上限（開始後数日間）"] == 2
    assert protocol["1回入浴時間上限（初期・分）"] == 10
    assert protocol["入浴後安静時間（分）"] == 30
    assert protocol["湯あたり発現時期（日）"] == [3, 7]

    drinking = queries.drinking_protocol(graph)
    assert drinking["1日飲用総量上限（mL）"] == 500
    assert drinking["飲用可能年齢の下限"] == 16


# --------------------------------------------------------------------------
# 含有成分別禁忌症の算術
# --------------------------------------------------------------------------


def test_component_contraindication_matches_notice_example(graph: Graph) -> None:
    """掲示基準の条文にある計算例を再現する。

    原文: ナトリウム3,000mg、カリウム200mg、マグネシウム60mg、よう化物1mg の温泉は
      1日100mL超（よう化物）→ 甲状腺機能亢進症
      1日400mL超（ナトリウム）→ 塩分制限の必要な病態
      カリウム・マグネシウムは算出値が500mL以上のため掲示不要
    """
    result = queries.evaluate_drinking_contraindications(graph, "計算例")
    by_rule = {r["ルール"]: r for r in result["評価結果"]}

    iodide = by_rule["よう化物イオンによる飲用禁忌"]
    assert iodide["限界飲用量(mL/日)"] == 100.0
    assert iodide["掲示が必要"] is True
    assert iodide["飲用禁忌症"] == ["甲状腺機能亢進症"]

    sodium = by_rule["ナトリウムイオンによる飲用禁忌"]
    assert sodium["限界飲用量(mL/日)"] == 400.0
    assert sodium["掲示が必要"] is True

    potassium = by_rule["カリウムイオンによる飲用禁忌"]
    assert potassium["限界飲用量(mL/日)"] == 4500.0
    assert potassium["掲示が必要"] is False

    magnesium = by_rule["マグネシウムイオンによる飲用禁忌"]
    assert magnesium["限界飲用量(mL/日)"] == 5000.0
    assert magnesium["掲示が必要"] is False


def test_component_contraindication_reports_missing_data(graph: Graph) -> None:
    """成分量が未確認なら「計算不能」と返す（0 や None で誤魔化さない）。"""
    result = queries.evaluate_drinking_contraindications(graph, "1号源泉")
    assert all("計算不能" in r.get("判定", "") for r in result["評価結果"])


# --------------------------------------------------------------------------
# 巡浴プラン
# --------------------------------------------------------------------------


def test_plan_orders_irritating_first_finishing_last(graph: Graph) -> None:
    """刺激の強い泉が先、仕上げ湯が後に並ぶ。"""
    plan = itinerary.plan_itinerary(graph, adapted=True, max_baths=3)
    steps = plan["プラン"]
    assert len(steps) == 3
    assert steps[0]["刺激の強い泉質を含む"] is True
    assert steps[-1]["刺激の強い泉質を含む"] is False


def test_plan_respects_bath_count_limit(graph: Graph) -> None:
    """回数上限は掲示基準の値から取る（初期2回、慣れて3回）。"""
    assert len(itinerary.plan_itinerary(graph)["プラン"]) <= 2
    assert len(itinerary.plan_itinerary(graph, adapted=True)["プラン"]) <= 3


def test_validate_detects_too_many_baths(graph: Graph) -> None:
    result = itinerary.describe_itinerary(
        graph, ["御座之湯", "日進舘", "玉川温泉", "さぎり湯"], consecutive_days=1
    )
    legal = [w for w in result["警告"] if w["severity"] == "legal"]
    assert any("上限を超えている" in w["message"] and "入浴回数" in w["basis"] for w in legal)


def test_validate_detects_too_long_bath(graph: Graph) -> None:
    result = itinerary.describe_itinerary(graph, ["御座之湯"], minutes=30)
    assert any("入浴時間" in w["basis"] for w in result["警告"])


def test_validate_detects_short_gap(graph: Graph) -> None:
    result = itinerary.describe_itinerary(graph, ["御座之湯", "さぎり湯"], gap_minutes=5)
    assert any("入浴後の注意" in w["basis"] for w in result["警告"])


def test_validate_detects_yuatari_window(graph: Graph) -> None:
    """3日目以降は湯あたりの発現時期に入る。"""
    result = itinerary.describe_itinerary(graph, ["御座之湯"], consecutive_days=4)
    assert any("湯あたり" in w["basis"] for w in result["警告"])
    result1 = itinerary.describe_itinerary(graph, ["御座之湯"], consecutive_days=1)
    assert not any("湯あたり" in w["basis"] for w in result1["警告"])


def test_validate_high_temperature_only_when_applicable(graph: Graph) -> None:
    """高温浴の警告は該当する人にだけ出す。"""
    without = itinerary.describe_itinerary(graph, ["日進舘"], high_temperature_caution=False)
    assert not any("入浴温度" in w["basis"] for w in without["警告"])
    with_caution = itinerary.describe_itinerary(graph, ["日進舘"], high_temperature_caution=True)
    assert any("入浴温度" in w["basis"] for w in with_caution["警告"])


def test_validate_detects_consecutive_irritating(graph: Graph) -> None:
    """刺激の強い泉の連続は heuristic 警告になる（legal ではない）。"""
    result = itinerary.describe_itinerary(graph, ["御座之湯", "日進舘"])
    consecutive = [
        w for w in result["警告"] if "刺激の強い泉質" in w["message"] and "連続" in w["message"]
    ]
    assert consecutive
    assert all(w["severity"] == "heuristic" for w in consecutive)


def test_warning_severity_values(graph: Graph) -> None:
    """警告の severity は legal か heuristic のみ。"""
    result = itinerary.plan_itinerary(
        graph, adapted=True, consecutive_days=5, high_temperature_caution=True
    )
    assert result["警告"]
    assert {w["severity"] for w in result["警告"]} <= {"legal", "heuristic"}


def test_plan_by_area(graph: Graph) -> None:
    plan = itinerary.plan_itinerary(graph, area="草津")
    assert plan["検討した候補数"] == 3
    assert all("草津" in s["温泉地"] for s in plan["プラン"])


def test_plan_unknown_area(graph: Graph) -> None:
    plan = itinerary.plan_itinerary(graph, area="存在しない温泉地")
    assert "error" in plan


def test_describe_itinerary_reports_unknown_facility(graph: Graph) -> None:
    result = itinerary.describe_itinerary(graph, ["御座之湯", "架空の湯"])
    assert result["見つからなかった施設"] == ["架空の湯"]

def test_源泉名から源泉を引ける(graph: Graph) -> None:
    """Phase 8 で足した穴埋め。**値はグラフにあるのに引けない**状態を解消する。

    Phase 7 の実測では「草津の湯畑源泉の pH は」に対して施設名で引くツールしか無く、
    Haiku は `describe_facility("湯畑")` が空振りしたところで諦めていた（該当2問で 4/15）。
    """
    result = queries.describe_spring_source(graph, "湯畑")
    assert result is not None
    assert result["源泉名"] == "湯畑源泉"
    assert result["pH"] == 2.08
    assert "酸性泉" in result["掲示用泉質"]
    assert result["この源泉を使う施設"] == ["御座之湯"]
    assert any(url.startswith("http") for url in result["出典"])
    # 未公表の理由も返す（「無い」と言える根拠になる）
    assert "源泉温度は未確認" in result["データ状態"]
    assert queries.describe_spring_source(graph, "秋保") is None


def test_仕上げ湯は逆向きにも引ける(graph: Graph) -> None:
    """「酸性泉のあとに何がよいか」は recommendedAfter の逆向きである。

    Phase 7 では順方向しか返しておらず、酸性泉の戻り値では仕上げ湯の欄が空だった。
    辺があっても引く向きを用意しなければ答えは出ない、という実測の教訓を固定する。
    """
    acidic = queries.describe_spring_quality(graph, "酸性泉")
    assert acidic["仕上げ湯として推奨される先行泉質"] == []
    after = acidic["この泉質のあとの仕上げ湯として推奨される泉質（経験則）"]
    assert "単純温泉" in after
    assert "炭酸水素塩泉" in after
    # 単純温泉から見れば順方向に酸性泉・硫黄泉が並ぶ（向きが逆であること自体の確認）
    simple = queries.describe_spring_quality(graph, "単純温泉")
    assert "酸性泉" in simple["仕上げ湯として推奨される先行泉質"]
