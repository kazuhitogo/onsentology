"""回答検算レイヤのテスト。

LLM の回答文をツール戻り値と照合する層なので、Bedrock は呼ばない。
ツール呼び出しの記録を手で組み立てて検査する。
"""

from __future__ import annotations

from rdflib import Graph

from onsen_ontology.agent import OnsenOntologyTools, ToolCallLog
from onsen_ontology.verify import (
    Finding,
    format_findings,
    ontology_vocabulary,
    revision_request,
    summarize,
    verify_answer,
)


def _call(tool: str, output: object, **arguments: object) -> ToolCallLog:
    return ToolCallLog(name=tool, input=dict(arguments), output=output)


def _kinds(findings: list[Finding]) -> set[str]:
    return {finding.kind for finding in findings}


# --------------------------------------------------------------------------
# 数値の検算
# --------------------------------------------------------------------------


def test_ツール戻り値にある数値は通る() -> None:
    calls = [_call("describe_facility", {"施設名": "ひょうたん温泉", "pH": 3.1, "源泉温度": 100.4})]
    findings = verify_answer(
        "ひょうたん温泉は pH3.1、源泉100.4℃じゃ。掲示基準の適応症を見るかい。", calls
    )
    assert findings == []


def test_ツール戻り値にない数値を検出する() -> None:
    calls = [_call("describe_facility", {"施設名": "ひょうたん温泉", "pH": 3.1})]
    findings = verify_answer("ひょうたん温泉は pH2.08 じゃ。", calls)
    assert _kinds(findings) == {"unsourced_quantity"}
    assert "2.08" in findings[0].text


def test_相談文にある数値は通る() -> None:
    calls = [_call("get_bathing_protocol", {"注意": "安静にする"})]
    findings = verify_answer(
        "3日の湯治なら気をつけることがあるのう。", calls, question="3日ほど湯治したい"
    )
    assert findings == []


def test_範囲表記の両端を検査する() -> None:
    """「10〜15分」の 10 は単位に隣接しないので、正規化しないと取り落とす。"""
    calls = [_call("get_bathing_protocol", {"入浴時間": "1回当たり初めは3〜10分程度"})]
    assert verify_answer("初めは3〜10分にしておけ。", calls) == []

    findings = verify_answer("10〜15分が目安じゃ。", calls)
    assert [finding.text for finding in findings] == ["時間 15"]


def test_条番号や箇条書きの数字は数値として拾わない() -> None:
    calls = [_call("get_bathing_protocol", {"根拠": "掲示基準 2.(2)① オ"})]
    findings = verify_answer("これは掲示基準 2.(2)① オ に書いてあるのう。", calls)
    assert findings == []


def test_桁区切りのカンマを含む数値を照合できる() -> None:
    calls = [_call("evaluate_drinking_contraindications", {"限界飲用量mL": 4500.0})]
    assert verify_answer("限界飲用量は 4,500mL じゃ。", calls) == []
    assert _kinds(verify_answer("限界飲用量は 9,000mL じゃ。", calls)) == {"unsourced_quantity"}


# --------------------------------------------------------------------------
# 語彙の検算
# --------------------------------------------------------------------------


def test_語彙にオントロジーの施設名と症状名が入る(graph: Graph) -> None:
    vocabulary = ontology_vocabulary(graph)
    assert "ひょうたん温泉" in vocabulary
    assert "酸性泉" in vocabulary
    assert "アトピー性皮膚炎" in vocabulary


def test_ツールが返していない施設名を検出する(graph: Graph) -> None:
    calls = [_call("describe_facility", {"施設名": "ひょうたん温泉"}, name="ひょうたん")]
    findings = verify_answer(
        "草津の大滝乃湯もええぞ。掲示基準の話はまた今度じゃ。", calls, graph=graph
    )
    assert "unsourced_term" in _kinds(findings)
    assert any(finding.text == "大滝乃湯" for finding in findings)


def test_ツールが返した適応症名は通る(graph: Graph) -> None:
    calls = [
        _call(
            "describe_spring_quality",
            {"泉質名": "酸性泉", "浴用適応症": ["アトピー性皮膚炎", "尋常性乾癬"]},
            name="酸性泉",
        )
    ]
    findings = verify_answer(
        "酸性泉はアトピー性皮膚炎が掲示基準の浴用適応症に挙げられておる。", calls, graph=graph
    )
    assert findings == []


def test_条文表記の言い換えを検出する(graph: Graph) -> None:
    """語彙の照合では素通りしていた言い換えを、対応表で捕まえる。

    「病気の活動期（特に熱のあるとき）」→「急性疾患」は意味が近いので誤りではないが、
    掲示基準の条文表記ではない。対応表はグラフ側の onsen:nonStandardParaphrase にある。
    """
    calls = [_call("get_general_indications", {"一般的禁忌症": ["病気の活動期（特に熱のあるとき）"]})]
    findings = verify_answer(
        "急性疾患のときは入るなと掲示基準の禁忌症にあるのう。", calls, graph=graph
    )
    assert [finding.kind for finding in findings] == ["paraphrased_term"]
    assert findings[0].text == "急性疾患"
    assert "病気の活動期（特に熱のあるとき）" in findings[0].detail
    assert findings[0].severity == "warning"


def test_条文表記を併記していれば言い換えは通す(graph: Graph) -> None:
    calls = [_call("get_general_indications", {"一般的禁忌症": ["病気の活動期（特に熱のあるとき）"]})]
    findings = verify_answer(
        "掲示基準の禁忌症は「病気の活動期（特に熱のあるとき）」じゃ。いわゆる急性疾患のことじゃな。",
        calls,
        graph=graph,
    )
    assert findings == []


def test_自由記述の作用機序は依然として検出できない(graph: Graph) -> None:
    """残っている限界を明示するテスト。

    語彙表・通俗表現表・言い換え表のいずれにも無い自由記述は素通りする。
    「塩分が皮膚に膜を張る」のような作用機序の説明がその例である。
    """
    calls = [_call("describe_spring_quality", {"泉質名": "塩化物泉"}, name="塩化物泉")]
    findings = verify_answer(
        "塩化物泉は塩分が皮膚に膜を張って水分の蒸発を防ぐんじゃ。掲示基準の適応症も見よ。",
        calls,
        graph=graph,
    )
    assert findings == []


# --------------------------------------------------------------------------
# 通俗表現・開示義務
# --------------------------------------------------------------------------


def test_通俗的な効能表現を検出する() -> None:
    calls = [_call("describe_spring_quality", {"泉質名": "硫酸塩泉"}, name="硫酸塩泉")]
    findings = verify_answer(
        "硫酸塩泉は美人の湯と呼ばれ、塩化物泉には保温効果があるんじゃ。", calls
    )
    assert _kinds(findings) == {"folk_expression"}
    assert {finding.text for finding in findings} == {"美人の湯", "保温効果"}
    assert all(finding.severity == "warning" for finding in findings)


def test_ツール戻り値に出典がある通俗表現は通す() -> None:
    """「保湿・美肌効果」は積善館の公式記述として実データに入っている。"""
    calls = [
        _call(
            "describe_facility",
            {"施設名": "四万温泉 積善館", "理由": "保湿・美肌効果のある四万に滞在する慣習"},
            name="積善館",
        )
    ]
    assert verify_answer("積善館は保湿・美肌効果があると自ら記しておる。", calls) == []


def test_相談文にある通俗表現も指摘する() -> None:
    """利用者が「デトックス」と言ったことは、その語を使ってよい根拠にはならない。"""
    calls = [_call("get_general_indications", {"一般的適応症": ["健康増進"]})]
    findings = verify_answer(
        "デトックスという言葉は掲示基準には出てこんのじゃ。",
        calls,
        question="温泉でデトックスできるかい",
    )
    assert [finding.text for finding in findings] == ["デトックス"]
    assert findings[0].severity == "warning"


def test_適応症に触れながら掲示基準を明示しない場合を検出する() -> None:
    calls = [_call("describe_spring_quality", {"泉質名": "酸性泉"}, name="酸性泉")]
    findings = verify_answer("酸性泉の適応症にはいろいろあるのう。", calls)
    assert _kinds(findings) == {"missing_disclosure"}


def test_経験則の警告を前置きなしで伝えた場合を検出する() -> None:
    calls = [
        _call(
            "plan_itinerary",
            {"警告": [{"severity": "heuristic", "message": "最後が刺激の強い泉質で終わっている"}]},
            area="草津",
        )
    ]
    findings = verify_answer("最後は刺激の強い湯じゃな。順番を変えるとええぞ。", calls)
    assert _kinds(findings) == {"missing_disclosure"}

    ok = verify_answer(
        "これはわしの経験則じゃが、最後は刺激の強い湯を避けるとええぞ。", calls
    )
    assert ok == []


def test_法定の警告には経験則の前置きを求めない() -> None:
    calls = [
        _call(
            "plan_itinerary",
            {"警告": [{"severity": "legal", "message": "湯あたり", "根拠": "掲示基準 2.(2)①"}]},
            area="草津",
        )
    ]
    assert verify_answer("掲示基準では湯あたりに注意とあるのう。", calls) == []


# --------------------------------------------------------------------------
# 呼ぶべきツールを呼んでいない
# --------------------------------------------------------------------------


def test_相談の意図はグラフから読む(graph: Graph) -> None:
    """キーワード表を Python の定数ではなく onsen:ConsultIntent の個体として持つ。"""
    from onsen_ontology.verify import consult_expectations

    expectations = {e.label: e for e in consult_expectations(graph)}
    assert set(expectations) == {"巡浴の相談", "飲用の相談", "入浴方法の相談"}

    itinerary = expectations["巡浴の相談"]
    assert "巡浴" in itinerary.keywords and "順番" in itinerary.keywords
    assert set(itinerary.tools) == {"plan_itinerary", "validate_itinerary"}
    assert itinerary.repair_tool == "plan_itinerary"
    assert itinerary.severity == "error", "規則5は「必ず」なので error"
    assert expectations["飲用の相談"].repair_tool == "get_drinking_protocol"
    assert expectations["飲用の相談"].severity == "warning"


def test_巡浴の相談でプラン検証を呼んでいない場合を検出する(graph: Graph) -> None:
    calls = [_call("list_facilities", [])]
    findings = verify_answer(
        "草津なら3軒ほど回れるじゃろ。掲示基準の適応症も見ておけ。",
        calls,
        graph=graph,
        question="草津で回る順番を組んでくれ",
    )
    assert [finding.kind for finding in findings] == ["missing_tool_call"]
    assert findings[0].severity == "error"
    # 温泉地が相談文にあれば、裏取りの引数に入れる
    assert findings[0].hint == ("plan_itinerary", {"area": "草津温泉"})


def test_プラン検証を呼んでいれば指摘しない(graph: Graph) -> None:
    calls = [_call("plan_itinerary", {"順路": []}, area="草津")]
    findings = verify_answer(
        "この順で回るとええ。掲示基準に沿っておる。",
        calls,
        graph=graph,
        question="草津で回る順番を組んでくれ",
    )
    assert findings == []


def test_飲用の相談でプロトコルを引いていない場合を検出する(graph: Graph) -> None:
    findings = verify_answer(
        "飲むのはやめておけ。", [], graph=graph, question="飲泉もしてみたい"
    )
    kinds = {finding.kind for finding in findings}
    assert kinds == {"missing_tool_call"}
    assert findings[0].hint == ("get_drinking_protocol", {})


def test_関係ない相談では呼ぶべきツールを求めない(graph: Graph) -> None:
    calls = [
        _call(
            "describe_facility",
            {"施設名": "ひょうたん温泉", "泉質": ["塩化物泉"]},
            name="ひょうたん",
        )
    ]
    findings = verify_answer(
        "ひょうたん温泉は塩化物泉じゃ。掲示基準の適応症を見よ。",
        calls,
        graph=graph,
        question="ひょうたん温泉はどんな湯かい",
    )
    assert findings == []


# --------------------------------------------------------------------------
# 出力の整形
# --------------------------------------------------------------------------


def test_指摘なしの表示() -> None:
    assert format_findings([]) == "検算: 指摘なし"


def test_指摘ありの表示と集計() -> None:
    findings = verify_answer(
        "美人の湯じゃ。pH9.9 じゃったかのう。", [_call("list_facilities", [])]
    )
    text = format_findings(findings)
    assert text.startswith("検算: 2件の指摘")
    assert "folk_expression" in text
    assert summarize(findings) == {
        "件数": 2,
        "種別別": {"unsourced_quantity": 1, "folk_expression": 1},
    }


def test_差し戻し指示に指摘が入る() -> None:
    findings = verify_answer("pH9.9 じゃ。", [_call("list_facilities", [])])
    request = revision_request(findings)
    assert "9.9" in request
    assert "書き直" in request


# --------------------------------------------------------------------------
# 実際のツール戻り値との結合
# --------------------------------------------------------------------------


def test_実際のツール戻り値で検算が通る(graph: Graph) -> None:
    """オントロジーから引いた値だけで書いた回答は指摘ゼロになる。"""
    tools = OnsenOntologyTools(graph)
    output = tools.call("describe_facility", {"name": "ひょうたん"})
    calls = [_call("describe_facility", output, name="ひょうたん")]
    answer = (
        f"{output['施設名']}は pH{output['源泉'][0]['pH']}、"
        f"{output['源泉'][0]['源泉温度']}℃の湯じゃ。"
        "掲示基準に基づく適応症は現地の掲示を見るとええ。"
    )
    assert verify_answer(answer, calls, graph=graph, question="ひょうたん温泉のこと") == []
