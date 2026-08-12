"""湯治コンシェルジュセッションのテスト。

Bedrock はスタブに差し替える。判断（申し送り・裏取り・書き直し）はすべて
:mod:`onsen_ontology.consult` 側にあるので、AWS 無しで検証できる。
"""

from __future__ import annotations

from rdflib import Graph
from test_agent import StubBedrockClient, _text_response, _tool_use_response

from onsen_ontology.agent import OnsenGeezerAgent, OnsenOntologyTools
from onsen_ontology.consult import ConsultSession


def _session(graph: Graph, responses: list[dict], **kwargs: bool) -> ConsultSession:
    agent = OnsenGeezerAgent(OnsenOntologyTools(graph), client=StubBedrockClient(responses))
    return ConsultSession(agent, **kwargs)


def _user_texts(session: ConsultSession) -> list[str]:
    """会話履歴のうち、こちらから送ったテキスト（相談文・差し戻し）。"""
    return [
        block["text"]
        for message in session.agent.messages
        if message["role"] == "user"
        for block in message["content"]
        if "text" in block
    ]


# --------------------------------------------------------------------------
# 自動裏取り
# --------------------------------------------------------------------------


def test_未出典の施設名をツールで裏取りして書き直させる(graph: Graph) -> None:
    session = _session(
        graph,
        [
            _text_response("草津の大滝乃湯がええぞ。"),
            _text_response("草津温泉の大滝乃湯は酸性泉・硫黄泉じゃ。掲示基準の適応症を見よ。"),
        ],
    )
    turn = session.ask("草津でどこがええかい")

    assert turn.repaired is True
    assert [finding.text for finding in turn.findings_before_repair] == ["大滝乃湯"]
    # 検算役が代わりに describe_facility を呼んでいる
    assert [call.name for call in turn.tool_calls] == ["describe_facility"]
    assert turn.tool_calls[0].input == {"name": "大滝乃湯"}
    # 裏が取れたので書き直し後の回答では指摘が消える
    assert turn.findings == []


def test_裏取りは切れる(graph: Graph) -> None:
    session = _session(graph, [_text_response("草津の大滝乃湯がええぞ。")], auto_repair=False)
    turn = session.ask("草津でどこがええかい")
    assert turn.repaired is False
    assert [finding.text for finding in turn.findings] == ["大滝乃湯"]


def test_指摘に裏取り用のツール呼び出しが付く(graph: Graph) -> None:
    session = _session(graph, [_text_response("草津の大滝乃湯がええぞ。")], auto_repair=False)
    turn = session.ask("草津でどこがええかい")
    finding = turn.findings[0]
    assert finding.hint == ("describe_facility", {"name": "大滝乃湯"})
    assert "describe_facility" in finding.format()


def test_症状名の裏取りは症状検索ツールを使う(graph: Graph) -> None:
    session = _session(
        graph, [_text_response("アトピー性皮膚炎にはええぞ。")], auto_repair=False
    )
    turn = session.ask("どこがええかい")
    hints = [finding.hint for finding in turn.findings]
    assert ("search_by_symptom", {"keyword": "アトピー性皮膚炎"}) in hints


def test_通俗表現だけなら裏取りしない(graph: Graph) -> None:
    """hint が無い指摘（通俗表現）では書き直しを起こさない。"""
    session = _session(graph, [_text_response("そこは美人の湯と呼ばれておる。")])
    turn = session.ask("どこがええかい")
    assert turn.repaired is False
    assert [finding.text for finding in turn.findings] == ["美人の湯"]


# --------------------------------------------------------------------------
# 申し送り
# --------------------------------------------------------------------------


def test_前のターンの指摘を次の相談に申し送る(graph: Graph) -> None:
    session = _session(
        graph,
        [
            _text_response("そこは美人の湯と呼ばれておる。"),
            _text_response("掲示基準に基づく適応症を見るとええぞ。"),
        ],
    )
    session.ask("草津はどうじゃ")
    session.ask("有馬はどうじゃ")

    sent = _user_texts(session)[-1]
    assert "申し送り" in sent
    assert "美人の湯" in sent
    assert sent.endswith("有馬はどうじゃ")


def test_申し送りは切れる(graph: Graph) -> None:
    session = _session(
        graph,
        [
            _text_response("そこは美人の湯と呼ばれておる。"),
            _text_response("掲示基準に基づく適応症を見るとええぞ。"),
        ],
        carry_over=False,
    )
    session.ask("草津はどうじゃ")
    session.ask("有馬はどうじゃ")
    assert _user_texts(session)[-1] == "有馬はどうじゃ"


def test_指摘がなければ申し送りは付かない(graph: Graph) -> None:
    session = _session(
        graph,
        [
            _text_response("掲示基準に基づく適応症を見るとええぞ。"),
            _text_response("うむ。"),
        ],
    )
    session.ask("草津はどうじゃ")
    session.ask("有馬はどうじゃ")
    assert _user_texts(session)[-1] == "有馬はどうじゃ"


# --------------------------------------------------------------------------
# 出典の範囲
# --------------------------------------------------------------------------


def test_前のターンで引いた値は次のターンでも出典になる(graph: Graph) -> None:
    """セッションでは出典の範囲をセッション全体にする。

    1ターン目に describe_facility を引いていれば、3ターン目に同じ pH を言うのは
    裏の取れた記述である。単発の ask はターン内しか見ないので、ここが違い。
    """
    session = _session(
        graph,
        [
            _tool_use_response("describe_facility", {"name": "西の河原露天風呂"}),
            _text_response("西の河原露天風呂は万代源泉じゃ。掲示基準の適応症を見よ。"),
            _text_response("さきに言うたとおり pH1.7 の強い酸性じゃ。"),
        ],
    )
    session.ask("草津の西の河原露天風呂はどうじゃ")
    turn = session.ask("酸性はどれくらい強いんじゃ")

    assert turn.findings == []
    assert turn.repaired is False


def test_ターン内にしか出典がなければ単発では検出される(graph: Graph) -> None:
    """比較のため、セッションを使わない場合の挙動を固定する。"""
    from onsen_ontology.verify import verify_answer

    findings = verify_answer("pH1.7 の強い酸性じゃ。", [], graph=graph)
    assert [finding.text for finding in findings] == ["pH 1.7"]


# --------------------------------------------------------------------------
# 集計
# --------------------------------------------------------------------------


def test_セッションの集計(graph: Graph) -> None:
    session = _session(
        graph,
        [
            _text_response("草津の大滝乃湯がええぞ。"),
            _text_response("草津温泉の大滝乃湯は酸性泉・硫黄泉じゃ。掲示基準の適応症を見よ。"),
            _text_response("そこは美人の湯と呼ばれておる。"),
        ],
    )
    session.ask("草津でどこがええかい")
    session.ask("ほかにはあるかい")

    report = session.report()
    assert report["ターン数"] == 2
    assert report["裏取りを行ったターン"] == 1
    assert report["検算"] == {"件数": 1, "種別別": {"folk_expression": 1}}
    assert report["ターンごと"][0]["裏取り前の検算"] == {
        "件数": 1,
        "種別別": {"unsourced_term": 1},
    }
