"""エージェントのテスト。

Bedrock を呼ぶ部分はスタブに差し替える。ツール層とループ制御は AWS 無しで検証できる。
実際に Bedrock を呼ぶテストは ``ONSEN_TEST_BEDROCK=1`` を付けたときだけ走る。
"""

from __future__ import annotations

import json
import os

import pytest
from rdflib import Graph

from onsen_ontology.agent import (
    SYSTEM_PROMPT,
    TOOL_SPECS,
    OnsenGeezerAgent,
    OnsenOntologyTools,
)

# --------------------------------------------------------------------------
# ツール層
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tools(graph: Graph) -> OnsenOntologyTools:
    return OnsenOntologyTools(graph)


def test_tool_specs_are_wellformed() -> None:
    """すべてのツールに対応するハンドラが存在し、スキーマが揃っている。"""
    names = set()
    for spec in TOOL_SPECS:
        tool = spec["toolSpec"]
        assert tool["name"]
        assert len(tool["description"]) > 20
        schema = tool["inputSchema"]["json"]
        assert schema["type"] == "object"
        for required in schema["required"]:
            assert required in schema["properties"], tool["name"]
        names.add(tool["name"])
    handlers = {a[len("_tool_") :] for a in dir(OnsenOntologyTools) if a.startswith("_tool_")}
    assert names == handlers


def test_all_tool_outputs_are_json_serializable(tools: OnsenOntologyTools) -> None:
    """LLM に渡すので全ツールの戻り値が JSON 化できる必要がある。"""
    calls = [
        ("search_by_symptom", {"keyword": "疲れが取れない"}),
        ("describe_facility", {"name": "ひょうたん"}),
        ("describe_spring_quality", {"name": "硫黄泉"}),
        ("list_facilities", {}),
        ("plan_itinerary", {"area": "草津"}),
        ("validate_itinerary", {"facilities": ["御座之湯", "さぎり湯"]}),
        ("get_general_indications", {}),
        ("get_bathing_protocol", {}),
        ("get_drinking_protocol", {}),
        ("evaluate_drinking_contraindications", {"source": "計算例"}),
    ]
    for name, arguments in calls:
        output = tools.call(name, arguments)
        assert output is not None, name
        json.dumps(output, ensure_ascii=False)  # 例外が出なければよい


def test_unknown_tool_returns_error(tools: OnsenOntologyTools) -> None:
    assert "error" in tools.call("no_such_tool", {})


def test_bad_arguments_return_error(tools: OnsenOntologyTools) -> None:
    assert "error" in tools.call("describe_facility", {"wrong_key": "x"})


def test_system_prompt_contains_guardrails() -> None:
    """人格設定だけでなく、幻覚と法定/経験則の混同を防ぐ制約が入っているか。"""
    assert "ツールが返した値だけ" in SYSTEM_PROMPT
    assert "掲示基準" in SYSTEM_PROMPT
    assert "legal" in SYSTEM_PROMPT and "heuristic" in SYSTEM_PROMPT
    assert "医師" in SYSTEM_PROMPT


# --------------------------------------------------------------------------
# Converse ループ（スタブ）
# --------------------------------------------------------------------------


class StubBedrockClient:
    """converse の応答を順番に返すスタブ。"""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    def converse(self, **kwargs: object) -> dict:
        self.requests.append(kwargs)
        return self.responses.pop(0)


def _text_response(text: str) -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }


def _tool_use_response(name: str, arguments: dict) -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tu-1",
                            "name": name,
                            "input": arguments,
                        }
                    }
                ],
            }
        },
        "usage": {"inputTokens": 20, "outputTokens": 8, "totalTokens": 28},
    }


def test_agent_resolves_tool_calls(tools: OnsenOntologyTools) -> None:
    client = StubBedrockClient(
        [
            _tool_use_response("describe_facility", {"name": "ひょうたん"}),
            _text_response("ひょうたん温泉は pH3.1、源泉100.4℃の塩化物泉じゃ。"),
        ]
    )
    agent = OnsenGeezerAgent(tools, client=client)
    result = agent.ask("別府のひょうたん温泉はどんな湯かい")

    assert result.turns == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "describe_facility"
    assert result.tool_calls[0].output["施設名"] == "ひょうたん温泉"
    assert "pH3.1" in result.text
    assert result.usage["totalTokens"] == 43

    # ツール結果が toolResult ブロックとして会話履歴に入っている
    tool_result_messages = [
        m
        for m in agent.messages
        if m["role"] == "user" and any("toolResult" in b for b in m["content"])
    ]
    assert len(tool_result_messages) == 1


def test_agent_sends_system_prompt_and_tools(tools: OnsenOntologyTools) -> None:
    client = StubBedrockClient([_text_response("はいはい。")])
    agent = OnsenGeezerAgent(tools, client=client)
    agent.ask("こんにちは")
    request = client.requests[0]
    assert request["system"][0]["text"] == SYSTEM_PROMPT
    assert len(request["toolConfig"]["tools"]) == len(TOOL_SPECS)


def test_agent_stops_at_max_turns(tools: OnsenOntologyTools) -> None:
    """ツールを呼び続ける応答でも max_turns で止まる。"""
    client = StubBedrockClient([_tool_use_response("list_facilities", {}) for _ in range(5)])
    agent = OnsenGeezerAgent(tools, client=client, max_turns=3)
    result = agent.ask("全部教えて")
    assert result.turns == 3
    assert len(result.tool_calls) == 3


def test_agent_keeps_conversation_history(tools: OnsenOntologyTools) -> None:
    client = StubBedrockClient([_text_response("一度目じゃ。"), _text_response("二度目じゃ。")])
    agent = OnsenGeezerAgent(tools, client=client)
    agent.ask("最初の質問")
    agent.ask("次の質問")
    assert len(agent.messages) == 4
    assert agent.messages[0]["content"][0]["text"] == "最初の質問"


# --------------------------------------------------------------------------
# 検算レイヤとの結合
# --------------------------------------------------------------------------


def test_検算結果が結果に入る(tools: OnsenOntologyTools) -> None:
    client = StubBedrockClient([_text_response("硫酸塩泉は美人の湯と呼ばれておる。")])
    agent = OnsenGeezerAgent(tools, client=client)
    result = agent.ask("硫酸塩泉について教えて")
    assert [finding.text for finding in result.findings] == ["美人の湯"]
    assert result.revised is False


def test_検算を切れる(tools: OnsenOntologyTools) -> None:
    client = StubBedrockClient([_text_response("硫酸塩泉は美人の湯と呼ばれておる。")])
    agent = OnsenGeezerAgent(tools, client=client)
    result = agent.ask("硫酸塩泉について教えて", verify=False)
    assert result.findings == []


def test_指摘があれば差し戻して書き直させる(tools: OnsenOntologyTools) -> None:
    client = StubBedrockClient(
        [
            _text_response("硫酸塩泉は美人の湯と呼ばれ、pH9.9 じゃ。"),
            _text_response("硫酸塩泉の適応症は掲示基準に列挙されておる。pH は現地の掲示を見よ。"),
        ]
    )
    agent = OnsenGeezerAgent(tools, client=client)
    result = agent.ask("硫酸塩泉について教えて", revise=True)

    assert result.revised is True
    assert result.findings == [], "書き直した回答は指摘ゼロになった"
    assert "美人の湯" not in result.text
    # 差し戻しの指示が会話履歴に入り、指摘内容を含んでいる
    revision = agent.messages[-3]["content"][0]["text"]
    assert "美人の湯" in revision and "9.9" in revision
    # トークン使用量は2回分の合計
    assert result.usage["totalTokens"] == 30


def test_指摘がなければ差し戻さない(tools: OnsenOntologyTools) -> None:
    client = StubBedrockClient([_text_response("掲示基準に基づく適応症を見るとええぞ。")])
    agent = OnsenGeezerAgent(tools, client=client)
    result = agent.ask("どこがええかい", revise=True)
    assert result.revised is False
    assert result.findings == []
    assert len(client.responses) == 0


# --------------------------------------------------------------------------
# 実 Bedrock（オプトイン）
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("ONSEN_TEST_BEDROCK") != "1",
    reason="ONSEN_TEST_BEDROCK=1 のときだけ実行する（Bedrock を呼ぶ）",
)
def test_agent_against_real_bedrock(tools: OnsenOntologyTools) -> None:
    agent = OnsenGeezerAgent(tools)
    result = agent.ask("肌がガサガサなんじゃが、どこの湯がええかのう")
    assert result.text
    assert result.tool_calls, "オントロジーを引かずに答えてはならない"
