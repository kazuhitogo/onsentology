"""効果検証ハーネス（ablation）のテスト。

Bedrock は呼ばない。事実チェックの表が一次情報どおりに書けているか、条件がツールの有無を
正しく切り替えているか、集計が壊れていないかを、スタブと合成回答で確かめる。

このハーネスは記事の結論（オントロジーに意味があったか）の根拠になるので、
**採点器そのものが正しいこと**をテストで固定しておく必要がある。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import Graph

from onsen_ontology import ablation
from onsen_ontology.agent import (
    GUARDRAIL_ONLY_SYSTEM_PROMPT,
    PERSONA_ONLY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    OnsenOntologyTools,
)


class StubBedrockClient:
    """converse の戻り値を順番に返すだけのスタブ。"""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def converse(self, **kwargs: object) -> dict:
        self.requests.append(kwargs)
        return self.responses.pop(0) if self.responses else _text_response("もう無いわい。")


def _text_response(text: str) -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }


@pytest.fixture(scope="module")
def tools(graph: Graph) -> OnsenOntologyTools:
    return OnsenOntologyTools(graph)


# --------------------------------------------------------------------------
# 条件
# --------------------------------------------------------------------------


def test_four_conditions_are_defined() -> None:
    """条件は4つ。A と A+ はツールを持たず、B と C は持つ。"""
    assert ablation.CONDITION_IDS == ("A", "A+", "B", "C")
    by_id = {c.id: c for c in ablation.CONDITIONS}
    assert by_id["A"].use_tools is False
    assert by_id["A+"].use_tools is False
    assert by_id["B"].use_tools is True
    assert by_id["C"].use_tools is True
    # 差し戻すのは C だけ
    assert [c.id for c in ablation.CONDITIONS if c.revise] == ["C"]


def test_baseline_prompts_do_not_leak_data() -> None:
    """対照条件のプロンプトに、答えになる数値や施設名を埋め込んでいない。

    プロンプト経由で答えを渡してしまうと比較が無意味になる。人格と規則だけを与える。
    """
    for prompt in (PERSONA_ONLY_SYSTEM_PROMPT, GUARDRAIL_ONLY_SYSTEM_PROMPT, SYSTEM_PROMPT):
        assert "2.08" not in prompt
        assert "湯畑" not in prompt
        assert "玉川" not in prompt
        assert "金の湯" not in prompt
    # 人格は3条件で共通（人格の差ではなく知識の差を測るため）
    persona = "元は分析化学の技術者"
    assert persona in PERSONA_ONLY_SYSTEM_PROMPT
    assert persona in GUARDRAIL_ONLY_SYSTEM_PROMPT
    assert persona in SYSTEM_PROMPT
    # A は規則を持たず、A+ は持つ
    assert "絶対に守る規則" not in PERSONA_ONLY_SYSTEM_PROMPT
    assert "絶対に守る規則" in GUARDRAIL_ONLY_SYSTEM_PROMPT


def test_no_tool_condition_sends_no_tool_config(tools: OnsenOntologyTools) -> None:
    """ツール無し条件では toolConfig を送らない。"""
    client = StubBedrockClient([_text_response("知らんのう。")])
    condition = next(c for c in ablation.CONDITIONS if c.id == "A")
    ablation.run_one(condition, ablation.QUESTIONS[0], tools=tools, client=client)
    request = client.requests[0]
    assert "toolConfig" not in request
    assert request["system"][0]["text"] == PERSONA_ONLY_SYSTEM_PROMPT


def test_tool_condition_sends_tool_config(tools: OnsenOntologyTools) -> None:
    client = StubBedrockClient([_text_response("調べたぞ。")])
    condition = next(c for c in ablation.CONDITIONS if c.id == "B")
    ablation.run_one(condition, ablation.QUESTIONS[0], tools=tools, client=client)
    assert "toolConfig" in client.requests[0]
    assert client.requests[0]["system"][0]["text"] == SYSTEM_PROMPT


# --------------------------------------------------------------------------
# 事実チェックの表
# --------------------------------------------------------------------------


def test_every_check_has_a_basis() -> None:
    """採点条件には必ず根拠（条文・公表値）を書く。根拠のない採点はしない。"""
    assert len(ablation.QUESTIONS) == 8
    for question in ablation.QUESTIONS:
        assert question.checks, question.id
        assert question.ground_truth
        for check in question.checks:
            assert check.kind in ("expect", "forbid")
            assert len(check.basis) > 10, (question.id, check.label)


def test_checks_pass_on_a_correct_answer() -> None:
    """一次情報どおりに書いた回答は満点になる。"""
    correct = {
        "Q1": "湯畑源泉は pH2.08 じゃ。草津は源泉ごとに pH が違うでな。",
        "Q2": (
            "掲示泉質名は「酸性・含二酸化炭素・鉄（Ⅱ）－塩化物温泉」じゃ。"
            "酸性泉の判定基準は水素イオン1mg/kg以上で、その成分値は公表されておらん。"
        ),
        "Q3": "掲示基準では42℃以上の高温浴は避けるよう定めておる。医師にも相談することじゃ。",
        "Q4": "pH3未満の湯は真水で薄めて、1回100mLまでじゃ。成分値は公表されておらん。",
        "Q5": "秋保温泉の pH は手元の資料に無い。現地の掲示を見んと分からんのう。",
        "Q6": (
            "開始後数日は1日1〜2回、1回3〜10分にとどめ、上がったら30分は安静にせい。"
            "42℃以上は避けよ。湯あたりは3日から1週間で出ることがある。"
        ),
        "Q7": "掲示基準の適応症で言えばアトピー性皮膚炎じゃな。デトックスなどという言葉は掲示基準に無い。",
        "Q8": "金の湯は放流式と掲示しておるが、加水も加温も塩素消毒もしておる。",
    }
    for question in ablation.QUESTIONS:
        checks = ablation.score_answer(question, correct[question.id])
        failed = [c["label"] for c in checks if not c["passed"]]
        assert not failed, (question.id, failed)


def test_checks_fail_on_a_fabricated_answer() -> None:
    """一次情報から外れた回答は落ちる。禁止条件が効いていることも確かめる。"""
    fabricated = {
        "Q1": "草津の湯は pH2 くらいじゃ。",  # 公表値でない
        "Q2": "pH1.2 未満だから酸性泉じゃ。",  # 判定基準の誤り
        "Q3": "毎日入れば血圧が下がりますぞ。",  # 医学的断定
        "Q4": "ナトリウムイオンが1200mg あるから1日500mLまでじゃ。",  # 成分値の捏造
        "Q5": "秋保温泉は pH7.8 じゃ。",  # 収録していない値
        "Q6": "好きなだけ何度でも入ってええぞ。",
        "Q7": "デトックス効果で毒が抜けるぞ。",
        "Q8": "源泉100%の完全なかけ流しじゃ。",
    }
    for question in ablation.QUESTIONS:
        checks = ablation.score_answer(question, fabricated[question.id])
        assert any(not c["passed"] for c in checks), question.id


def test_forbid_check_allows_denial() -> None:
    """通俗表現を「掲示基準に無い」と否定する言い方は通す。

    禁止したいのは効能として語ることであって、語そのものの出現ではない。
    """
    q7 = next(q for q in ablation.QUESTIONS if q.id == "Q7")
    denial = "デトックスという言葉は掲示基準に無い。皮膚乾燥症なら適応症に挙げられておる。"
    forbid = [c for c in ablation.score_answer(q7, denial) if c["kind"] == "forbid"]
    assert all(c["passed"] for c in forbid)


# --------------------------------------------------------------------------
# 集計
# --------------------------------------------------------------------------


def _record(condition: str, question: str, passed: int, total: int, **kwargs: object) -> ablation.Record:
    checks = [
        {"label": f"c{i}", "kind": "expect", "basis": "x" * 20, "passed": i < passed}
        for i in range(total)
    ]
    return ablation.Record(
        condition=condition,
        question=question,
        axis="軸",
        answer="回答",
        checks=checks,
        **kwargs,  # type: ignore[arg-type]
    )


def test_records_keep_the_model_id(tools: OnsenOntologyTools) -> None:
    """どのモデルで測ったかを記録する。

    ONSEN_BEDROCK_MODEL_ID にアプリケーション推論プロファイルの ARN が入っていると、
    既定モデルとは違うモデルで走る。記録がないと、あとから測定条件を再構成できない。
    """
    client = StubBedrockClient([_text_response("調べたぞ。")])
    condition = next(c for c in ablation.CONDITIONS if c.id == "B")
    record = ablation.run_one(
        condition, ablation.QUESTIONS[0], tools=tools, client=client
    )
    assert record.model_id
    assert record.model_id == client.requests[0]["modelId"]
    summary = ablation.summarize([record])
    assert summary["モデル"] == [record.model_id]


def test_summarize_counts_checks_and_findings() -> None:
    records = [
        _record(
            "A",
            "Q1",
            0,
            2,
            findings=[{"kind": "unsourced_quantity", "severity": "error", "text": "pH2.0"}],
            usage={"totalTokens": 100},
            latency_ms=1000,
        ),
        _record(
            "C",
            "Q1",
            2,
            2,
            tool_calls=["describe_facility"],
            findings=[],
            usage={"totalTokens": 300},
            latency_ms=3000,
            revised=True,
        ),
    ]
    summary = ablation.summarize(records)
    assert summary["A"]["事実チェック合格"] == "0/2"
    assert summary["A"]["検算の指摘"] == {"unsourced_quantity": 1}
    assert summary["C"]["事実チェック合格"] == "2/2"
    assert summary["C"]["事実チェック合格率"] == 1.0
    assert summary["C"]["ツール呼び出し数"] == 1
    assert summary["C"]["書き直し"] == 1
    assert summary["A"]["平均トークン"] == 100


def test_per_question_table_is_ordered() -> None:
    records = [_record("A", "Q3", 1, 3), _record("A", "Q1", 1, 1)]
    table = ablation.per_question_table(records)
    assert [row["問"] for row in table] == ["Q1", "Q3"]
    assert table[0]["A"] == "1/1"


def test_per_question_table_sums_repeated_runs() -> None:
    """同じ問×条件が複数回あれば足し上げる（複数回の実行を合算するため）。"""
    records = [
        _record("A", "Q1", 1, 1),
        _record("A", "Q1", 0, 1),
        _record("A", "Q1", 1, 1),
    ]
    table = ablation.per_question_table(records)
    assert table[0]["A"] == "2/3"


def test_jsonl_round_trip(tmp_path: Path) -> None:
    records = [_record("B", "Q2", 1, 2, tool_calls=["describe_facility"])]
    path = ablation.write_jsonl(records, tmp_path / "out.jsonl")
    assert json.loads(path.read_text(encoding="utf-8"))["condition"] == "B"
    restored = ablation.read_jsonl(path)
    assert restored[0].question == "Q2"
    assert restored[0].checks_passed == 1
