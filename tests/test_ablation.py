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
    DOCUMENT_SEARCH_SYSTEM_PROMPT,
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
    """既定の条件は4つ。A はツールを持たず、B・C はオントロジー、D は文書検索を持つ。"""
    assert ablation.CONDITION_IDS == ("A", "B", "C", "D")
    by_id = {c.id: c for c in ablation.CONDITIONS}
    assert by_id["A"].use_tools is False
    assert by_id["B"].use_tools is True
    assert by_id["C"].use_tools is True
    assert by_id["D"].use_tools is True
    # 差し戻すのは C だけ
    assert [c.id for c in ablation.CONDITIONS if c.revise] == ["C"]
    # B・C はオントロジーのツール（既定）、D は文書検索に差し替える
    assert by_id["B"].tool_specs is None
    assert [spec["toolSpec"]["name"] for spec in by_id["D"].tool_specs] == [
        "search_documents",
        "fetch_document",
    ]


def test_optional_conditions_are_available_but_not_default() -> None:
    """A+（Phase 6 で役目を終えた）、E（両方）、F（揃えた生ドキュメント）は明示すれば走る。

    保存済みの Phase 6 の結果には A+ のレコードが入っているので、集計のラベルは残す必要がある。
    F は Phase 7 の実測を一度回したあとに足した追試なので、既定条件には入れない
    （既定で走らせると A〜D の288サンプルと同時に測ったように見えてしまう）。
    """
    optional = {c.id for c in ablation.OPTIONAL_CONDITIONS}
    assert optional == {"A+", "E", "F"}
    assert optional.isdisjoint({c.id for c in ablation.CONDITIONS})
    assert {c.id for c in ablation.ALL_CONDITIONS} == optional | set(ablation.CONDITION_IDS)
    condition_e = next(c for c in ablation.OPTIONAL_CONDITIONS if c.id == "E")
    names = [spec["toolSpec"]["name"] for spec in condition_e.tool_specs]
    assert "describe_facility" in names
    assert "search_documents" in names


def test_aligned_document_condition_differs_only_in_the_corpus() -> None:
    """条件 F は D とツールもプロンプトも同じで、コーパスだけが違う。

    ここが崩れると「揃えた文書なら届くのか」ではなく別のものを測ってしまう。
    """
    d = next(c for c in ablation.CONDITIONS if c.id == "D")
    f = next(c for c in ablation.OPTIONAL_CONDITIONS if c.id == "F")
    assert f.system_prompt == d.system_prompt
    assert f.tool_specs == d.tool_specs
    assert f.revise == d.revise is False
    assert d.corpus_dir is None
    assert f.corpus_dir == "corpus-aligned"


def test_baseline_prompts_do_not_leak_data() -> None:
    """対照条件のプロンプトに、答えになる数値や施設名を埋め込んでいない。

    プロンプト経由で答えを渡してしまうと比較が無意味になる。人格と規則だけを与える。
    """
    for prompt in (
        PERSONA_ONLY_SYSTEM_PROMPT,
        GUARDRAIL_ONLY_SYSTEM_PROMPT,
        SYSTEM_PROMPT,
        DOCUMENT_SEARCH_SYSTEM_PROMPT,
    ):
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


def test_document_condition_sends_only_document_tools(tools: OnsenOntologyTools) -> None:
    """条件 D にはオントロジーのツールを見せない。

    生テキストのまま検索する側の条件なので、``describe_facility`` が見えていたら
    「トリプルに整理する価値」を測れなくなる。規則の文言も文書検索向けに差し替える。
    """
    client = StubBedrockClient([_text_response("文書を引いたぞ。")])
    condition = next(c for c in ablation.CONDITIONS if c.id == "D")
    ablation.run_one(condition, ablation.QUESTIONS[0], tools=tools, client=client)
    sent = [spec["toolSpec"]["name"] for spec in client.requests[0]["toolConfig"]["tools"]]
    assert sent == ["search_documents", "fetch_document"]
    assert client.requests[0]["system"][0]["text"] == DOCUMENT_SEARCH_SYSTEM_PROMPT
    # 出典を語らせる規則は残す（RAG 側が出典を言えないと比較が成立しない）
    assert "出典URL" in DOCUMENT_SEARCH_SYSTEM_PROMPT


# --------------------------------------------------------------------------
# 事実チェックの表
# --------------------------------------------------------------------------


def test_every_check_has_a_basis() -> None:
    """採点条件には必ず根拠（条文・公表値）を書く。根拠のない採点はしない。"""
    assert len(ablation.QUESTIONS) == 12
    for question in ablation.QUESTIONS:
        assert question.checks, question.id
        assert question.ground_truth
        # 生テキスト検索がどうなるかの予想を先に書いておく（あとから解釈を変えないため）
        assert question.rag_forecast, question.id
        for check in question.checks:
            assert check.kind in ("expect", "forbid")
            assert len(check.basis) > 10, (question.id, check.label)


def test_question_axes_cover_the_four_types() -> None:
    """問い合わせの4つの型（照会・判定・計画・不明の申告）＋汚染を網羅している。"""
    axes = " ".join(question.axis for question in ablation.QUESTIONS)
    for kind in ("照会", "判定", "計画", "不明の申告", "汚染"):
        assert kind in axes, kind


def test_checks_pass_on_a_correct_answer() -> None:
    """一次情報どおりに書いた回答は満点になる。"""
    correct = {
        "Q1": "湯畑源泉は pH2.08 じゃ。草津は源泉ごとに pH が違うでな。",
        "Q2": (
            "草津は源泉ごとに違うんじゃ。湯畑源泉は pH2.08、万代源泉は pH1.7、"
            "煮川源泉は pH2.1 と掲示されておる。"
        ),
        "Q3": "金の湯は放流式と掲示しておるが、加水も加温も塩素消毒もしておる。",
        "Q4": "掲示基準では42℃以上の高温浴は避けるよう定めておる。医師にも相談することじゃ。",
        "Q5": "pH3未満の湯は真水で薄めて、1回100mLまでじゃ。成分値は公表されておらん。",
        "Q6": (
            "大滝乃湯は煮川源泉、掲示泉質名は酸性硫黄泉じゃ。掲示基準の泉質別適応症では"
            "アトピー性皮膚炎や尋常性乾癬が挙げられておる。"
        ),
        "Q7": (
            "これはわしの経験則じゃが、酸性の強い湯のあとは単純温泉のような穏やかな湯で"
            "肌を落ち着かせるとええ。掲示基準に定めがあるわけではない。"
        ),
        "Q8": (
            "開始後数日は1日1〜2回、1回3〜10分にとどめ、上がったら30分は安静にせい。"
            "42℃以上は避けよ。湯あたりは3日から1週間で出ることがある。"
        ),
        "Q9": (
            "掲示から確認できるのは、登別温泉のさぎり湯と長湯温泉の御前湯じゃ。"
            "掲示が無い施設は、実施しておらんという意味にはならん。"
        ),
        "Q10": "秋保温泉の pH は手元の資料に無い。現地の掲示を見んと分からんのう。",
        "Q11": (
            "掲示泉質名は「酸性・含二酸化炭素・鉄（Ⅱ）－塩化物温泉」じゃ。"
            "酸性泉の判定基準は水素イオン1mg/kg以上でな。ラドン濃度は公表されておらんから、"
            "放射能泉かどうかは分からん。"
        ),
        "Q12": (
            "掲示基準の適応症で言えばアトピー性皮膚炎じゃな。"
            "デトックスなどという言葉は掲示基準に無い。美肌も同じで、法令の記述ではない。"
        ),
    }
    for question in ablation.QUESTIONS:
        checks = ablation.score_answer(question, correct[question.id])
        failed = [c["label"] for c in checks if not c["passed"]]
        assert not failed, (question.id, failed)


def test_checks_fail_on_a_fabricated_answer() -> None:
    """一次情報から外れた回答は落ちる。禁止条件が効いていることも確かめる。"""
    fabricated = {
        "Q1": "草津の湯は pH2 くらいじゃ。",  # 公表値でない
        "Q2": "草津温泉の pH は 2.1 じゃ。どの湯も同じでな。",  # 温泉地単位にまとめた
        "Q3": "源泉100%の完全なかけ流しじゃ。",
        "Q4": "毎日入れば血圧が下がりますぞ。",  # 医学的断定
        "Q5": "ナトリウムイオンが1200mg あるから1日500mLまでじゃ。",  # 成分値の捏造
        "Q6": "効能は神経痛、うちみ、慢性婦人病、美肌じゃな。",  # 施設ページの旧表記
        "Q7": "仕上げは硫黄泉がええぞ。",  # 刺激の強い泉質を仕上げ湯にした
        "Q8": "好きなだけ何度でも入ってええぞ。",
        "Q9": "どこもかけ流しじゃから、加水も消毒もしておらんぞ。",
        "Q10": "秋保温泉は pH7.8 じゃ。",  # 収録していない値
        "Q11": "玉川温泉は放射能泉である。ラドンがたっぷりじゃ。",
        "Q12": "デトックス効果で毒が抜けるぞ。美肌にもええ。",
    }
    for question in ablation.QUESTIONS:
        checks = ablation.score_answer(question, fabricated[question.id])
        assert any(not c["passed"] for c in checks), question.id


def test_finishing_bath_check_does_not_punish_correct_explanation() -> None:
    """「酸性泉・硫黄泉の後の仕上げ湯として推奨される」という正しい説明を落とさない。

    実測で見つかった採点器の誤検出である。最初の禁止条件は「仕上げ」の近くに「酸性泉」
    「硫黄泉」があれば落としていたので、オントロジーが返した関係をそのまま説明した回答
    （炭酸水素塩泉を勧めつつ、それが酸性泉・硫黄泉の後の仕上げ湯だと述べる）が
    不合格になっていた。禁止したいのは**刺激の強い泉質を仕上げに勧めること**である。
    """
    q7 = next(q for q in ablation.QUESTIONS if q.id == "Q7")
    correct = (
        "炭酸水素塩泉じゃな。これは酸性泉・硫黄泉の後の仕上げ湯として推奨されておる関係じゃ。"
        "ただし法令ではなく、わしの経験則の類じゃ。"
    )
    assert all(c["passed"] for c in ablation.score_answer(q7, correct))
    wrong = "仕上げは硫黄泉がええぞ。"
    forbid = [c for c in ablation.score_answer(q7, wrong) if c["kind"] == "forbid"]
    assert not any(c["passed"] for c in forbid)


def test_area_level_ph_is_treated_as_an_error() -> None:
    """草津の pH を温泉地単位で1つの値として述べたら落とす（源泉ごとに違う）。"""
    q2 = next(q for q in ablation.QUESTIONS if q.id == "Q2")
    lumped = "草津温泉の pH は 2.1 じゃ。"
    forbid = [c for c in ablation.score_answer(q2, lumped) if c["kind"] == "forbid"]
    assert not any(c["passed"] for c in forbid)
    # 「草津温泉の pH は源泉ごとに違う」は落とさない
    correct = "草津温泉の pH は源泉ごとに違う。湯畑源泉2.08、万代源泉1.7、煮川源泉2.1 じゃ。"
    assert all(c["passed"] for c in ablation.score_answer(q2, correct))


def test_forbid_check_allows_denial() -> None:
    """通俗表現を「掲示基準に無い」と否定する言い方は通す。

    禁止したいのは効能として語ることであって、語そのものの出現ではない。
    """
    q12 = next(q for q in ablation.QUESTIONS if q.id == "Q12")
    denial = (
        "デトックスという言葉は掲示基準に無い。美肌も載っておらん。"
        "皮膚乾燥症なら適応症に挙げられておる。"
    )
    forbid = [c for c in ablation.score_answer(q12, denial) if c["kind"] == "forbid"]
    assert all(c["passed"] for c in forbid)


def test_nonstatutory_wording_is_allowed_when_flagged_as_such() -> None:
    """旧い効能表記を「現行の掲示基準には無い」と断って紹介するのは通す。

    施設の掲示をそのまま伝えることは間違いではない。間違いなのは、それを現行の掲示基準の
    適応症として述べることである。
    """
    q6 = next(q for q in ablation.QUESTIONS if q.id == "Q6")
    answer = (
        "大滝乃湯の公式ページは「うちみ、慢性婦人病」と書いておるが、"
        "これは現行の掲示基準の適応症一覧には無い表記じゃ。"
        "掲示基準の泉質別適応症で言えばアトピー性皮膚炎などになる。"
    )
    assert all(c["passed"] for c in ablation.score_answer(q6, answer))


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


def test_rescore_applies_current_checks_to_saved_answers() -> None:
    """採点条件を直したら、保存済みの回答も同じ条件で採点し直せる。

    条件を変えたときに一部のサンプルだけ古い採点のまま残ると、条件間の比較が崩れる。
    """
    record = ablation.Record(
        condition="A",
        question="Q1",
        axis="照会: 源泉の公表値",
        answer="湯畑源泉は pH2.08 じゃ。",
        question_set=ablation.QUESTION_SET,
        checks=[{"label": "古い採点", "kind": "expect", "basis": "x" * 20, "passed": False}],
    )
    rescored = ablation.rescore([record])[0]
    assert [c["label"] for c in rescored.checks] == [
        c.label for c in next(q for q in ablation.QUESTIONS if q.id == "Q1").checks
    ]
    assert rescored.checks_passed == 1


def test_rescore_leaves_other_question_sets_alone() -> None:
    """問セットが違う結果には現在の採点条件を当てない。

    Phase 6 は8問、Phase 7 は12問で、同じ ``Q3`` が別の相談を指す（旧 Q3 は高血圧、
    新 Q3 は金の湯）。当ててしまうと無関係な条件で採点した数字が出る。
    """
    old = ablation.Record(
        condition="A",
        question="Q3",
        axis="法定の言い回し",
        answer="掲示基準では42℃以上の高温浴は避けるとある。",
        checks=[{"label": "旧: 42℃に触れる", "kind": "expect", "basis": "x" * 20, "passed": True}],
    )
    rescored = ablation.rescore([old])[0]
    assert [c["label"] for c in rescored.checks] == ["旧: 42℃に触れる"]
    assert ablation.question_sets([old]) == {"phase6-8q（問セット未記録）": 1}


def test_missing_data_is_accepted_in_several_wordings() -> None:
    """「分からない」の言い方は1つではない。実測で出た言い方を採点条件に含める。

    Sonnet 4.6 は「わしが持っているデータの中に見当たらなかった」と答えた。
    実質は「収録していない」と言えているので、これを不合格にするのは採点器の取りこぼしである。
    """
    q10 = next(q for q in ablation.QUESTIONS if q.id == "Q10")
    wordings = [
        "秋保温泉の pH は手元の資料に無い。",
        "わしが持っているデータの中に秋保温泉の施設は見当たらなかった。",
        "秋保温泉は登録されておらんのう。現地の掲示板を見るとよい。",
        "そこは公表されておらん。",
    ]
    for answer in wordings:
        checks = ablation.score_answer(q10, answer)
        assert all(c["passed"] for c in checks), answer
    # 数値を作ったら落ちることは変わらない
    assert not all(c["passed"] for c in ablation.score_answer(q10, "秋保温泉は pH7.8 じゃ。"))
