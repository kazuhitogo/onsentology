"""オントロジーの効果検証（ablation）。

「温泉オントロジーを作った意味はあったのか」を測るための比較実験である。同じ相談を
4条件に投げ、**同じ計器で採点する**。計器は :mod:`onsen_ontology.verify`（回答の検算）と、
一次情報から書き起こした事実チェックの2つ。

条件は4つ。狙いは「オントロジーが効いた」ではなく「オントロジーでなければ届かなかった箇所は
どこか」を切り分けることである。特に **A+（プロンプトだけ厳しくする）** を置いたのは、
「オントロジーなど作らずプロンプトで注意させれば十分ではないか」という反論に答えるため。

===== ==================================== ====== ======== ========
条件   内容                                 ツール  検算     差し戻し
===== ==================================== ====== ======== ========
A      素の LLM（人格のみ）                  なし    採点のみ  しない
A+     プロンプトだけ厳しくした LLM          なし    採点のみ  しない
B      オントロジーのツールあり              あり    採点のみ  しない
C      ツールあり＋検算を差し戻して書き直し   あり    採点あり  する
===== ==================================== ====== ======== ========

A・A+ の検算はツール戻り値が空なので、単位付き数値と語彙はすべて「出典なし」に倒れる。
これは計器の性質であって、それ自体を成績にしてはならない。**条件間で比較できるのは
事実チェックの合否**であり、検算の指摘件数は「回答のどこが裏取り不能か」の分布として読む。

事実チェックは、一次情報（掲示基準・鉱泉分析法指針・各施設の公表値）とオントロジーの
実データから書き起こした正規表現の期待／禁止条件である。「回答に何が書かれていれば
一次情報と対応が取れるか」を人が明示した表なので、採点の根拠がレビューできる。
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agent import (
    GUARDRAIL_ONLY_SYSTEM_PROMPT,
    PERSONA_ONLY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    OnsenGeezerAgent,
    OnsenOntologyTools,
)

# --------------------------------------------------------------------------
# 条件
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """比較条件。"""

    id: str
    label: str
    use_tools: bool
    system_prompt: str
    revise: bool
    note: str


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        id="A",
        label="素の LLM（人格のみ）",
        use_tools=False,
        system_prompt=PERSONA_ONLY_SYSTEM_PROMPT,
        revise=False,
        note="オントロジーもツールも規則も与えない。LLM が温泉について何を知っているかを測る。",
    ),
    Condition(
        id="A+",
        label="プロンプトだけ厳しくした LLM",
        use_tools=False,
        system_prompt=GUARDRAIL_ONLY_SYSTEM_PROMPT,
        revise=False,
        note="規則は同じ強さで与えるがデータは与えない。プロンプトで代替できるかを測る。",
    ),
    Condition(
        id="B",
        label="オントロジーのツールあり",
        use_tools=True,
        system_prompt=SYSTEM_PROMPT,
        revise=False,
        note="tool calling でオントロジーを引く。検算はするが回答には反映しない。",
    ),
    Condition(
        id="C",
        label="ツールあり＋検算の差し戻し",
        use_tools=True,
        system_prompt=SYSTEM_PROMPT,
        revise=True,
        note="検算の指摘を温泉爺自身に差し戻して1回だけ書き直させる。",
    ),
)

CONDITION_IDS: tuple[str, ...] = tuple(condition.id for condition in CONDITIONS)


# --------------------------------------------------------------------------
# 事実チェック
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """一次情報との対応を見る1件の条件。

    :param kind: ``expect``（この表記があってほしい）または ``forbid``（あってはならない）
    :param pattern: 正規表現。回答文に対して :func:`re.search` する（``re.DOTALL``）
    :param basis: 期待の根拠（条文・公表値）。記事とレビューのために必ず書く
    """

    label: str
    kind: str
    pattern: str
    basis: str

    def evaluate(self, answer: str) -> bool:
        """満たしていれば真。"""
        found = re.search(self.pattern, answer, re.DOTALL) is not None
        return found if self.kind == "expect" else not found


@dataclass(frozen=True)
class Question:
    """1つの相談と、その採点条件。"""

    id: str
    axis: str
    text: str
    ground_truth: str
    checks: tuple[Check, ...]


QUESTIONS: tuple[Question, ...] = (
    Question(
        id="Q1",
        axis="源泉の粒度",
        text="草津の湯畑源泉の pH はいくつじゃ",
        ground_truth=(
            "御座之湯の公表値は pH2.08（観光協会は 2.1／2.05 とも表記）。"
            "草津は源泉ごとに pH が違い、万代1.7・煮川2.1 である。温泉地単位の値ではない。"
        ),
        checks=(
            Check(
                label="湯畑源泉の公表値 2.08 を答える",
                kind="expect",
                pattern=r"2\.0[58]",
                basis="御座之湯 公式（http://onsen-kusatsu.com/gozanoyu/hot-spring/）",
            ),
        ),
    ),
    Question(
        id="Q2",
        axis="定義の厳密さ",
        text="玉川温泉は酸性泉なのかい。理由も教えてくれ",
        ground_truth=(
            "掲示泉質名は「酸性・含二酸化炭素・鉄（Ⅱ）－塩化物温泉」。酸性泉の判定基準は pH ではなく"
            "水素イオン1mg/kg以上で、その成分値は未公表。ラドン濃度も未公表かつ名称に現れないので"
            "放射能泉かどうかは分からない。"
        ),
        checks=(
            Check(
                label="判定基準が水素イオン量であることを言う",
                kind="expect",
                pattern=r"水素イオン|1\s*mg/kg",
                basis="鉱泉分析法指針 第1-3表（酸性泉: 水素イオン1mg/kg以上）",
            ),
            Check(
                label="pH が判定基準であるという誤りを述べない",
                kind="forbid",
                pattern=r"pH[^。]{0,30}(以下|未満)[^。]{0,10}(だから|ため|ので|なので)[^。]{0,10}酸性泉",
                basis="液性区分（pH<3=酸性）と泉質（酸性泉）は別の指標である",
            ),
        ),
    ),
    Question(
        id="Q3",
        axis="法定の言い回し",
        text="高血圧の薬を飲んどるんじゃが、草津の湯に入ってよいかのう",
        ground_truth=(
            "掲示基準 2.(2)① は「高齢者、高血圧症若しくは心臓病の人、脳卒中を経験した人は"
            "42℃以上の高温浴は避ける」と定める。軽症高血圧は一般的適応症に挙げられている。"
        ),
        checks=(
            Check(
                label="42℃以上の高温浴を避ける規定に触れる",
                kind="expect",
                pattern=r"42",
                basis="掲示基準 2.(2)① 浴用の方法及び注意",
            ),
            Check(
                label="掲示基準に基づく話であることを明示する",
                kind="expect",
                pattern=r"掲示基準|環境省",
                basis="掲示基準 2.(1)①・2.(2)①",
            ),
            Check(
                label="医学的な断定をしない",
                kind="forbid",
                pattern=r"(血圧が下がり|血圧が下がる|治ります|治るじゃ|効果があります)",
                basis="掲示基準は医師の指導のもとで行うことが望ましいと定める",
            ),
        ),
    ),
    Question(
        id="Q4",
        axis="数量と条文",
        text="玉川温泉の湯は飲めるかい。1日どれくらいまでじゃ",
        ground_truth=(
            "掲示基準 2.(2)② は1回100〜150mL・1日200〜500mL、"
            "かつ pH3未満の温泉は真水で pH3以上に薄め1回100mLまでと定める。"
            "玉川温泉は pH1.2 で、施設は38倍に薄めて飲用に供している。"
            "成分別の限界飲用量は成分値が未公表のため計算できない。"
        ),
        checks=(
            Check(
                label="pH3未満は薄めるという規定に触れる",
                kind="expect",
                pattern=r"薄め|希釈|薄く",
                basis="掲示基準 2.(2)②（pH3未満の温泉は真水でpH3以上に薄める）",
            ),
            Check(
                label="1回100mLまでという上限を出す",
                kind="expect",
                pattern=r"100\s*(mL|ml|ミリ)",
                basis="掲示基準 2.(2)②（pH3未満は1回100mLまで）",
            ),
            Check(
                label="未公表の成分値を作らない",
                kind="forbid",
                pattern=r"(ナトリウムイオン|よう化物イオン|カリウムイオン)[^。]{0,20}[0-9]",
                basis="玉川温泉の成分値は未公表（オントロジーは計算不能と答える）",
            ),
        ),
    ),
    Question(
        id="Q5",
        axis="データが無い対象",
        text="秋保温泉の pH を教えてくれ",
        ground_truth=(
            "秋保温泉はオントロジーに収録していない。主要施設・仙台市・宮城県のいずれにも"
            "源泉データの公表を確認できなかった。正しい応答は「分からない」である。"
        ),
        checks=(
            Check(
                label="分からない／手元にないと言う",
                kind="expect",
                pattern=r"(分から|わから|公表|収録|手元|掲示を見|データが(無|な)い)",
                basis="データが無いことを欠損として持つ設計",
            ),
            Check(
                label="pH の数値を作らない",
                kind="forbid",
                pattern=r"pH\s*(は|が)?\s*[0-9]",
                basis="秋保温泉の pH は一次情報を確認できていない",
            ),
        ),
    ),
    Question(
        id="Q6",
        axis="法定プロトコル",
        text="草津で3日、1日2回で湯を回りたい。順番を組んでくれ",
        ground_truth=(
            "掲示基準 2.(2)① の浴用プロトコル: 開始後数日は1日1〜2回、1回3〜10分、"
            "入浴後は保温と30分程度の安静、42℃以上の高温浴に注意、"
            "開始後おおむね3日〜1週間で湯あたりが出ることがある。"
        ),
        checks=(
            Check(
                label="湯あたりの発現時期に触れる",
                kind="expect",
                pattern=r"湯あたり|湯当たり",
                basis="掲示基準 2.(2)① オ（開始後おおむね3日〜1週間前後）",
            ),
            Check(
                label="入浴回数または入浴時間の上限に触れる",
                kind="expect",
                pattern=r"(1日1〜2回|1〜2回|2回まで|3〜10分|10分程度|15〜20分)",
                basis="掲示基準 2.(2)①（回数・時間の上限）",
            ),
            Check(
                label="入浴後の安静に触れる",
                kind="expect",
                pattern=r"(30分|安静)",
                basis="掲示基準 2.(2)①（保温及び30分程度の安静）",
            ),
        ),
    ),
    Question(
        id="Q7",
        axis="通俗表現の誘導",
        text="美肌とデトックスができる湯はどこじゃ",
        ground_truth=(
            "「美肌」「デトックス」は掲示基準の適応症に無い。対応する法定の症状名は"
            "酸性泉の浴用適応症「アトピー性皮膚炎」「尋常性乾癬」「表皮化膿症」などである。"
        ),
        checks=(
            Check(
                label="掲示基準の症状名に翻訳する",
                kind="expect",
                pattern=r"(アトピー性皮膚炎|尋常性乾癬|表皮化膿症|皮膚乾燥症|皮膚炎)",
                basis="掲示基準 3.(1) 泉質別適応症",
            ),
            Check(
                label="デトックスを効能として語らない",
                kind="forbid",
                pattern=r"デトックス(?!.{0,80}(無い|ない|載っ|掲示基準|法令|そんな))",
                basis="掲示基準に存在しない通俗表現である",
            ),
        ),
    ),
    Question(
        id="Q8",
        axis="自主表示の検証",
        text="有馬の金の湯はかけ流しかい",
        ground_truth=(
            "金の湯は「放流式（かけ流し）」を掲示しているが、加水・加温・塩素消毒はいずれも実施。"
            "「源泉かけ流し」は法令上の定義がない自主表示である。"
        ),
        checks=(
            Check(
                label="加水の実施に触れる",
                kind="expect",
                pattern=r"加水",
                basis="有馬温泉 金の湯の掲示（加水あり）",
            ),
            Check(
                label="消毒の実施に触れる",
                kind="expect",
                pattern=r"消毒|塩素",
                basis="有馬温泉 金の湯の掲示（塩素系薬剤による消毒）",
            ),
            Check(
                label="無加工の供給だと述べない",
                kind="forbid",
                pattern=(
                    r"(源泉100|完全なかけ流し|純粋なかけ流し|正真正銘のかけ流し"
                    r"|加水も加温も[^。]{0,12}(しておらん|していない|せず|なし)"
                    r"|(加水|消毒)[^。]{0,12}(しておらん|していない|されておらん))"
                ),
                basis="法定4類型のうち3つ（加水・加温・消毒）を実施している",
            ),
        ),
    ),
)

QUESTION_IDS: tuple[str, ...] = tuple(question.id for question in QUESTIONS)


# --------------------------------------------------------------------------
# 実行
# --------------------------------------------------------------------------


@dataclass
class Record:
    """1条件×1問の結果。"""

    condition: str
    question: str
    axis: str
    answer: str
    model_id: str = ""
    tool_calls: list[str] = field(default_factory=list)
    findings: list[dict[str, str]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    revised: bool = False

    @property
    def checks_passed(self) -> int:
        return sum(1 for check in self.checks if check["passed"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "question": self.question,
            "axis": self.axis,
            "answer": self.answer,
            "model_id": self.model_id,
            "tool_calls": self.tool_calls,
            "findings": self.findings,
            "checks": self.checks,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "revised": self.revised,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Record:
        return cls(**data)


def score_answer(question: Question, answer: str) -> list[dict[str, Any]]:
    """回答を事実チェックにかける。"""
    return [
        {
            "label": check.label,
            "kind": check.kind,
            "basis": check.basis,
            "passed": check.evaluate(answer),
        }
        for check in question.checks
    ]


def run_one(
    condition: Condition,
    question: Question,
    *,
    tools: OnsenOntologyTools,
    client: Any = None,
) -> Record:
    """1条件×1問を実行する。会話履歴は持ち越さない（1問ごとに新しいエージェント）。"""
    agent = OnsenGeezerAgent(
        tools,
        client=client,
        use_tools=condition.use_tools,
        system_prompt=condition.system_prompt,
    )
    started = time.monotonic()
    result = agent.ask(question.text, verify=True, revise=condition.revise)
    latency_ms = int((time.monotonic() - started) * 1000)
    return Record(
        condition=condition.id,
        question=question.id,
        axis=question.axis,
        answer=result.text,
        # どのモデルで測ったかを必ず残す。ONSEN_BEDROCK_MODEL_ID に
        # アプリケーション推論プロファイルの ARN が入っていると、既定モデルとは違うものが
        # 使われる。記録しておかないと、あとから条件を再構成できない。
        model_id=agent.model_id,
        tool_calls=[call.name for call in result.tool_calls],
        findings=[{"kind": f.kind, "severity": f.severity, "text": f.text} for f in result.findings],
        checks=score_answer(question, result.text),
        usage=result.usage,
        latency_ms=latency_ms,
        revised=result.revised,
    )


def run_ablation(
    *,
    conditions: Sequence[Condition] = CONDITIONS,
    questions: Sequence[Question] = QUESTIONS,
    tools: OnsenOntologyTools | None = None,
    client: Any = None,
    progress: bool = False,
) -> list[Record]:
    """全条件×全問を実行する。"""
    tools = tools if tools is not None else OnsenOntologyTools()
    records: list[Record] = []
    for condition in conditions:
        for question in questions:
            if progress:
                print(f"[{condition.id}] {question.id} {question.axis} ...", file=sys.stderr, flush=True)
            records.append(run_one(condition, question, tools=tools, client=client))
    return records


# --------------------------------------------------------------------------
# 集計
# --------------------------------------------------------------------------

FINDING_KINDS: tuple[str, ...] = (
    "unsourced_quantity",
    "unsourced_term",
    "folk_expression",
    "paraphrased_term",
    "missing_disclosure",
    "missing_tool_call",
)


def summarize(records: Iterable[Record]) -> dict[str, Any]:
    """条件ごとに集計する。"""
    by_condition: dict[str, list[Record]] = {}
    models: set[str] = set()
    for record in records:
        by_condition.setdefault(record.condition, []).append(record)
        if record.model_id:
            models.add(record.model_id)

    labels = {condition.id: condition.label for condition in CONDITIONS}
    summary: dict[str, Any] = {}
    if models:
        # 条件間で同じモデルを使っていることは、比較の前提として明示する
        summary["モデル"] = sorted(models)
    for cid, group in by_condition.items():
        checks_total = sum(len(record.checks) for record in group)
        checks_passed = sum(record.checks_passed for record in group)
        kinds = dict.fromkeys(FINDING_KINDS, 0)
        for record in group:
            for finding in record.findings:
                kinds[finding["kind"]] = kinds.get(finding["kind"], 0) + 1
        summary[cid] = {
            "条件": labels.get(cid, cid),
            "問数": len(group),
            "事実チェック合格": f"{checks_passed}/{checks_total}",
            "事実チェック合格率": round(checks_passed / checks_total, 3) if checks_total else None,
            "検算の指摘": {k: v for k, v in kinds.items() if v},
            "ツール呼び出し数": sum(len(record.tool_calls) for record in group),
            "平均トークン": (
                round(sum(r.usage.get("totalTokens", 0) for r in group) / len(group))
                if group
                else 0
            ),
            "平均レイテンシ(ms)": round(sum(r.latency_ms for r in group) / len(group)) if group else 0,
            "平均文字数": round(sum(len(r.answer) for r in group) / len(group)) if group else 0,
            "書き直し": sum(1 for r in group if r.revised),
        }
    return summary


def per_question_table(records: Iterable[Record]) -> list[dict[str, Any]]:
    """問ごとに条件を横並びにした表を作る（記事用）。

    同じ問×条件が複数回ある場合（複数回の実行を合算した場合）は、合格数と検査数を足し上げる。
    1回の実行では ±1 の差がサンプリングの揺れに埋もれるので、合算して読む前提の表である。
    """
    totals: dict[str, dict[str, list[int]]] = {}
    axes: dict[str, str] = {}
    for record in records:
        axes[record.question] = record.axis
        cell = totals.setdefault(record.question, {}).setdefault(record.condition, [0, 0])
        cell[0] += record.checks_passed
        cell[1] += len(record.checks)

    order = {qid: index for index, qid in enumerate(QUESTION_IDS)}
    rows: list[dict[str, Any]] = []
    for question in sorted(totals, key=lambda q: order.get(q, 99)):
        row: dict[str, Any] = {"問": question, "軸": axes[question]}
        for condition, (passed, total) in totals[question].items():
            row[condition] = f"{passed}/{total}"
        rows.append(row)
    return rows


def write_jsonl(records: Iterable[Record], path: Path) -> Path:
    """結果を JSONL に保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[Record]:
    """保存した結果を読み戻す。"""
    with path.open(encoding="utf-8") as handle:
        return [Record.from_dict(json.loads(line)) for line in handle if line.strip()]


def default_output_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(".cache") / f"ablation-{stamp}.jsonl"


__all__ = [
    "CONDITIONS",
    "CONDITION_IDS",
    "FINDING_KINDS",
    "QUESTIONS",
    "QUESTION_IDS",
    "Check",
    "Condition",
    "Question",
    "Record",
    "default_output_path",
    "per_question_table",
    "read_jsonl",
    "run_ablation",
    "run_one",
    "score_answer",
    "summarize",
    "write_jsonl",
]
