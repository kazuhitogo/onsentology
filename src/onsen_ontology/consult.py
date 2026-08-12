"""湯治コンシェルジュのセッション。

第3話で作った検算レイヤ（:mod:`onsen_ontology.verify`）は、1回の相談に対して
「ツールが返していない記述」を指摘するところまでだった。実戦で通しの相談に使うと、
それだけでは足りないことが3つ出てくる。

1. **同じ轍を踏む。** 前のターンで「美人の湯」を指摘されても、次のターンでまた言う。
   → 指摘を申し送り（:func:`verify.carry_over_note`）として次のターンに渡す。
2. **削らせるのは惜しい。** 「大滝乃湯もええぞ」と言ってツールを引いていないなら、
   消させるより**引かせた**ほうがよい。指摘に付いた ``hint`` のツールを代わりに呼び、
   結果を渡して書き直させる。
3. **良くなったのか分からない。** ターンごとの指摘件数を記録して比べられるようにする。

セッションは Bedrock を呼ぶ層（:class:`onsen_ontology.agent.OnsenGeezerAgent`）に依存するが、
判断はすべてこのモジュール側にあるので、エージェントをスタブに差し替えれば単体でテストできる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .verify import (
    Finding,
    carry_over_note,
    repair_hints,
    revision_request,
    summarize,
    verify_answer,
)

if TYPE_CHECKING:
    from .agent import OnsenGeezerAgent, ToolCallLog

#: 自動裏取りで1ターンに呼ぶツールの上限。指摘が多いときに際限なく呼ばないため。
MAX_REPAIR_CALLS = 4


@dataclass
class Turn:
    """相談1往復の記録。"""

    question: str
    answer: str
    findings: list[Finding] = field(default_factory=list)
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    #: 検算 → 裏取り → 書き直しを行ったか
    repaired: bool = False
    #: 書き直し前の指摘（``repaired`` が真のときだけ入る）
    findings_before_repair: list[Finding] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        result = {
            "相談": self.question,
            "ツール呼び出し": [call.name for call in self.tool_calls],
            "検算": summarize(self.findings),
        }
        if self.repaired:
            result["裏取り前の検算"] = summarize(self.findings_before_repair)
        return result


class ConsultSession:
    """湯治コンシェルジュの相談セッション。

    :param agent: 温泉爺エージェント。会話履歴は agent 側が持つ。
    :param auto_repair: 指摘に ``hint`` があれば、そのツールを呼んでから書き直させる。
    :param carry_over: 前のターンまでの指摘を申し送りとして次の相談に添える。
    """

    def __init__(
        self,
        agent: OnsenGeezerAgent,
        *,
        auto_repair: bool = True,
        carry_over: bool = True,
    ) -> None:
        self.agent = agent
        self.auto_repair = auto_repair
        self.carry_over = carry_over
        self.turns: list[Turn] = []

    # -- 内部 ------------------------------------------------------------
    @property
    def _past_findings(self) -> list[Finding]:
        return [finding for turn in self.turns for finding in turn.findings]

    @property
    def _session_calls(self) -> list[ToolCallLog]:
        """これまでのターンで呼んだツールの記録。

        検算の出典の範囲を**セッション全体**にするために使う。1ターン目で
        ``describe_facility`` を引いて pH 1.7 を得たなら、3ターン目に「pH 1.7 じゃ」と
        言うのは裏の取れた記述である。単発の :meth:`OnsenGeezerAgent.ask` は
        そのターンのツール呼び出ししか見ないので、ここが相談セッション固有の判断になる。
        """
        return [call for turn in self.turns for call in turn.tool_calls]

    def _verify(self, answer: str, calls: list[ToolCallLog], question: str) -> list[Finding]:
        return verify_answer(
            answer,
            self._session_calls + calls,
            graph=self.agent.tools.graph,
            question=question,
        )

    def _compose(self, question: str) -> str:
        """相談文に申し送りを添える。"""
        if not self.carry_over:
            return question
        note = carry_over_note(self._past_findings)
        return f"{note}\n\n{question}" if note else question

    def _lookup(self, findings: list[Finding]) -> list[ToolCallLog]:
        """指摘の hint に従ってツールを代わりに呼ぶ。"""
        from .agent import ToolCallLog

        logs: list[ToolCallLog] = []
        for name, arguments in repair_hints(findings)[:MAX_REPAIR_CALLS]:
            output = self.agent.tools.call(name, dict(arguments))
            logs.append(ToolCallLog(name=name, input=dict(arguments), output=output))
        return logs

    @staticmethod
    def _repair_request(findings: list[Finding], logs: list[ToolCallLog]) -> str:
        """裏取りの結果を添えた書き直し指示。"""
        lines = [revision_request(findings), "", "検算役が代わりに引いておいた。これを使え。"]
        for log in logs:
            lines.append(f"- {log.name}({log.input}) → {log.output}")
        return "\n".join(lines)

    # -- 公開 API --------------------------------------------------------
    def ask(self, question: str) -> Turn:
        """1往復を実行する。検算と（必要なら）裏取り・書き直しまで含む。"""
        result = self.agent.ask(self._compose(question), verify=False)
        turn = Turn(
            question=question,
            answer=result.text,
            findings=self._verify(result.text, result.tool_calls, question),
            tool_calls=result.tool_calls,
        )

        hints = repair_hints(turn.findings)
        if self.auto_repair and hints:
            before = turn.findings
            logs = self._lookup(before)
            revised = self.agent.continue_with(self._repair_request(before, logs))
            turn.findings_before_repair = before
            turn.repaired = True
            turn.answer = revised.text
            turn.tool_calls = result.tool_calls + logs + revised.tool_calls
            turn.findings = self._verify(revised.text, turn.tool_calls, question)

        self.turns.append(turn)
        return turn

    def report(self) -> dict[str, Any]:
        """セッション全体の集計。記事と検証で使う。"""
        return {
            "ターン数": len(self.turns),
            "ツール呼び出し総数": sum(len(turn.tool_calls) for turn in self.turns),
            "裏取りを行ったターン": sum(1 for turn in self.turns if turn.repaired),
            "検算": summarize(self._past_findings),
            "ターンごと": [turn.summary() for turn in self.turns],
        }


__all__ = ["MAX_REPAIR_CALLS", "ConsultSession", "Turn"]
