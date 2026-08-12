"""温泉爺エージェント（Amazon Bedrock Converse API + tool calling）。

設計方針は「LLM に温泉知識を持たせない」こと。泉質・pH・適応症はすべてツール経由で
オントロジーから取得させ、LLM の役割は次の3つに限定する。

1. 曖昧な相談（「疲れが取れん」）から検索キーワードを抽出する
2. どのツールをどの順で呼ぶか決める
3. 返ってきた構造化データを温泉爺の口調で説明する

こうするとハルシネーションの余地が「口調」の部分にしか残らない。数値を捏造したら
ツールの戻り値と照合すれば検出できる。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from rdflib import Graph

from . import itinerary, queries
from .graph import load_inferred_graph
from .verify import Finding, revision_request, verify_answer

#: 既定のモデル ID。日本国内で推論が完結する Japan クロスリージョン推論プロファイルを指す。
#: アカウント固有のアプリケーション推論プロファイル ARN を使う場合や別モデルに変える場合は、
#: 環境変数 ONSEN_BEDROCK_MODEL_ID で上書きする（ARN をコードに埋め込まない）。
DEFAULT_MODEL_ID = "jp.anthropic.claude-sonnet-4-6"
DEFAULT_REGION = "ap-northeast-1"

SYSTEM_PROMPT = """\
あんたは「温泉爺（おんせんじい）」という人格を持つ温泉の案内人じゃ。

## 人物設定
元は分析化学の技術者。定年後に温泉地へ隠居し、生涯で1,200湯を巡った。温泉分析書を読むのが趣味。
一人称は「わし」、相手は「あんた」と呼ぶ。語尾は「〜じゃ」「〜のう」「〜じゃろ」「〜かい」を
3文に1回程度の頻度で使う。使いすぎると読みにくいので抑えること。

## 絶対に守る規則
1. 泉質・pH・源泉温度・適応症・禁忌症・湯使いは、**ツールが返した値だけ**を使うこと。
   ツールが値を返さなかった項目は「そこは現地の掲示を見んと分からんのう」と正直に言う。
   知識から補ってはならない。数値の捏造は絶対にしない。
2. 適応症・禁忌症に触れるときは、それが環境省の「掲示基準」に基づく区分であることを
   会話の中で一度は明示する。
3. 医学的な断定をしない。「治る」「効く」ではなく「掲示基準では〜が適応症に挙げられておる」と言う。
4. 相手が発熱中・急性症状・持病の急な悪化などを訴えている場合は、温泉の話をする前に
   まず医師に相談するよう伝える。掲示基準の一般的禁忌症に該当しうるからじゃ。
5. 巡浴（はしご湯）の相談には必ずツールで検証し、警告があれば伝える。
   警告には severity が付いておる。"legal" は掲示基準の条文に根拠があるもの、
   "heuristic" はわしの経験則じゃ。**この2つを混同して伝えてはならん。**
   経験則を語るときは「これはわしの経験則じゃが」と前置きすること。
6. 回答の最後に、根拠にした施設名・源泉名を挙げる。
7. 聞き返す前に、まず調べる。相談内容から検索できることが1つでもあるなら、ツールを呼んでから
   answer を組み立てること。足りない条件（入浴時間、連泊日数など）は既定値で一度試算し、
   「この前提で見たがどうじゃ」と結果を見せながら確認する。最初のターンで質問だけ返すのは避けよ。

## やらないこと
- 出典のない数値を出す
- 湯あたりを軽視して連続入浴を勧める
- 相手の症状を診断する

## 口調の例
「ほう、肌がガサガサかい。……あんた、酸性の強い湯に長く浸かったじゃろ。
掲示基準にも、酸性泉の禁忌症として挙げられておる症状がある。
わしが勧めるのは仕上げの湯じゃ。……おっと、話が長うなった。湯が冷めるわい。」
"""

#: 効果検証（ablation）用の対照条件その1。人格だけを与え、オントロジーもツールも与えない。
#: 「LLM は温泉のことをどれくらい知っているのか」を測るための素の条件である。
#: 人物設定と口調は :data:`SYSTEM_PROMPT` と同一にする（人格の差ではなく知識の差を測るため）。
#: 具体的な数値・施設名を例に含めないのは、プロンプト経由で答えを渡さないためである。
PERSONA_ONLY_SYSTEM_PROMPT = """\
あんたは「温泉爺（おんせんじい）」という人格を持つ温泉の案内人じゃ。

## 人物設定
元は分析化学の技術者。定年後に温泉地へ隠居し、生涯で1,200湯を巡った。温泉分析書を読むのが趣味。
一人称は「わし」、相手は「あんた」と呼ぶ。語尾は「〜じゃ」「〜のう」「〜じゃろ」「〜かい」を
3文に1回程度の頻度で使う。使いすぎると読みにくいので抑えること。

相手の相談に、あんたの知識で答えてやってくれ。
"""

#: 効果検証用の対照条件その2。**規則だけ**を与え、オントロジーもツールも与えない。
#: 「オントロジーなど作らずプロンプトで注意させれば十分ではないか」という主張に対する対照。
#: :data:`SYSTEM_PROMPT` の規則からツールへの言及を落とし、規律そのものは同じ強さで残している。
GUARDRAIL_ONLY_SYSTEM_PROMPT = """\
あんたは「温泉爺（おんせんじい）」という人格を持つ温泉の案内人じゃ。

## 人物設定
元は分析化学の技術者。定年後に温泉地へ隠居し、生涯で1,200湯を巡った。温泉分析書を読むのが趣味。
一人称は「わし」、相手は「あんた」と呼ぶ。語尾は「〜じゃ」「〜のう」「〜じゃろ」「〜かい」を
3文に1回程度の頻度で使う。使いすぎると読みにくいので抑えること。

## 絶対に守る規則
1. 泉質・pH・源泉温度・適応症・禁忌症・湯使いは、**出典が確認できる値だけ**を使うこと。
   確認できない項目は「そこは現地の掲示を見んと分からんのう」と正直に言う。
   うろ覚えで補ってはならない。数値の捏造は絶対にしない。
2. 適応症・禁忌症に触れるときは、それが環境省の「掲示基準」に基づく区分であることを
   会話の中で一度は明示する。条文にある表記をそのまま使い、要約した言い換えを避ける。
3. 医学的な断定をしない。「治る」「効く」ではなく「掲示基準では〜が適応症に挙げられておる」と言う。
4. 相手が発熱中・急性症状・持病の急な悪化などを訴えている場合は、温泉の話をする前に
   まず医師に相談するよう伝える。掲示基準の一般的禁忌症に該当しうるからじゃ。
5. 巡浴（はしご湯）の相談では、掲示基準の浴用プロトコル（入浴回数・入浴時間・高温浴・湯あたり）に
   照らして注意点を伝える。法令に根拠のある話と、あんた自身の経験則を混同してはならん。
   経験則を語るときは「これはわしの経験則じゃが」と前置きすること。
6. 回答の最後に、根拠にした施設名・源泉名を挙げる。
7. 「美人の湯」「デトックス」のような通俗表現は使わない。掲示基準にある症状名で言う。

## やらないこと
- 出典のない数値を出す
- 湯あたりを軽視して連続入浴を勧める
- 相手の症状を診断する
"""


# --------------------------------------------------------------------------
# ツール定義
# --------------------------------------------------------------------------


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict:
    return {
        "toolSpec": {
            "name": name,
            "description": description,
            "inputSchema": {
                "json": {"type": "object", "properties": properties, "required": required}
            },
        }
    }


TOOL_SPECS: list[dict[str, Any]] = [
    _tool(
        "search_by_symptom",
        "利用者が訴える症状や悩みの口語表現から、掲示基準の泉質別浴用適応症に一致する温泉施設を探す。"
        "「疲れが取れない」「肌荒れ」「冷え性」のような曖昧な表現をそのまま渡してよい。",
        {"keyword": {"type": "string", "description": "症状の口語表現。例: 肌がガサガサする"}},
        ["keyword"],
    ),
    _tool(
        "describe_facility",
        "温泉施設1件の詳細を返す。源泉・掲示泉質名・pH・源泉温度・液性/泉温/浸透圧区分・"
        "湯使い（加水/加温/循環ろ過/入浴剤/消毒とその理由）・飲用許可・出典URLを含む。",
        {"name": {"type": "string", "description": "施設名の一部。例: 金の湯、さぎり湯"}},
        ["name"],
    ),
    _tool(
        "describe_spring_quality",
        "掲示用泉質10種のうち1件の詳細を返す。判定基準の成分と閾値、法的根拠、浴用/飲用適応症、"
        "泉質別禁忌症、皮膚刺激の強さ、仕上げ湯としての推奨関係を含む。",
        {"name": {"type": "string", "description": "泉質名。例: 酸性泉、硫黄泉、炭酸水素塩泉"}},
        ["name"],
    ),
    _tool(
        "list_facilities",
        "オントロジーに登録されている全温泉施設の一覧（施設名・温泉地・所在地）を返す。",
        {},
        [],
    ),
    _tool(
        "plan_itinerary",
        "巡浴（はしご湯）プランを生成し、掲示基準の浴用プロトコルで検証する。"
        "刺激の強い泉質を先に、仕上げ湯を後に並べる。警告には severity（legal/heuristic）が付く。",
        {
            "area": {"type": "string", "description": "温泉地名で候補を絞る。例: 草津"},
            "max_baths": {"type": "integer", "description": "1日に入る湯の数"},
            "adapted": {
                "type": "boolean",
                "description": "温泉に慣れているか。true で回数上限2→3回、時間上限10→20分に緩む",
            },
            "minutes": {"type": "integer", "description": "1回の入浴時間（分）"},
            "gap_minutes": {"type": "integer", "description": "入浴の間隔（分）"},
            "consecutive_days": {
                "type": "integer",
                "description": "連続して温泉療養している日数。3日以上で湯あたりの警告が出る",
            },
            "high_temperature_caution": {
                "type": "boolean",
                "description": "高齢者・高血圧症・心臓病・脳卒中経験者に該当するか",
            },
        },
        [],
    ),
    _tool(
        "validate_itinerary",
        "利用者が指定した順序の巡浴プランを、並べ替えずにそのまま検証する。",
        {
            "facilities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "訪問順に並べた施設名",
            },
            "minutes": {"type": "integer", "description": "1回の入浴時間（分）"},
            "gap_minutes": {"type": "integer", "description": "入浴の間隔（分）"},
            "adapted": {"type": "boolean", "description": "温泉に慣れているか"},
            "consecutive_days": {"type": "integer", "description": "連続療養日数"},
            "high_temperature_caution": {
                "type": "boolean",
                "description": "高温浴を避けるべき人か",
            },
        },
        ["facilities"],
    ),
    _tool(
        "get_general_indications",
        "療養泉の一般的適応症（泉質を問わず共通）と、温泉の一般的禁忌症（浴用）を返す。"
        "利用者が体調不良を訴えている場合は必ずこれを確認すること。",
        {},
        [],
    ),
    _tool(
        "get_bathing_protocol",
        "掲示基準が定める浴用の方法及び注意（入浴回数・時間・温度・安静・湯あたり）を返す。",
        {},
        [],
    ),
    _tool(
        "get_drinking_protocol",
        "掲示基準・温泉利用基準が定める飲用（飲泉）の方法及び注意を返す。",
        {},
        [],
    ),
    _tool(
        "evaluate_drinking_contraindications",
        "指定した源泉について、含有成分別禁忌症（飲用）の限界飲用量を計算する。"
        "限界飲用量 =（閾値mg / 成分量mg）× 1000mL。500mL以上なら掲示不要。",
        {"source": {"type": "string", "description": "源泉名の一部。例: 1号源泉、計算例"}},
        ["source"],
    ),
]


@dataclass
class ToolCallLog:
    """ツール呼び出しの記録。回答の検証に使う。

    :param turn: 何ターン目の呼び出しか。1ターンで複数のツールを並べて呼ぶことがあるので、
        呼び出しの順序だけでは「同時に決めたのか、前の結果を見て決めたのか」が区別できない。
    """

    name: str
    input: dict[str, Any]
    output: Any
    turn: int = 0


class OnsenOntologyTools:
    """オントロジーをツールとして提供する。Bedrock に依存しないので単体テストできる。"""

    def __init__(self, graph: Graph | None = None) -> None:
        self.graph = graph if graph is not None else load_inferred_graph()

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"error": f"未知のツール: {name}"}
        try:
            return handler(**arguments)
        except TypeError as exc:
            return {"error": f"引数が不正: {exc}"}

    # -- 個別ツール ------------------------------------------------------
    def _tool_search_by_symptom(self, keyword: str) -> Any:
        return queries.find_facilities_by_symptom(self.graph, keyword)

    def _tool_describe_facility(self, name: str) -> Any:
        result = queries.describe_facility(self.graph, name)
        return result if result is not None else {"error": f"施設が見つからない: {name}"}

    def _tool_describe_spring_quality(self, name: str) -> Any:
        result = queries.describe_spring_quality(self.graph, name)
        return result if result is not None else {"error": f"泉質が見つからない: {name}"}

    def _tool_list_facilities(self) -> Any:
        return queries.list_facilities(self.graph)

    def _tool_plan_itinerary(self, **kwargs: Any) -> Any:
        return itinerary.plan_itinerary(self.graph, **kwargs)

    def _tool_validate_itinerary(self, facilities: list[str], **kwargs: Any) -> Any:
        return itinerary.describe_itinerary(self.graph, facilities, **kwargs)

    def _tool_get_general_indications(self) -> Any:
        return queries.general_indications(self.graph)

    def _tool_get_bathing_protocol(self) -> Any:
        return queries.bathing_protocol(self.graph)

    def _tool_get_drinking_protocol(self) -> Any:
        return queries.drinking_protocol(self.graph)

    def _tool_evaluate_drinking_contraindications(self, source: str) -> Any:
        return queries.evaluate_drinking_contraindications(self.graph, source)


@dataclass
class AgentResult:
    """エージェントの1ターン分の結果。"""

    text: str
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    turns: int = 0
    findings: list[Finding] = field(default_factory=list)
    revised: bool = False


class OnsenGeezerAgent:
    """温泉爺エージェント。

    Bedrock Converse API の tool calling ループを回す。会話履歴は保持するので、
    同じインスタンスに続けて ``ask`` を呼べば文脈が続く。
    """

    def __init__(
        self,
        tools: OnsenOntologyTools | None = None,
        *,
        model_id: str | None = None,
        region: str | None = None,
        max_turns: int = 8,
        client: Any = None,
        use_tools: bool = True,
        system_prompt: str | None = None,
    ) -> None:
        self.tools = tools if tools is not None else OnsenOntologyTools()
        self.model_id = model_id or os.environ.get("ONSEN_BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
        self.region = region or os.environ.get("AWS_REGION", DEFAULT_REGION)
        self.max_turns = max_turns
        self._client = client
        self.use_tools = use_tools
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.messages: list[dict[str, Any]] = []

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def ask(self, question: str, *, verify: bool = True, revise: bool = False) -> AgentResult:
        """1つの相談に答える。ツール呼び出しは自動で解決する。

        ``verify`` が真なら、回答をツール戻り値と照合した結果を
        :attr:`AgentResult.findings` に入れる（回答文自体は変えない）。
        ``revise`` も真なら、指摘があった場合に検算結果を差し戻して1回だけ書き直させる。
        """
        self.messages.append({"role": "user", "content": [{"text": question}]})
        result = self._converse_loop()

        if not verify:
            return result

        result.findings = verify_answer(
            result.text, result.tool_calls, graph=self.tools.graph, question=question
        )
        if revise and result.findings:
            self.messages.append(
                {"role": "user", "content": [{"text": revision_request(result.findings)}]}
            )
            revised = self._converse_loop()
            revised.tool_calls = result.tool_calls + revised.tool_calls
            for key, value in result.usage.items():
                revised.usage[key] = revised.usage.get(key, 0) + value
            revised.turns += result.turns
            revised.revised = True
            revised.findings = verify_answer(
                revised.text, revised.tool_calls, graph=self.tools.graph, question=question
            )
            return revised
        return result

    def continue_with(self, message: str) -> AgentResult:
        """会話を続ける（検算はしない）。差し戻しや裏取り結果の投入に使う。"""
        self.messages.append({"role": "user", "content": [{"text": message}]})
        return self._converse_loop()

    def _converse_loop(self) -> AgentResult:
        """テキスト回答が返るまで tool calling を回す。"""
        logs: list[ToolCallLog] = []
        usage_total = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
        text_parts: list[str] = []
        turns = 0

        while turns < self.max_turns:
            turns += 1
            request: dict[str, Any] = {
                "modelId": self.model_id,
                "system": [{"text": self.system_prompt}],
                "messages": self.messages,
                "inferenceConfig": {"maxTokens": 4096, "temperature": 0.4},
            }
            # ツールを渡さない条件（効果検証の対照）では toolConfig を付けない。
            # 空の tools を渡すと Bedrock がバリデーションエラーを返すため、キー自体を落とす。
            if self.use_tools:
                request["toolConfig"] = {"tools": TOOL_SPECS}
            response = self.client.converse(**request)
            for key in usage_total:
                usage_total[key] += response.get("usage", {}).get(key, 0)

            message = response["output"]["message"]
            self.messages.append(message)

            tool_uses = [block["toolUse"] for block in message["content"] if "toolUse" in block]
            text_parts.extend(
                block["text"] for block in message["content"] if block.get("text", "").strip()
            )

            if not tool_uses:
                break

            tool_results = []
            for use in tool_uses:
                output = self.tools.call(use["name"], use.get("input") or {})
                logs.append(
                    ToolCallLog(
                        name=use["name"],
                        input=use.get("input") or {},
                        output=output,
                        turn=turns,
                    )
                )
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": use["toolUseId"],
                            "content": [
                                {"text": json.dumps(output, ensure_ascii=False, default=str)}
                            ],
                        }
                    }
                )
            self.messages.append({"role": "user", "content": tool_results})

        return AgentResult(
            text="\n\n".join(text_parts).strip(),
            tool_calls=logs,
            usage=usage_total,
            turns=turns,
        )


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_REGION",
    "GUARDRAIL_ONLY_SYSTEM_PROMPT",
    "PERSONA_ONLY_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "TOOL_SPECS",
    "AgentResult",
    "Finding",
    "OnsenGeezerAgent",
    "OnsenOntologyTools",
    "ToolCallLog",
]
