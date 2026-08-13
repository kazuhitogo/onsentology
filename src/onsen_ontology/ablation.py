"""オントロジーの効果検証（ablation）。

「温泉オントロジーを作った意味はあったのか」を測るための比較実験である。同じ相談を
複数の条件に投げ、**同じ計器で採点する**。計器は :mod:`onsen_ontology.verify`（回答の検算）と、
一次情報から書き起こした事実チェックの2つ。

Phase 6 では「知識を与えるか否か」を比べた（A / A+ / B / C）。結論は「ツールは効く。
プロンプトだけでは公表値に届かない」で、これは当たり前の結果でもある。実務で比較されるべき
対抗案は「何もしない」ではなく「**同じ情報を文書として置いて検索させる**」である。
そこで Phase 7 では条件 D を足し、問いをこう立て直した。

    同じ一次情報を**トリプルに整理する価値はあるのか**。生テキストのままで足りるのか。

===== ==================================== ============== ======== ========
条件   内容                                 ツール          検算     差し戻し
===== ==================================== ============== ======== ========
A      素の LLM（人格のみ）                  なし            採点のみ  しない
B      オントロジーのツールあり              オントロジー10  採点のみ  しない
C      ツールあり＋検算を差し戻して書き直し   オントロジー10  採点あり  する
D      生テキストの文書検索（BM25）           文書検索2       採点のみ  しない
===== ==================================== ============== ======== ========

任意条件（``--conditions`` で明示すれば走る）は A+（プロンプトだけ厳しくする。Phase 6 で
役目を終えた）と E（オントロジー＋文書検索。実務での最終形）。

A の検算はツール戻り値が空なので、単位付き数値と語彙はすべて「出典なし」に倒れる。
これは計器の性質であって、それ自体を成績にしてはならない。**条件間で比較できるのは
事実チェックの合否**であり、検算の指摘件数は「回答のどこが裏取り不能か」の分布として読む。

D は出典を持てる（コーパスの全チャンクに出典URLと取得日がある）ので、数値・語彙の照合では
B・C と同じ土俵に立つ。ただし通俗表現・条文の言い換え・非法定の効能表記の判定だけは
**グラフ（法定知識）を基準にする**。生テキストに書いてあることと、法定の記述であることは違う。

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
    DOCUMENT_SEARCH_SYSTEM_PROMPT,
    DOCUMENT_TOOL_SPECS,
    GUARDRAIL_ONLY_SYSTEM_PROMPT,
    PERSONA_ONLY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TOOL_SPECS,
    OnsenGeezerAgent,
    OnsenOntologyTools,
)

# --------------------------------------------------------------------------
# 条件
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """比較条件。

    :param tool_specs: モデルに見せるツール。``None`` ならオントロジーのツール10個。
        条件 D では生テキスト検索の2つに差し替える。
    """

    id: str
    label: str
    use_tools: bool
    system_prompt: str
    revise: bool
    note: str
    tool_specs: tuple[dict[str, Any], ...] | None = None


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
    Condition(
        id="D",
        label="生テキストの文書検索（BM25）",
        use_tools=True,
        system_prompt=DOCUMENT_SEARCH_SYSTEM_PROMPT,
        revise=False,
        note=(
            "同じ一次情報をトリプルに整理せず、文章のまま BM25 で検索させる。"
            "コーパスは docs/ が出典として記録している URL 群（案A）。"
        ),
        tool_specs=tuple(DOCUMENT_TOOL_SPECS),
    ),
)

#: 既定では走らせない条件。``--conditions`` で明示すれば走る。
#: ``A+`` は Phase 6 で役目を終えた（プロンプトだけで口調と規律は半分以上改善する、という結論が出た）。
#: 保存済みの Phase 6 の結果に ``A+`` のレコードが入っているので、集計のラベルは残しておく。
#: ``E`` は実務での最終形（オントロジー＋文書検索）だが、費用と時間の都合で任意扱いとする。
OPTIONAL_CONDITIONS: tuple[Condition, ...] = (
    Condition(
        id="A+",
        label="プロンプトだけ厳しくした LLM",
        use_tools=False,
        system_prompt=GUARDRAIL_ONLY_SYSTEM_PROMPT,
        revise=False,
        note="規則は同じ強さで与えるがデータは与えない。プロンプトで代替できるかを測る。",
    ),
    Condition(
        id="E",
        label="オントロジー＋文書検索",
        use_tools=True,
        system_prompt=SYSTEM_PROMPT,
        revise=False,
        note="両方を渡す。実務での最終形。どちらを引くかはモデルに任せる。",
        tool_specs=tuple(TOOL_SPECS) + tuple(DOCUMENT_TOOL_SPECS),
    ),
)

#: 集計・選択に使う全条件。
ALL_CONDITIONS: tuple[Condition, ...] = CONDITIONS + OPTIONAL_CONDITIONS

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
    """1つの相談と、その採点条件。

    :param axis: 何を測る問なのか（照会／判定／計画／不明の申告／汚染）
    :param rag_forecast: 生テキスト検索（条件 D）がどうなるかの**事前の予想**。
        あとから都合よく解釈しないために書いておく。``○`` 取れる／``△`` 半分／``×`` 取れない
    """

    id: str
    axis: str
    text: str
    ground_truth: str
    checks: tuple[Check, ...]
    rag_forecast: str = ""


QUESTIONS: tuple[Question, ...] = (
    Question(
        id="Q1",
        axis="照会: 源泉の公表値",
        rag_forecast="○ 御座之湯のページに書いてある",
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
        axis="照会: 源泉ごとの粒度",
        rag_forecast="△ 3ページに分かれており、観光協会は温泉地単位で 2.1 と書いている",
        text="草津は源泉ごとに湯が違うと聞いた。湯畑・万代・煮川で pH はどう違うんじゃ",
        ground_truth=(
            "湯畑源泉 pH2.08（御座之湯）、万代源泉 pH1.7（御座之湯・西の河原）、煮川源泉 pH2.1（大滝乃湯）。"
            "「草津温泉の pH」という単一の値は存在しない。観光協会のページは温泉地単位で"
            "「pH値はナント2.1（湯畑源泉）」と書いており、施設の掲示値と一致しない。"
        ),
        checks=(
            Check(
                label="万代源泉の 1.7 を答える",
                kind="expect",
                pattern=r"1\.7",
                basis="御座之湯・西の河原露天風呂 公式（万代源泉 pH1.7）",
            ),
            Check(
                label="煮川源泉の 2.1 を答える",
                kind="expect",
                pattern=r"2\.1",
                basis="大滝乃湯 公式（煮川源泉 pH2.1）",
            ),
            Check(
                label="湯畑源泉の 2.08 を答える",
                kind="expect",
                pattern=r"2\.0[58]",
                basis="御座之湯 公式（湯畑源泉 pH2.08）",
            ),
            Check(
                label="温泉地単位の pH を単一の値として述べない",
                kind="forbid",
                pattern=r"草津(温泉)?の\s*pH\s*(は|が)\s*(約|およそ)?[0-9]",
                basis=(
                    "草津は源泉ごとに pH が違うので温泉地単位の値は存在しない"
                    "（観光協会のページは温泉地単位で 2.1 と書いている）"
                ),
            ),
        ),
    ),
    Question(
        id="Q3",
        axis="照会: 自主表示の検証",
        rag_forecast="△〜○ 金の湯の掲示表が1ページに載っているが、放流式と加水が別の行にある",
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
                basis="有馬温泉 金の湯の掲示（加水あり。理由: 温度供給量の不足を補うため）",
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
    Question(
        id="Q4",
        axis="判定: 法定の言い回し",
        rag_forecast="○ 掲示基準の条文が PDF にそのまま入っている",
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
        id="Q5",
        axis="判定: 数量と条文の結合",
        rag_forecast="○〜△ 掲示基準の条文と施設の38倍希釈が別文書にある",
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
        id="Q6",
        axis="判定: 施設→源泉→泉質→適応症の連鎖",
        rag_forecast="× 施設ページの効能書き（旧表記・通俗表現つき）をそのまま引く",
        text="草津の大滝乃湯は、どんな症状に向くと掲示基準では言っておるんじゃ",
        ground_truth=(
            "大滝乃湯は煮川源泉（掲示泉質名「酸性硫黄泉」）＝酸性泉かつ硫黄泉。"
            "掲示基準 3.(1) の泉質別浴用適応症は、酸性泉がアトピー性皮膚炎・尋常性乾癬・"
            "耐糖能異常（糖尿病）・表皮化膿症、硫黄泉がアトピー性皮膚炎・尋常性乾癬・慢性湿疹・表皮化膿症。"
            "一方、大滝乃湯の公式ページの効能書きは「神経痛、関節痛、うちみ、ねんざ、やけど、"
            "慢性消化器病、病後回復期、美肌、慢性婦人病」で、うちみ・ねんざ・やけど・慢性消化器病・"
            "慢性婦人病・美肌は現行の掲示基準の適応症一覧に無い。同ページは「美人の湯」も掲げている。"
        ),
        checks=(
            Check(
                label="掲示基準の泉質別適応症の病名を挙げる",
                kind="expect",
                pattern=r"(アトピー性皮膚炎|尋常性乾癬|表皮化膿症|慢性湿疹|耐糖能異常)",
                basis="掲示基準 3.(1) 酸性泉・硫黄泉の浴用適応症",
            ),
            Check(
                label="掲示基準に基づく区分であることを明示する",
                kind="expect",
                pattern=r"掲示基準|環境省",
                basis="温泉爺の規則2（適応症に触れるときは掲示基準に基づくと明示する）",
            ),
            Check(
                label="現行の掲示基準に無い効能表記を適応症として挙げない",
                kind="forbid",
                pattern=(
                    r"(うちみ|くじき|慢性婦人病|慢性消化器病|やけど|虚弱児童)"
                    r"(?!.{0,60}(無い|ない|載っておらん|現行|改訂|掲示基準では))"
                ),
                basis="現行の掲示基準の適応症一覧に無い（onsen:NonStatutoryIndication）",
            ),
        ),
    ),
    Question(
        id="Q7",
        axis="判定: 泉質間の順序関係",
        rag_forecast="× 対称性・順序関係は文書に書かれていない（積善館の記述が唯一の手掛かり）",
        text="酸性の強い湯に入ったあと、仕上げにはどんな湯がええんじゃ",
        ground_truth=(
            "オントロジーは酸性泉・硫黄泉の後の仕上げ湯として単純温泉・炭酸水素塩泉・"
            "硫酸塩泉・塩化物泉を推奨する（onsen:recommendedAfter）。これは法令ではなく"
            "本オントロジー独自の経験則で、四万温泉 積善館の「草津の仕上げ湯」という公式記述と、"
            "塩化物泉・炭酸水素塩泉・硫酸塩泉の適応症に「皮膚乾燥症」が含まれることを支持根拠とする。"
            "硫黄泉は酸性泉と連続利用が非推奨（どちらも刺激が強い）。"
        ),
        checks=(
            Check(
                label="仕上げ湯にふさわしい泉質を挙げる",
                kind="expect",
                pattern=r"(単純温泉|炭酸水素塩泉|硫酸塩泉|塩化物泉)",
                basis="onsen:recommendedAfter（酸性泉・硫黄泉 → 単純温泉/炭酸水素塩泉/硫酸塩泉/塩化物泉）",
            ),
            Check(
                label="法令ではなく経験則だと前置きする",
                kind="expect",
                pattern=r"(経験則|法令ではな|法令に根拠|掲示基準には(無|な)い|わしの)",
                basis="recommendedAfter は onsen:heuristicRule 注記つき（温泉爺の規則5）",
            ),
            Check(
                label="刺激の強い泉質を仕上げ湯として勧めない",
                kind="forbid",
                # 「酸性泉・硫黄泉の後の仕上げ湯として推奨される」という**正しい**説明を
                # 落とさないよう、勧める形（「仕上げは硫黄泉」）に限って検出する。
                pattern=r"仕上げ(の湯|湯)?\s*(に|には|は)\s*[^。]{0,8}(硫黄泉|酸性泉)",
                basis="掲示基準 2.(2)①エ(ア) は酸性泉・硫黄泉を刺激の強い泉質として挙げる",
            ),
        ),
    ),
    Question(
        id="Q8",
        axis="計画: 法定プロトコルの制約充足",
        rag_forecast="△ 条文は PDF にあるが、施設の並べ方と連泊日数の判定は文書に無い",
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
        id="Q9",
        axis="計画: 閉世界の否定",
        rag_forecast="× 「宣言が無いこと」と「実施していないこと」の区別が文書からは出せない",
        text="加水も加温も循環ろ過も消毒もしとらん湯はあるかい",
        ground_truth=(
            "法定4類型すべて非実施と掲示から確認できるのは、登別温泉 さぎり湯と"
            "長湯温泉 御前湯（竹田市営）の2施設（onsen:isUnmodifiedSupply の推論値、R6）。"
            "掲示が無いことは非実施の証明ではないので、他の施設は「分からない」である。"
        ),
        checks=(
            Check(
                label="さぎり湯または御前湯を挙げる",
                kind="expect",
                pattern=r"(さぎり湯|御前湯)",
                basis="R6: 法定4類型すべて非実施 → 無加工供給（さぎり湯・御前湯）",
            ),
            Check(
                label="掲示から確認できた範囲の話だと分かる形で答える",
                kind="expect",
                pattern=r"(掲示|公表|確認|手元|データ)",
                basis="掲示が無いことは非実施の証明ではない（閉世界の否定の扱い）",
            ),
            Check(
                label="加水・消毒を実施している施設を無加工として挙げない",
                kind="forbid",
                pattern=r"(金の湯|銀の湯|御座之湯)[^。]{0,40}(加水|加温|消毒|循環)[^。]{0,20}(しておらん|していない|なし|せず)",
                basis="有馬 金の湯・銀の湯は加水・加温・消毒を実施（掲示）",
            ),
        ),
    ),
    Question(
        id="Q10",
        axis="不明の申告: 収録していない対象",
        rag_forecast="× コーパスに秋保が1件も無いのに、隣接する何かを引いて答える",
        text="秋保温泉の pH を教えてくれ",
        ground_truth=(
            "秋保温泉はオントロジーに収録しておらず、コーパスにも1件も含まれない。"
            "主要施設・仙台市・宮城県のいずれにも源泉データの公表を確認できなかった。"
            "正しい応答は「分からない」である。"
        ),
        checks=(
            Check(
                label="分からない／手元にないと言う",
                kind="expect",
                pattern=(
                    r"(分から|わから|公表|収録|手元|掲示|見当たら|登録され|入っておらん"
                    r"|データに(は)?(無|な)い|データが(無|な)い)"
                ),
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
        id="Q11",
        axis="不明の申告: 未公表の成分",
        rag_forecast="× ラドンが未公表であることは、書かれていないので検索では出てこない",
        text="玉川温泉は放射能泉なのかい。酸性泉でもあるのじゃろ",
        ground_truth=(
            "掲示泉質名は「酸性・含二酸化炭素・鉄（Ⅱ）－塩化物温泉」。名称から酸性泉・"
            "二酸化炭素泉・含鉄泉・塩化物泉と読めるが、放射能泉の判定基準であるラドン濃度は"
            "名称に現れず未公表なので分からない。酸性泉の判定基準も pH ではなく水素イオン1mg/kg以上で、"
            "その成分値も未公表である（掲示泉質名の表記からは酸性泉と読める）。"
        ),
        checks=(
            Check(
                label="放射能泉かどうかは分からないと言う",
                kind="expect",
                pattern=(
                    r"(ラドン|ラジウム|Rn)[^。]{0,40}(未公表|公表され|分から|わから|不明|(無|な)い)"
                    r"|(放射能泉[^。]{0,40}(分から|わから|不明|判断できん|判断できない|言えん))"
                ),
                basis="放射能泉の判定基準はラドン濃度（指針 第1-3表）。玉川温泉は未公表",
            ),
            Check(
                label="放射能泉だと断定しない",
                kind="forbid",
                pattern=r"放射能泉(じゃ|である|です|に該当|に当た|の一つ|のひとつ)",
                basis="ラドン濃度が未公表なので導けない",
            ),
            Check(
                label="酸性泉の判定基準が水素イオン量であることを言う",
                kind="expect",
                pattern=r"水素イオン|1\s*mg/kg",
                basis="鉱泉分析法指針 第1-3表（酸性泉: 水素イオン1mg/kg以上）",
            ),
        ),
    ),
    Question(
        id="Q12",
        axis="汚染: 通俗表現と旧表記",
        rag_forecast="× コーパスの8文書に「美肌」、1文書に「美人の湯」が載っている",
        text="美肌とデトックスができる湯はどこじゃ",
        ground_truth=(
            "「美肌」「デトックス」「美人の湯」は掲示基準の適応症に無い。対応する法定の症状名は"
            "酸性泉・硫黄泉の浴用適応症「アトピー性皮膚炎」「尋常性乾癬」「表皮化膿症」や、"
            "塩化物泉などの「皮膚乾燥症」である。コーパスには「美肌」を載せた施設ページが8件あり、"
            "生テキストを引くと出典ありとして通ってしまう。"
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
            Check(
                label="美肌・美人の湯を効能として語らない",
                kind="forbid",
                pattern=r"(美肌|美人の湯)(?!.{0,80}(無い|ない|載っ|掲示基準|法令|表現|そんな))",
                basis="掲示基準の適応症に無い（施設の公表ページには載っている）",
            ),
        ),
    ),
)

QUESTION_IDS: tuple[str, ...] = tuple(question.id for question in QUESTIONS)

#: 問セットの識別子。**Phase 6 と Phase 7 で問の中身が変わった**（8問 → 12問。同じ ``Q3`` が
#: 別の相談を指す）。保存済みの結果を現在の採点条件で採点し直す :func:`rescore` が、
#: 古い結果に新しい採点条件を当てて壊さないよう、レコードにどの問セットで測ったかを記録する。
QUESTION_SET = "phase7-12q"


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
    #: どの問セットで測ったか（:data:`QUESTION_SET`）。空文字は Phase 6 の8問セット。
    question_set: str = ""
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
            "question_set": self.question_set,
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
        tool_specs=list(condition.tool_specs) if condition.tool_specs is not None else None,
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
        question_set=QUESTION_SET,
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
    "nonstatutory_indication",
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

    labels = {condition.id: condition.label for condition in ALL_CONDITIONS}
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


def rescore(records: Iterable[Record]) -> list[Record]:
    """保存済みの回答文を、現在の採点条件でもう一度採点する。

    採点条件（:data:`QUESTIONS` の :class:`Check`）を直したときに、**過去の結果も同じコードで
    採点し直す**ためにある。回答文は保存してあるので Bedrock を呼び直す必要はない。

    実験の途中で採点条件をいじるのは、都合のよい方向に結果を動かしうる操作である。だからこそ
    「全サンプルに同じ条件を当てる」経路を用意して、条件を変えたら必ず全部を再採点する。

    ただし**問セットが違う結果には当てない**。Phase 6 の8問と Phase 7 の12問では同じ ``Q3`` が
    別の相談を指すので、当てると無関係な条件で採点してしまう。異なる問セットのレコードは
    保存時の採点をそのまま残す。
    """
    by_id = {question.id: question for question in QUESTIONS}
    updated: list[Record] = []
    for record in records:
        question = by_id.get(record.question)
        if question is not None and record.question_set == QUESTION_SET:
            record.checks = score_answer(question, record.answer)
        updated.append(record)
    return updated


def question_sets(records: Iterable[Record]) -> dict[str, int]:
    """問セットごとの件数。混ざった結果を集計していないかの確認に使う。"""
    counts: dict[str, int] = {}
    for record in records:
        key = record.question_set or "phase6-8q（問セット未記録）"
        counts[key] = counts.get(key, 0) + 1
    return counts


def default_output_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(".cache") / f"ablation-{stamp}.jsonl"


__all__ = [
    "ALL_CONDITIONS",
    "CONDITIONS",
    "CONDITION_IDS",
    "FINDING_KINDS",
    "OPTIONAL_CONDITIONS",
    "QUESTIONS",
    "QUESTION_IDS",
    "QUESTION_SET",
    "Check",
    "Condition",
    "Question",
    "Record",
    "default_output_path",
    "per_question_table",
    "question_sets",
    "read_jsonl",
    "rescore",
    "run_ablation",
    "run_one",
    "score_answer",
    "summarize",
    "write_jsonl",
]
