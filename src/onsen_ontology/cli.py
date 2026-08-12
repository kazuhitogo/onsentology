"""コマンドラインインターフェース。

``uv run onsen`` で起動する。サブコマンドは記事内でそのままコピペして試せるように、
1コマンド1機能に揃えてある。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import itinerary, queries
from .env import load_dotenv
from .graph import load_graph, load_inferred_graph
from .reasoning import SPARQL_RULES, apply_reasoning


def _print(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def cmd_stats(args: argparse.Namespace) -> int:
    """グラフの統計と推論ルールの発火数を表示する。"""
    graph = load_graph()
    base = len(graph)
    added = apply_reasoning(graph)
    rules = {rule.id: rule.label for rule in SPARQL_RULES}
    _print(
        {
            "推論前トリプル数": base,
            "推論後トリプル数": len(graph),
            "追加トリプル数": {
                ("OWL 2 RL 演繹閉包" if k == "OWL" else f"{k}: {rules.get(k, k)}"): v
                for k, v in added.items()
            },
        }
    )
    return 0


def cmd_facilities(args: argparse.Namespace) -> int:
    _print(queries.list_facilities(load_inferred_graph()))
    return 0


def cmd_facility(args: argparse.Namespace) -> int:
    result = queries.describe_facility(load_inferred_graph(), args.name)
    if result is None:
        print(f"施設が見つからない: {args.name}", file=sys.stderr)
        return 1
    _print(result)
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    result = queries.describe_spring_quality(load_inferred_graph(), args.name)
    if result is None:
        print(f"泉質が見つからない: {args.name}", file=sys.stderr)
        return 1
    _print(result)
    return 0


def cmd_symptom(args: argparse.Namespace) -> int:
    _print(queries.find_facilities_by_symptom(load_inferred_graph(), args.keyword))
    return 0


def cmd_protocol(args: argparse.Namespace) -> int:
    graph = load_inferred_graph()
    _print(
        {
            "浴用": queries.bathing_protocol(graph),
            "飲用": queries.drinking_protocol(graph),
            "一般的適応症・禁忌症": queries.general_indications(graph),
        }
    )
    return 0


def cmd_drinking(args: argparse.Namespace) -> int:
    _print(queries.evaluate_drinking_contraindications(load_inferred_graph(), args.source))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    graph = load_inferred_graph()
    if args.facilities:
        _print(
            itinerary.describe_itinerary(
                graph,
                args.facilities,
                minutes=args.minutes,
                gap_minutes=args.gap,
                adapted=args.adapted,
                consecutive_days=args.days,
                high_temperature_caution=args.high_temp_caution,
            )
        )
    else:
        _print(
            itinerary.plan_itinerary(
                graph,
                area=args.area,
                max_baths=args.max_baths,
                minutes=args.minutes,
                gap_minutes=args.gap,
                adapted=args.adapted,
                consecutive_days=args.days,
                high_temperature_caution=args.high_temp_caution,
            )
        )
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .agent import OnsenGeezerAgent, OnsenOntologyTools
    from .verify import format_findings

    agent = OnsenGeezerAgent(OnsenOntologyTools(load_inferred_graph()))
    result = agent.ask(args.question, verify=not args.no_verify, revise=args.revise)
    print(result.text)
    if not args.no_verify:
        print(f"\n--- 検算 ---\n{format_findings(result.findings)}", file=sys.stderr)
        if result.revised:
            print("  （指摘を差し戻して書き直させた後の結果）", file=sys.stderr)
    if args.show_tools:
        print("\n--- ツール呼び出し ---", file=sys.stderr)
        for log in result.tool_calls:
            print(
                f"  {log.name}({json.dumps(log.input, ensure_ascii=False)})",
                file=sys.stderr,
            )
        print(f"  turns={result.turns} usage={result.usage}", file=sys.stderr)
    return 0


def cmd_consult(args: argparse.Namespace) -> int:
    """湯治コンシェルジュとして複数ターンの相談を受ける。"""
    from .agent import OnsenGeezerAgent, OnsenOntologyTools
    from .consult import ConsultSession
    from .verify import format_findings

    agent = OnsenGeezerAgent(OnsenOntologyTools(load_inferred_graph()))
    session = ConsultSession(
        agent, auto_repair=not args.no_repair, carry_over=not args.no_carry_over
    )

    questions: list[str] = list(args.script or [])
    interactive = not questions

    while True:
        if interactive:
            try:
                question = input("\nあんたの相談を聞こう（空行で終わり）> ").strip()
            except EOFError:
                break
            if not question:
                break
        else:
            if not questions:
                break
            question = questions.pop(0)
            print(f"\n> {question}")

        turn = session.ask(question)
        print(f"\n{turn.answer}")
        print(f"\n--- 検算 ---\n{format_findings(turn.findings)}", file=sys.stderr)
        if turn.repaired:
            before = len(turn.findings_before_repair)
            print(
                f"  （裏取りして書き直させた: 指摘 {before}件 → {len(turn.findings)}件）",
                file=sys.stderr,
            )
        if args.show_tools:
            names = ", ".join(call.name for call in turn.tool_calls) or "なし"
            print(f"  ツール: {names}", file=sys.stderr)

    if session.turns:
        print("\n--- セッション集計 ---", file=sys.stderr)
        print(json.dumps(session.report(), ensure_ascii=False, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onsen",
        description="温泉オントロジー CLI（オントロジー検索・推論・温泉爺エージェント）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="グラフ統計と推論ルールの発火数").set_defaults(func=cmd_stats)
    sub.add_parser("facilities", help="全施設一覧").set_defaults(func=cmd_facilities)

    p = sub.add_parser("facility", help="施設の詳細")
    p.add_argument("name", help="施設名の一部")
    p.set_defaults(func=cmd_facility)

    p = sub.add_parser("quality", help="掲示用泉質の詳細")
    p.add_argument("name", help="泉質名。例: 酸性泉")
    p.set_defaults(func=cmd_quality)

    p = sub.add_parser("symptom", help="症状から施設を検索")
    p.add_argument("keyword", help="症状の口語表現。例: 肌がガサガサする")
    p.set_defaults(func=cmd_symptom)

    sub.add_parser("protocol", help="法定の入浴・飲用プロトコル").set_defaults(func=cmd_protocol)

    p = sub.add_parser("drinking", help="含有成分別禁忌症（飲用）の限界飲用量を計算")
    p.add_argument("source", help="源泉名の一部")
    p.set_defaults(func=cmd_drinking)

    p = sub.add_parser("plan", help="巡浴プランの生成・検証")
    p.add_argument("--area", help="温泉地名で候補を絞る")
    p.add_argument("--facilities", nargs="+", help="訪問順に施設名を指定（並べ替えずに検証する）")
    p.add_argument("--max-baths", type=int, help="1日に入る湯の数")
    p.add_argument("--minutes", type=int, default=10, help="1回の入浴時間（分）")
    p.add_argument("--gap", type=int, default=30, help="入浴の間隔（分）")
    p.add_argument("--adapted", action="store_true", help="温泉に慣れている")
    p.add_argument("--days", type=int, default=1, help="連続療養日数")
    p.add_argument(
        "--high-temp-caution",
        action="store_true",
        help="高齢者・高血圧症・心臓病・脳卒中経験者に該当する",
    )
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("ask", help="温泉爺エージェントに相談する（Bedrock を呼ぶ）")
    p.add_argument("question", help="相談内容")
    p.add_argument("--show-tools", action="store_true", help="ツール呼び出しを標準エラーに出す")
    p.add_argument(
        "--revise",
        action="store_true",
        help="検算で指摘が出たら差し戻して1回書き直させる",
    )
    p.add_argument("--no-verify", action="store_true", help="回答の検算を行わない")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("consult", help="湯治コンシェルジュとして複数ターン相談する")
    p.add_argument(
        "script",
        nargs="*",
        help="相談を順に指定する（省略すると対話モード）",
    )
    p.add_argument("--show-tools", action="store_true", help="呼んだツール名を表示する")
    p.add_argument(
        "--no-repair",
        action="store_true",
        help="未出典の語をツールで裏取りして書き直させる処理を切る",
    )
    p.add_argument("--no-carry-over", action="store_true", help="指摘の申し送りを次ターンに渡さない")
    p.set_defaults(func=cmd_consult)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
