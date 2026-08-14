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
from .visualize import VIEWS


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


def cmd_eval(args: argparse.Namespace) -> int:
    """オントロジーの効果検証（ablation）を実行する、または保存済みの結果を集計する。"""
    from pathlib import Path

    from . import ablation
    from .agent import OnsenOntologyTools

    if args.report:
        records: list[ablation.Record] = []
        for path in args.report:
            records += ablation.read_jsonl(Path(path))
        # 採点条件を直したときに古い採点が残らないよう、常に現在の条件で採点し直す
        records = ablation.rescore(records)
        _print(
            {
                "件数": len(records),
                "実行ファイル数": len(args.report),
                "問セット": ablation.question_sets(records),
                "条件ごとの集計": ablation.summarize(records),
                "問ごとの事実チェック": ablation.per_question_table(records),
            }
        )
        return 0

    # A+ と E は既定では走らせないが、--conditions で明示すれば走る
    selected = ablation.ALL_CONDITIONS if args.conditions else ablation.CONDITIONS
    conditions = [c for c in selected if not args.conditions or c.id in args.conditions]
    questions = [q for q in ablation.QUESTIONS if not args.questions or q.id in args.questions]
    if not conditions or not questions:
        print("条件または問が空である", file=sys.stderr)
        return 1

    if args.dry_run:
        _print(
            {
                "呼び出し回数（差し戻しを除く）": len(conditions) * len(questions),
                "問セット": ablation.QUESTION_SET,
                "条件": [
                    {
                        "id": c.id,
                        "内容": c.label,
                        "ツール": (
                            [spec["toolSpec"]["name"] for spec in c.tool_specs]
                            if c.tool_specs is not None
                            else ("オントロジー10件" if c.use_tools else False)
                        ),
                        "差し戻し": c.revise,
                    }
                    for c in conditions
                ],
                "問": [
                    {
                        "id": q.id,
                        "軸": q.axis,
                        "相談": q.text,
                        "検査数": len(q.checks),
                        "RAGの予想": q.rag_forecast,
                    }
                    for q in questions
                ],
            }
        )
        return 0

    records = ablation.run_ablation(
        conditions=conditions,
        questions=questions,
        tools=OnsenOntologyTools(load_inferred_graph()),
        progress=True,
    )
    out = Path(args.out) if args.out else ablation.default_output_path()
    ablation.write_jsonl(records, out)
    print(f"結果を保存した: {out}", file=sys.stderr)
    _print(
        {
            "条件ごとの集計": ablation.summarize(records),
            "問ごとの事実チェック": ablation.per_question_table(records),
        }
    )
    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    """比較実験用のコーパス（生テキスト側）を構築・検索する。

    コーパスは ``docs/*.md`` が出典として記録している URL 群である（人が選別しない）。
    取得物は第三者の著作物なので git 管理外に置き、このコマンドで組み直せるようにしている。
    """
    from pathlib import Path

    from . import corpus as corpus_module
    from .retrieval import DocumentIndex, DocumentSearchTools

    directory = Path(args.dir)

    if args.action == "align":
        from .aligned import build_aligned_corpus

        _print(build_aligned_corpus(out_dir=args.dir if args.dir != "corpus" else "corpus-aligned"))
        return 0

    if args.action == "build":
        urls = corpus_module.source_urls()
        print(f"docs が出典として記録している URL: {len(urls)}件", file=sys.stderr)
        report = corpus_module.build_corpus(
            out_dir=directory, interval=args.interval, progress=True
        )
        _print(report.to_dict())
        return 0

    if not directory.exists():
        print(f"コーパスが無い: {directory}（uv run onsen corpus build で作る）", file=sys.stderr)
        return 1

    index = DocumentIndex.from_directory(directory)
    if args.action == "stats":
        _print(
            {
                "文書数": len(index.document_ids),
                "チャンク数": len(index),
                "総文字数": sum(len(chunk.text) for chunk in index.chunks),
            }
        )
        return 0

    if not args.query:
        print("検索語を指定する", file=sys.stderr)
        return 1
    _print(DocumentSearchTools(index).search_documents(" ".join(args.query), top_k=args.top_k))
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    """グラフの一部を図にする（DOT を書き出し、graphviz があれば画像化する）。"""
    import shutil
    import subprocess
    from pathlib import Path

    from .visualize import build_view

    try:
        dot_source = build_view(args.view, name=args.facility)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else Path(f"{args.view}.{args.format}")
    if args.format == "dot":
        out.write_text(dot_source, encoding="utf-8")
        print(f"書き出した: {out}", file=sys.stderr)
        return 0

    if shutil.which("dot") is None:
        print(
            "graphviz の dot が見つからない。DOT のまま出すか graphviz を入れること"
            "（apt install graphviz / brew install graphviz）。",
            file=sys.stderr,
        )
        return 1
    subprocess.run(  # noqa: S603 - 入力は自分が生成した DOT のみ
        ["dot", f"-T{args.format}", "-Gdpi=110", "-o", str(out)],
        input=dot_source.encode("utf-8"),
        check=True,
    )
    print(f"書き出した: {out}", file=sys.stderr)
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

    p = sub.add_parser(
        "eval",
        help="オントロジーの効果検証（ablation）を実行する（Bedrock を呼ぶ）",
    )
    p.add_argument(
        "--conditions",
        nargs="*",
        metavar="ID",
        help=(
            "実行する条件。既定は A（素のLLM）/ B（オントロジー）/ C（検算の差し戻し）/ "
            "D（生テキストの文書検索）。明示すれば A+（規則のみ）と E（両方）も走る"
        ),
    )
    p.add_argument("--questions", nargs="*", metavar="ID", help="実行する問。既定は全部（Q1〜Q12）")
    p.add_argument("--out", help="結果の JSONL 出力先。既定は .cache/ablation-<時刻>.jsonl")
    p.add_argument(
        "--report",
        nargs="+",
        metavar="PATH",
        help="保存済みの JSONL を集計するだけ（Bedrock を呼ばない）。複数指定すると合算する",
    )
    p.add_argument("--dry-run", action="store_true", help="呼び出し計画を表示するだけ")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser(
        "corpus",
        help="比較実験用のコーパスを構築・検索する（docs の出典URLを取得して Markdown 化）",
    )
    p.add_argument(
        "action",
        choices=["build", "align", "search", "stats"],
        help=(
            "build=Web から取得しなおす / align=グラフから揃えた生ドキュメントを書き出す"
            " / search=BM25で検索してみる / stats=索引の統計"
        ),
    )
    p.add_argument("query", nargs="*", help="search のときの検索語")
    p.add_argument("--dir", default="corpus", help="コーパスの置き場所（既定 corpus/）")
    p.add_argument("--top-k", type=int, default=5, help="search の件数")
    p.add_argument("--interval", type=float, default=1.0, help="取得の間隔（秒）")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("graph", help="グラフの一部を図にする（DOT / PNG / SVG）")
    p.add_argument(
        "view",
        choices=list(VIEWS),
        help="切り口。schema=スキーマ / facility=1施設のサブグラフ / quality=泉質と適応症",
    )
    p.add_argument("--facility", default="御座之湯", help="facility のときの施設名（部分一致）")
    p.add_argument("--format", default="png", choices=["dot", "png", "svg"], help="出力形式")
    p.add_argument("--out", help="出力先。既定は <view>.<format>")
    p.set_defaults(func=cmd_graph)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
