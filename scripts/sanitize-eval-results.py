#!/usr/bin/env python
"""実測結果の JSONL からアカウント固有の情報を落とす。

`ablation.Record.model_id` には、実行に使ったアプリケーション推論プロファイルの ARN が
そのまま入っている。**ARN にはアカウント ID が含まれる**ので、公開する前にモデル名へ
置き換える。どのモデルで測ったかは残さなければならない（記録がないと測定条件を
再構成できない）が、アカウント ID は残す必要がない。

置換表はプロファイル ID → モデル名。ID は ARN の末尾で、これ自体はアカウントを
特定しないが、モデル名のほうが読み手に意味がある。

べき等に書いてあるので、何度実行しても壊れない（既に置換済みなら何もしない）。

    uv run python scripts/sanitize-eval-results.py            # 置換して書き戻す
    uv run python scripts/sanitize-eval-results.py --check    # 残っていないかの確認だけ
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: アプリケーション推論プロファイル ID → 公開用のモデル名
PROFILE_TO_MODEL = {
    "flnf7r9i5pgb": "claude-haiku-4-5",
    "pbrje5prrfgi": "claude-sonnet-4-6",
}

#: 落とすべきパターン（アカウント ID を含む ARN）
_ARN = re.compile(r"arn:aws:bedrock:[\w-]+:\d{12}:application-inference-profile/(\w+)")

#: 念のため ARN の残骸が無いかも見る。**単に12桁の数字を探してはいけない**：
#: コーパスの出典URLに12桁の数字が含まれる（草津町 contents/1485254967207、
#: nifty の記事ID 231013904524 など）ので、ARN のアカウント位置（":数字12桁:"）で判定する。
_ACCOUNT = re.compile(r"arn:aws[^\"]*|:\d{12}:")

RESULTS_DIR = Path("eval-results")


def _replace(text: str) -> str:
    def sub(match: re.Match[str]) -> str:
        profile = match.group(1)
        return PROFILE_TO_MODEL.get(profile, f"application-inference-profile/{profile}")

    return _ARN.sub(sub, text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="置換せず、残っていないかだけ見る")
    parser.add_argument("--dir", default=str(RESULTS_DIR), help="対象ディレクトリ")
    args = parser.parse_args(argv)

    directory = Path(args.dir)
    changed: list[str] = []
    remaining: list[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        original = path.read_text(encoding="utf-8")
        replaced = _replace(original)
        if _ACCOUNT.search(replaced):
            remaining.append(path.name)
        if replaced != original:
            changed.append(path.name)
            if not args.check:
                path.write_text(replaced, encoding="utf-8")
                # 置換後に model_id がモデル名になっているかを1行だけ検算する
                first = json.loads(replaced.splitlines()[0])
                assert first["model_id"] in PROFILE_TO_MODEL.values(), first["model_id"]

    print(f"対象 {len(list(directory.glob('*.jsonl')))} ファイル")
    print(("要置換" if args.check else "置換した") + f": {len(changed)} ファイル")
    if remaining:
        print(f"**12桁の数字が残っている: {remaining}**", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
