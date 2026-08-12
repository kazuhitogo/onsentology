"""`.env` の読み込み。

アカウント固有の推論プロファイル ARN やプロファイル名をコードに書かず、
リポジトリ管理外の `.env` に置くための最小実装。既に設定されている環境変数は
上書きしない（シェルの export が常に優先される）。

依存を増やさないために python-dotenv は使わない。対応する記法は
``KEY=VALUE`` と ``export KEY=VALUE``、``#`` 始まりのコメント、引用符の除去のみ。
"""

from __future__ import annotations

import os
from pathlib import Path


def find_dotenv(start: Path | None = None) -> Path | None:
    """カレントディレクトリから上に向かって `.env` を探す。"""
    current = (start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def parse_dotenv(text: str) -> dict[str, str]:
    """`.env` の本文を辞書にする。"""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """`.env` を読んで環境変数に反映する。反映した項目を返す。"""
    dotenv = path or find_dotenv()
    if dotenv is None or not dotenv.is_file():
        return {}
    applied = {}
    for key, value in parse_dotenv(dotenv.read_text(encoding="utf-8")).items():
        if key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
