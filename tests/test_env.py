"""`.env` 読み込みのテスト。

アカウント固有の ARN を `.env` に置く運用の前提が壊れたら落ちるようにする。
"""

from __future__ import annotations

import os
from pathlib import Path

from onsen_ontology.env import find_dotenv, load_dotenv, parse_dotenv


def test_記法を解釈する() -> None:
    values = parse_dotenv(
        "\n".join(
            [
                "# コメント",
                "",
                "AWS_PROFILE=dev-vm",
                "export AWS_REGION=ap-northeast-1",
                'QUOTED="quoted value"',
                "SINGLE='single'",
                "  SPACED = spaced  ",
                "ARN=arn:aws:bedrock:ap-northeast-1:000000000000:application-inference-profile/abc",
                "不正な行",
            ]
        )
    )
    assert values == {
        "AWS_PROFILE": "dev-vm",
        "AWS_REGION": "ap-northeast-1",
        "QUOTED": "quoted value",
        "SINGLE": "single",
        "SPACED": "spaced",
        "ARN": "arn:aws:bedrock:ap-northeast-1:000000000000:application-inference-profile/abc",
    }


def test_既存の環境変数を上書きしない(tmp_path: Path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("ONSEN_TEST_A=fromfile\nONSEN_TEST_B=fromfile\n", encoding="utf-8")
    monkeypatch.setenv("ONSEN_TEST_A", "fromshell")
    monkeypatch.delenv("ONSEN_TEST_B", raising=False)

    applied = load_dotenv(dotenv)

    assert applied == {"ONSEN_TEST_B": "fromfile"}
    assert os.environ["ONSEN_TEST_A"] == "fromshell"
    assert os.environ["ONSEN_TEST_B"] == "fromfile"


def test_存在しなければ何もしない(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "missing.env") == {}


def test_親ディレクトリを辿って探す(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("ONSEN_TEST_C=1\n", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_dotenv(nested) == (tmp_path / ".env").resolve()


def test_dotenv_example_がリポジトリにある() -> None:
    root = Path(__file__).resolve().parents[1]
    example = root / ".env.example"
    assert example.is_file(), "`.env` のテンプレートは公開リポジトリに残す"
    assert "<account-id>" in example.read_text(encoding="utf-8"), "実アカウント ID を書かない"


def test_dotenv_が_git_管理外である() -> None:
    root = Path(__file__).resolve().parents[1]
    ignored = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in ignored], "`.env` は .gitignore に入れる"
