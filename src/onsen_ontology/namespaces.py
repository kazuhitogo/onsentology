"""温泉オントロジー — 名前空間定義。"""

from rdflib import Namespace

#: スキーマ（クラス・プロパティ・区分の個体）の名前空間
ONSEN = Namespace("https://example.org/onsen#")

#: インスタンス（温泉地・源泉・施設・浴槽）の名前空間
OID = Namespace("https://example.org/onsen/id/")

__all__ = ["ONSEN", "OID"]
