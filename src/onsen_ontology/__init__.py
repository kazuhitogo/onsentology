"""温泉オントロジー — 温泉の法制度・分類体系を OWL/RDF でモデル化し、
SPARQL 推論と Bedrock の tool calling を組み合わせた「温泉爺エージェント」を動かすパッケージ。

免責: 本パッケージは知識表現の技術検証物であり、医療的助言ではない。
適応症・禁忌症は環境省「掲示基準」の条文内容の引用である。
"""

from .cli import main
from .graph import load_graph, load_inferred_graph
from .namespaces import OID, ONSEN

__all__ = ["OID", "ONSEN", "load_graph", "load_inferred_graph", "main"]
