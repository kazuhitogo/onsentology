"""生テキストの文書検索（BM25）。Phase 7 の条件 D で使う。

オントロジー側（``queries.py``）の対抗馬である。同じ一次情報を**トリプルに整理せず、
文章のまま**置いて、キーワード検索で引かせる。

なぜ BM25 から始めるか
    決定的で再現でき、API キーも費用も要らない。埋め込み検索は「取れ高が変わるか」の
    頑健性チェックとして後から足せばよい。生テキストの限界は検索精度ではなく
    (1) 計算できない (2)「無い」と言えない (3) 現行と旧基準を区別できない、の3点にあると
    見ているので、検索方式を変えても結論は動かないという予想である。

日本語の分割
    形態素解析器を入れず、**文字bigram**で索引する（``rank_bm25`` も MeCab も使わない）。
    「湯畑」「源泉」のような語がそのまま bigram に落ちるので、この規模では十分に引ける。
    英数字（``pH``, ``2.08``, ``mL``）は語として切り出す。数値が落ちると
    「pH2.08 を答えられるか」という肝心の問いが検索の失敗にすり替わってしまう。

出典
    チャンクは必ず ``source_url`` と ``retrieved_at`` を持つ。オントロジー側は
    ``onsen:sourceUrl`` を返すので、片方だけ出典を言えない状態では条件の比較が成立しない。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .corpus import DEFAULT_CORPUS_DIR

#: 1チャンクの目安文字数。段落の境界で切るので前後する。
CHUNK_TARGET_CHARS = 700

#: チャンクの上限。表が続く PDF は段落境界が現れないので強制的に切る。
CHUNK_MAX_CHARS = 1200

#: 検索結果の既定件数。
DEFAULT_TOP_K = 5

#: ``fetch_document`` が1回に返す文字数。鉱泉分析法指針の PDF は数十万字あるので窓で返す。
FETCH_WINDOW_CHARS = 4000

_BM25_K1 = 1.2
_BM25_B = 0.75

#: 英数字の語（``pH``, ``2.08``, ``mL``, ``100mL`` の数値部）。
_ASCII_TOKEN = re.compile(r"[A-Za-z]+|[0-9]+(?:\.[0-9]+)?")

#: 日本語として bigram を作る文字の範囲（漢字・ひらがな・カタカナ）。
_JP_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff々〆ヶ]+")

_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def tokenize(text: str) -> list[str]:
    """BM25 用のトークン列にする。英数字は語、日本語は文字bigram。"""
    lowered = text.lower()
    tokens = _ASCII_TOKEN.findall(lowered)
    for run in _JP_RUN.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass(frozen=True)
class Chunk:
    """検索単位。出典URLと取得日を必ず持つ。"""

    chunk_id: str
    document_id: str
    source_url: str
    retrieved_at: str
    title: str
    text: str

    def to_dict(self, *, score: float | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "出典URL": self.source_url,
            "取得日": self.retrieved_at,
            "見出し": self.title,
            "本文": self.text,
        }
        if score is not None:
            data["score"] = round(score, 3)
        return data


def parse_document(text: str) -> tuple[dict[str, str], str]:
    """front matter と本文に分ける。"""
    match = _FRONT_MATTER.match(text)
    if match is None:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, text[match.end() :]


def split_chunks(
    body: str, *, target: int = CHUNK_TARGET_CHARS, maximum: int = CHUNK_MAX_CHARS
) -> list[str]:
    """本文を段落境界でチャンクに切る。

    段落が現れないまま上限に達したら切る。1つの段落が上限を超える場合（PDF の表など）は
    その段落を分割する。
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal current, size
        joined = "\n".join(current).strip()
        if joined:
            chunks.append(joined)
        current = []
        size = 0

    for paragraph in re.split(r"\n\s*\n", body):
        block = paragraph.strip()
        if not block:
            continue
        while len(block) > maximum:
            flush()
            chunks.append(block[:maximum])
            block = block[maximum:]
        if size + len(block) > maximum:
            flush()
        current.append(block)
        size += len(block) + 1
        if size >= target:
            flush()
    flush()
    return chunks


def load_chunks(corpus_dir: Path | str = DEFAULT_CORPUS_DIR) -> list[Chunk]:
    """コーパスの Markdown を読み、チャンクに切る。"""
    directory = Path(corpus_dir)
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        meta, body = parse_document(path.read_text(encoding="utf-8"))
        document_id = path.stem
        title = meta.get("title") or document_id
        for index, text in enumerate(split_chunks(body)):
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}#{index}",
                    document_id=document_id,
                    source_url=meta.get("source_url", ""),
                    retrieved_at=meta.get("retrieved_at", ""),
                    title=title,
                    text=text,
                )
            )
    return chunks


class DocumentIndex:
    """BM25 の索引。コーパスの規模（数十万字）ならインメモリで足りる。"""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._tokens: list[Counter[str]] = []
        self._lengths: list[int] = []
        document_frequency: Counter[str] = Counter()
        for chunk in chunks:
            counts = Counter(tokenize(chunk.text))
            self._tokens.append(counts)
            self._lengths.append(sum(counts.values()))
            document_frequency.update(counts.keys())
        self._df = document_frequency
        self._n = len(chunks)
        self._avg_length = (sum(self._lengths) / self._n) if self._n else 0.0

    @classmethod
    def from_directory(cls, corpus_dir: Path | str = DEFAULT_CORPUS_DIR) -> DocumentIndex:
        return cls(load_chunks(corpus_dir))

    def __len__(self) -> int:
        return self._n

    @property
    def document_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for chunk in self.chunks:
            seen.setdefault(chunk.document_id, None)
        return list(seen)

    def _idf(self, token: str) -> float:
        df = self._df.get(token, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str], index: int) -> float:
        counts = self._tokens[index]
        length = self._lengths[index]
        total = 0.0
        for token in set(query_tokens):
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            denominator = frequency + _BM25_K1 * (
                1 - _BM25_B + _BM25_B * (length / self._avg_length if self._avg_length else 1)
            )
            total += self._idf(token) * (frequency * (_BM25_K1 + 1)) / denominator
        return total

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[tuple[Chunk, float]]:
        """BM25 で上位 ``top_k`` 件を返す。該当が無ければ空リスト。"""
        tokens = tokenize(query)
        if not tokens or not self._n:
            return []
        scored = [(index, self.score(tokens, index)) for index in range(self._n)]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(self.chunks[index], score) for index, score in scored[:top_k]]

    def document_text(self, document_id: str) -> tuple[Chunk, str] | None:
        """1文書の全文（チャンクを連結したもの）を返す。"""
        head = document_id.split("#")[0]
        parts = [chunk for chunk in self.chunks if chunk.document_id == head]
        if not parts:
            return None
        return parts[0], "\n\n".join(chunk.text for chunk in parts)


class DocumentSearchTools:
    """文書検索をツールとして提供する（条件 D）。"""

    def __init__(
        self,
        index: DocumentIndex | None = None,
        *,
        corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
    ) -> None:
        self._index = index
        self._corpus_dir = corpus_dir

    @property
    def index(self) -> DocumentIndex:
        if self._index is None:
            self._index = DocumentIndex.from_directory(self._corpus_dir)
        return self._index

    def search_documents(self, query: str, top_k: int = DEFAULT_TOP_K) -> Any:
        """キーワードで検索する。見つからなければ件数0を返す（黙って隣を返さない）。"""
        hits = self.index.search(query, top_k=top_k)
        if not hits:
            return {
                "query": query,
                "件数": 0,
                "結果": [],
                "注記": "この語を含む文書は見つからなかった",
            }
        return {
            "query": query,
            "件数": len(hits),
            "結果": [chunk.to_dict(score=score) for chunk, score in hits],
        }

    def fetch_document(self, document_id: str, offset: int = 0) -> Any:
        """文書の全文を窓で返す。長い PDF は ``offset`` を進めて読む。"""
        found = self.index.document_text(document_id)
        if found is None:
            return {"error": f"文書が見つからない: {document_id}"}
        chunk, full = found
        start = max(0, offset)
        window = full[start : start + FETCH_WINDOW_CHARS]
        end = start + len(window)
        return {
            "document_id": chunk.document_id,
            "出典URL": chunk.source_url,
            "取得日": chunk.retrieved_at,
            "見出し": chunk.title,
            "総文字数": len(full),
            "offset": start,
            "本文": window,
            "続きの offset": end if end < len(full) else None,
        }


__all__ = [
    "CHUNK_MAX_CHARS",
    "CHUNK_TARGET_CHARS",
    "DEFAULT_TOP_K",
    "FETCH_WINDOW_CHARS",
    "Chunk",
    "DocumentIndex",
    "DocumentSearchTools",
    "load_chunks",
    "parse_document",
    "split_chunks",
    "tokenize",
]
