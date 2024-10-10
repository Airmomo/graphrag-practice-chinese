import re
from typing import Any, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections.abc import Iterable
from typing import Any

import tiktoken
from datashaper import ProgressTicker

from graphrag.index.text_splitting import Tokenizer
from graphrag.index.verbs.text.chunk.typing import TextChunk

# CHUNK_SIZE是指在处理大型数据集时，将数据分成多个小块（chunk）时，每个小块的大小。这样做可以有效地管理内存使用，避免一次性加载过多数据导致内存溢出。在这里应根据大模型API请求的上下文 tokens 大小进行设置。
# CHUNK_OVERLAP是指在处理文本数据时，将文本分成多个小块（chunk）时，相邻块之间重叠的部分。这样做可以确保在分块处理时不会丢失重要信息，特别是在进行文本分类、实体识别等任务时，有助于提高模型的准确性和连贯性。
DEFAULT_CHUNK_SIZE = 2500  # tokens
DEFAULT_CHUNK_OVERLAP = 300  # tokens


def run(
        input: list[str], args: dict[str, Any], tick: ProgressTicker
) -> Iterable[TextChunk]:
    """切分文本"""
    tokens_per_chunk = args.get("chunk_size", DEFAULT_CHUNK_SIZE)
    chunk_overlap = args.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
    encoding_name = args.get("encoding_name", "cl100k_base")
    enc = tiktoken.get_encoding(encoding_name)

    def encode(text: str) -> list[int]:
        if not isinstance(text, str):
            text = f"{text}"
        return enc.encode(text)

    def decode(tokens: list[int]) -> str:
        return enc.decode(tokens)

    return split_text_on_tokens(
        input,
        Tokenizer(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk,
            encode=encode,
            decode=decode,
        ),
        tick,
        chunk_overlap=chunk_overlap,
        tokens_per_chunk=tokens_per_chunk
    )


def split_text_on_tokens(
        texts: list[str], enc: Tokenizer, tick: ProgressTicker, chunk_overlap, tokens_per_chunk
) -> list[TextChunk]:
    result = []
    mapped_ids = []

    for source_doc_idx, text in enumerate(texts):
        tick(1)
        mapped_ids.append((source_doc_idx, text))

    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True, is_separator_regex=True, chunk_size=tokens_per_chunk, chunk_overlap=chunk_overlap
    )

    for source_doc_idx, text in mapped_ids:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            result.append(
                TextChunk(
                    text_chunk=chunk,
                    source_doc_indices=[source_doc_idx] * len(chunk),
                    n_tokens=len(chunk),
                )
            )

    return result


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    if separator:
        if keep_separator:
            # 模式中的括号会保留结果中的分隔符。
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            r"\n\n",
            r"\n",
            r"。|！|？",
            r"\.\s|\!\s|\?\s",
            r"；|;\s",
            r"，|,\s",
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """拆分传入的文本并返回处理后的块。"""
        final_chunks = []
        # 获取适当的分隔符以使用
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [
            re.sub(r"\n{2,}", "\n", chunk.strip())
            for chunk in final_chunks
            if chunk.strip() != ""
        ]