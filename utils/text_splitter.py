from typing import List, Dict
from common.logger import get_logger

logger = get_logger(__name__)

class TextSplitter:
    """固定长度+重叠区间文本分块器，优先按段落切割"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split(self, text: str) -> List[Dict]:
        """
        执行文本分块
        返回: [{ "content": str, "chunk_index": int, "start_pos": int }, ...]
        """
        if not text or not isinstance(text, str):
            return []

        text = text.strip()
        if len(text) <= self.chunk_size:
            return [{"content": text, "chunk_index": 0, "start_pos": 0}]

        # 第一步：按分隔符拆成片段
        segments = [s.strip() for s in text.split(self.separator) if s.strip()]
        if not segments:
            segments = [text]

        chunks = []
        current_chunk = ""
        current_start = 0

        for seg in segments:
            # 单个片段超长，强制字符切割
            if len(seg) > self.chunk_size:
                # 先把当前累积的块保存
                if current_chunk:
                    chunks.append({
                        "content": current_chunk,
                        "chunk_index": len(chunks),
                        "start_pos": current_start
                    })
                    current_chunk = ""
                # 强制切割超长片段
                for i in range(0, len(seg), self.chunk_size - self.chunk_overlap):
                    piece = seg[i:i + self.chunk_size]
                    chunks.append({
                        "content": piece,
                        "chunk_index": len(chunks),
                        "start_pos": text.find(piece, i)
                    })
                continue

            # 正常合并
            if not current_chunk:
                current_start = text.find(seg)
                current_chunk = seg
            else:
                temp = current_chunk + "\n" + seg
                if len(temp) <= self.chunk_size:
                    current_chunk = temp
                else:
                    # 保存当前块
                    chunks.append({
                        "content": current_chunk,
                        "chunk_index": len(chunks),
                        "start_pos": current_start
                    })
                    # 下一块保留重叠内容
                    overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current_chunk = overlap_text + seg if overlap_text else seg
                    current_start = text.find(current_chunk, current_start)

        # 处理最后剩余的块
        if current_chunk:
            chunks.append({
                "content": current_chunk,
                "chunk_index": len(chunks),
                "start_pos": current_start
            })

        return chunks