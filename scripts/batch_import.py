"""
批量文档导入命令行工具
递归加载目录下的 txt/md 文档，切分后批量向量化并写入向量库
依赖：ZhipuEmbeddingClient、ChromaVectorStore
"""
import os
import sys
import argparse
from pathlib import Path

# 定位项目根目录，注入到 Python 路径最前面
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.client.zhipu_embedding_client import ZhipuEmbeddingClient
from app.vector_store.chroma_store import ChromaVectorStore

def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """按字符长度切分文本，保留重叠上下文，避免语义断裂"""
    if len(text) <= chunk_size:
        return [text.strip()]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - chunk_overlap
    
    return chunks

def load_documents(doc_dir: str) -> list[dict]:
    """递归加载目录下所有 txt/md 文档"""
    docs = []
    doc_path = Path(doc_dir)
    if not doc_path.exists():
        raise FileNotFoundError(f"文档目录不存在: {doc_dir}")
    
    for file_path in doc_path.rglob("*"):
        if file_path.suffix.lower() not in ['.txt', '.md']:
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = split_text(content)
        for idx, chunk in enumerate(chunks):
            docs.append({
                "content": chunk,
                "metadata": {
                    "source": file_path.name,
                    "chunk_id": idx,
                    "file_path": str(file_path.relative_to(doc_path))
                }
            })
    
    return docs

def batch_import(doc_dir: str, collection_name: str = None):
    """批量入库主入口"""
    if not collection_name:
        collection_name = "gov_policy_base"

    print(f"[1/4] 初始化核心组件，集合: {collection_name}")
    # 1. 初始化嵌入客户端和向量库
    embedding_client = ZhipuEmbeddingClient()
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist=True
    )
    
    print(f"[2/4] 开始加载文档，目录: {doc_dir}")
    # 2. 加载并切分文档
    docs = load_documents(doc_dir)
    print(f"[3/4] 文档切分完成，共 {len(docs)} 个文本片段")
    
    # 3. 批量生成向量
    contents = [doc["content"] for doc in docs]
    print(f"[4/4] 正在生成向量并写入向量库，共 {len(contents)} 条...")
    embeddings = embedding_client.embed_batch(contents)

    # 4. 批量入库
    ids = [f"doc_{i}_{Path(doc['metadata']['source']).stem}" for i, doc in enumerate(docs)]
    metadatas = [doc["metadata"] for doc in docs]
    
    vector_store.add_documents(
        ids=ids,
        texts=contents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"✅ 批量入库完成，共写入 {len(docs)} 条文档")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="政务知识库批量文档入库脚本")
    parser.add_argument("--dir", type=str, default="./data/docs", help="文档目录路径")
    parser.add_argument("--collection", type=str, default=None, help="向量集合名称")
    args = parser.parse_args()
    
    batch_import(args.dir, args.collection)
