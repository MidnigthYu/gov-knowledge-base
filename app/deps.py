from client.zhipu_client import ZhipuClient
from client.zhipu_embedding_client import ZhipuEmbeddingClient
from vector_store.chroma_store import ChromaVectorStore
from service.knowledge_manager import KnowledgeManager
from service.rag_service import RagService

# 全局服务实例，由 lifespan 在启动时初始化
llm_client: ZhipuClient = None
embedding_client: ZhipuEmbeddingClient = None
vector_store: ChromaVectorStore = None
kb_manager: KnowledgeManager = None
rag_service: RagService = None