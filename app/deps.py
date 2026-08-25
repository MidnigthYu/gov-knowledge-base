"""
依赖注入容器模块
集中托管各类业务服务单例（大模型、嵌入、向量库、知识库、问答），由 lifespan 在应用启动时统一装配
依赖：ZhipuClient、ZhipuEmbeddingClient、ChromaVectorStore、KnowledgeManager、RagService
"""
from app.client.zhipu_client import ZhipuClient
from app.client.zhipu_embedding_client import ZhipuEmbeddingClient
from app.vector_store.chroma_store import ChromaVectorStore
from app.service.knowledge_manager import KnowledgeManager
from app.service.rag_service import RagService

# 全局服务实例，由 lifespan 在启动时初始化，供路由层延迟引用
llm_client: ZhipuClient = None
embedding_client: ZhipuEmbeddingClient = None
vector_store: ChromaVectorStore = None
kb_manager: KnowledgeManager = None
rag_service: RagService = None