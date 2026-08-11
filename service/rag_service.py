from common.logger import get_logger
from common.exceptions import LLMAPIError, VectorStoreError, EmbeddingError
from config.settings import settings

logger = get_logger(__name__)

class RagService:
    """
    RAG 问答服务
    串联向量检索、Prompt组装、大模型生成全链路，对外提供标准化问答接口
    """

    def __init__(self, vector_store, llm_client, embedding_client, top_k: int = None):
        """依赖注入初始化"""
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.top_k = top_k if top_k is not None else settings.RAG_DEFAULT_TOP_K

        # 政务场景专属系统提示词
        self.system_prompt_template = """你是专业的政务政策咨询助手，请严格遵守以下规则：
1. 所有回答必须严格基于下方提供的【参考内容】，不得编造、引申任何政策信息
2. 如果参考内容中没有相关信息，请明确回复："抱歉，暂无与该问题相关的政策信息"
3. 回答条理清晰、表述正式严谨，避免口语化表达
4. 不要在回答中提及"参考内容"、"根据资料"等表述

【参考内容】
{context}
"""
    def query(self, user_question: str, top_k: int = None,
          similarity_threshold: float = 0.0, return_sources: bool = True,
          collection_name: str = None) -> dict:
        """执行一次RAG问答"""
        # 将用户问题转为查询向量
        try:
            query_embedding = self.embedding_client.embed(user_question)
        except Exception as e:
            logger.error(f"查询文本向量化失败: {str(e)}")
            raise EmbeddingError(f"向量化失败: {str(e)}") from e

        # 执行向量相似度检索
        actual_top_k = top_k if top_k is not None else self.top_k
        try:
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=actual_top_k,
                collection_name=collection_name
            )
        except Exception as e:
            logger.error(f"向量检索执行失败, 问题: {user_question}, 错误: {str(e)}")
            raise VectorStoreError(f"检索失败: {str(e)}") from e

        # 按相似度阈值过滤结果
        filtered_results = [
            doc for doc in search_results
            if doc.get("similarity", 0) >= similarity_threshold
        ]

        if not filtered_results:
            logger.info(f"问题[{user_question}]未检索到任何匹配片段")
            return {
                "answer": "抱歉，暂无与该问题相关的政策信息。",
                "sources": [],
            "hit_count": 0
        }

        # 拼接参考上下文
        context_blocks = []
        for idx, doc in enumerate(filtered_results):
            context_blocks.append(f"片段{idx + 1}:{doc['content']}")
        context_text = "\n---\n".join(context_blocks)

        # 格式化生成带上文的系统提示词
        final_system_prompt = self.system_prompt_template.format(context=context_text)

        # 调用大模型生成答案
        try:
            self.llm_client.clear_history()
            self.llm_client.messages.append({
                "role": "system",
                "content": final_system_prompt
            })
            # 发送用户问题，获取回答
            answer = self.llm_client.chat(prompt=user_question)
        except Exception as e:
            logger.error(f"大模型生成答案失败，问题: {user_question}，错误: {str(e)}")
            raise LLMAPIError(f"答案生成失败: {str(e)}") from e

        result = {
            "answer": answer,
            "sources": filtered_results if return_sources else [],
            "hit_count": len(filtered_results)
        }

        logger.info(f"RAG问答完成，命中{result['hit_count']}条片段")
        return result