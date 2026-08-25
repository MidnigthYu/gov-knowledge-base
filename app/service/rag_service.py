"""
RAG 问答核心服务模块
串联向量检索、重排序、Prompt 组装、大模型生成全链路，对外提供同步/流式两类标准化问答接口，内置会话管理与问题改写能力
依赖：ZhipuReranker、EmbeddingError、VectorStoreError、LLMAPIError、app.config.settings
"""
import time, json
from app.client.zhipu_reranker import ZhipuReranker
from app.common.logger import get_logger
from app.common.exceptions import LLMAPIError, VectorStoreError, EmbeddingError
from app.config.settings import settings
from typing import Optional, List, Dict

# 会话历史内存存储，键为 session_id，值为 {history, expire_at}
_session_store: Dict[str, dict] = {}
# 会话过期时间（秒），超过则视为失效并清理
_SESSION_EXPIRE_SECONDS = 30 * 60
# 多轮对话最多保留的历史轮数，控制上下文长度
_MAX_HISTORY_ROUNDS = 3

logger = get_logger(__name__)

class RagService:
    """
    RAG 问答服务类
    采用依赖注入模式初始化，支持可配置的召回量、相似度阈值、重排序开关
    提供同步 query、流式 stream_query 两类对外接口，内置会话管理与问题改写能力
    """

    def __init__(self, vector_store, llm_client, embedding_client, top_k: int = None):
        """依赖注入初始化

        Args:
            vector_store: 向量存储实例，负责相似度检索
            llm_client: 大模型客户端实例，负责答案生成
            embedding_client: 嵌入客户端实例，负责查询向量化
            top_k: 默认召回量，缺省取配置 RAG_DEFAULT_TOP_K
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.top_k = top_k if top_k is not None else settings.RAG_DEFAULT_TOP_K
        self.reranker = ZhipuReranker() if settings.RERANK_ENABLE else None

        self.system_prompt_template = """你是专业的政务政策咨询助手，请严格遵守以下规则：
1. 所有回答必须严格基于下方提供的【参考内容】，不得编造、引申任何政策信息
2. 如果参考内容中没有相关信息，请明确回复："抱歉，暂无与该问题相关的政策信息"
3. 回答条理清晰、表述正式严谨，避免口语化表达
4. 不要在回答中提及"参考内容"、"根据资料"等表述

{history_block}
【参考内容】
{context}
"""

    def _clean_expired_sessions(self) -> None:
        """清理过期会话，避免内存泄漏"""
        now = time.time()
        expired_keys = [k for k, v in _session_store.items() if v["expire_at"] < now]
        for k in expired_keys:
            del _session_store[k]

    def _get_session_history(self, session_id: Optional[str]) -> List[Dict[str, str]]:
        """获取指定会话的历史对话，自动清理过期会话并刷新有效期"""
        if not session_id:
            return []
        
        self._clean_expired_sessions()
        session = _session_store.get(session_id)
        if not session:
            return []
        
        # 每次访问自动刷新过期时间
        session["expire_at"] = time.time() + _SESSION_EXPIRE_SECONDS
        return session["history"]

    def _append_session_history(self, session_id: Optional[str], user_question: str, assistant_answer: str) -> None:
        """将本轮问答追加到会话历史，仅保留最近N轮"""
        if not session_id:
            return
        
        # setdefault 替代 check-then-create，消除 TOCTOU 隐患
        session = _session_store.setdefault(session_id, {
            "history": [],
            "expire_at": time.time() + _SESSION_EXPIRE_SECONDS
        })
        
        session["history"].append({"role": "user", "content": user_question})
        session["history"].append({"role": "assistant", "content": assistant_answer})
        
        # 只保留最近 N 轮对话（每轮2条消息）
        if len(session["history"]) > _MAX_HISTORY_ROUNDS * 2:
            session["history"] = session["history"][-_MAX_HISTORY_ROUNDS * 2:]
        
        session["expire_at"] = time.time() + _SESSION_EXPIRE_SECONDS

    def _format_history_text(self, history: List[Dict[str, str]]) -> str:
        """统一格式化历史对话文本，改写、Prompt注入三处复用"""
        if not history:
            return ""
        history_text = "【历史对话】\n"
        for msg in history:
            if msg["role"] == "user":
                history_text += f"用户：{msg['content']}\n"
            else:
                history_text += f"助手：{msg['content']}\n"
        return history_text

    def _rewrite_query(self, current_question: str, history: List[Dict[str, str]]) -> str:
        """结合对话历史改写用户问题，补全省略的指代信息，用于提升检索准确率"""
        if not history:
            return current_question
        
        # 取最近2轮对话用于改写，避免上下文过长
        recent_history = history[-4:]
        history_text = self._format_history_text(recent_history)
        
        rewrite_prompt = f"""请结合历史对话，将用户当前问题改写为一个语义完整、可独立理解的问题，补全省略的指代内容。
要求：
1. 仅输出改写后的问题本身，不要任何解释、前缀或标点符号以外的内容
2. 保留原问题的核心诉求，不新增信息
3. 若原问题本身语义完整，直接返回原问题

历史对话：
{history_text}
用户当前问题：{current_question}
改写后的问题："""

        # 备份原始消息状态，改写调用绝不污染共享客户端
        original_messages = self.llm_client.messages.copy()
        try:
            rewritten = self.llm_client.chat(rewrite_prompt)
            rewritten = rewritten.strip()
            logger.info(f"问题改写完成：原问题「{current_question}」→ 改写后「{rewritten}」")
            return rewritten if rewritten else current_question
        except Exception as e:
            logger.warning(f"问题改写失败，使用原问题检索：{str(e)}")
            return current_question
        finally:

            self.llm_client.messages = original_messages

    def query(self, user_question: str, top_k: int = None,
            similarity_threshold: float = 0.0,
            return_sources: bool = True,
            collection_name: str = None,
            enable_rerank: bool = None,
            rerank_top_n: int = None,
            session_id: Optional[str] = None) -> dict:
        """执行一次同步 RAG 问答，返回答案与来源片段

        Args:
            user_question: 用户原始问题文本
            top_k: 召回片段数量，缺省按是否启用重排序取 RECALL_TOP_K 或默认值
            similarity_threshold: 相似度过滤阈值
            return_sources: 是否返回来源片段
            collection_name: 目标知识库名称，缺省使用默认集合
            enable_rerank: 是否启用重排序，缺省取配置
            rerank_top_n: 重排序返回片段数，缺省取配置
            session_id: 会话 ID，用于多轮上下文记忆

        Returns:
            包含 answer、sources、hit_count 三个字段的结果字典

        Raises:
            EmbeddingError: 查询文本向量化失败时抛出
            VectorStoreError: 向量检索执行失败时抛出
            LLMAPIError: 大模型生成答案失败时抛出
        """
        # 获取会话历史
        history = self._get_session_history(session_id)
        # 改写问题，提升检索准确率
        search_query = self._rewrite_query(user_question, history)
        # 将改写后的问题转为查询向量
        try:
            query_embedding = self.embedding_client.embed_single(search_query)
        except Exception as e:
            logger.error(f"查询文本向量化失败: {str(e)}")
            raise EmbeddingError(f"向量化失败: {str(e)}") from e

        # 重排序参数兜底
        if enable_rerank is None:
            enable_rerank = settings.RERANK_ENABLE
        if rerank_top_n is None:
            rerank_top_n = settings.RERANK_TOP_N

        # 实际召回量
        actual_top_k = top_k
        if actual_top_k is None:
            actual_top_k = settings.RECALL_TOP_K if enable_rerank else self.top_k

        # 执行向量相似度检索
        try:
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=actual_top_k,
                similarity_threshold=similarity_threshold,
                collection_name=collection_name
            )
        except Exception as e:
            logger.error(f"向量检索执行失败, 问题: {user_question}, 错误: {str(e)}")
            raise VectorStoreError(f"检索失败: {str(e)}") from e

        filtered_results = search_results

        if not filtered_results:
            empty_answer = "抱歉，暂无与该问题相关的政策信息。"
            logger.info(f"问题[{user_question}]未检索到任何匹配片段")
            # 空结果也保存历史，与流式行为对齐，保证多轮上下文连续
            self._append_session_history(session_id, user_question, empty_answer)
            return {
                "answer": empty_answer,
                "sources": [],
                "hit_count": 0
            }

        # 重排序精排：统一使用改写后的问题计算相关性，前后语义一致
        if self.reranker and enable_rerank and filtered_results:
            original_count = len(filtered_results)
            actual_rerank_n = min(rerank_top_n, original_count)

            start_time = time.time()
            filtered_results = self.reranker.rerank(
                query=search_query,
                documents=filtered_results,
                top_n=actual_rerank_n
            )
            cost_ms = (time.time() - start_time) * 1000

            logger.info(f"重排序完成：初筛{original_count}条 → 精排{len(filtered_results)}条，耗时{cost_ms:.0f}ms")

        # 拼接参考上下文
        context_blocks = []
        for idx, doc in enumerate(filtered_results):
            context_blocks.append(f"片段{idx + 1}:{doc['content']}")
        context_text = "\n---\n".join(context_blocks)

        history_block = self._format_history_text(history)
        final_system_prompt = self.system_prompt_template.format(
            history_block=history_block,
            context=context_text
        )

        # 调用大模型生成答案
        try:
            self.llm_client.clear_history()
            self.llm_client.messages[0]["content"] = final_system_prompt
            # 发送用户问题，获取回答
            answer = self.llm_client.chat(prompt=user_question)
            # 保存本轮对话到会话历史
            self._append_session_history(session_id, user_question, answer)

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

    def prepare_query_context(
            self, 
            user_question: str, 
            top_k: int = None,
            similarity_threshold: float = 0.0,
            collection_name: str = None,
            enable_rerank: bool = None, 
            rerank_top_n: int = None,
            session_id: Optional[str] = None):
        """同步执行检索、重排、Prompt 组装，返回最终提示词与过滤后结果列表

        供流式接口复用，与 query 保持一致的检索与重排逻辑，但不调用大模型生成

        Args:
            user_question: 用户原始问题文本
            top_k: 召回片段数量，缺省按是否启用重排序取 RECALL_TOP_K 或默认值
            similarity_threshold: 相似度过滤阈值
            collection_name: 目标知识库名称，缺省使用默认集合
            enable_rerank: 是否启用重排序，缺省取配置
            rerank_top_n: 重排序返回片段数，缺省取配置
            session_id: 会话 ID，用于多轮上下文记忆

        Returns:
            (final_system_prompt, filtered_results) 二元组

        Raises:
            EmbeddingError: 查询文本向量化失败时抛出
            VectorStoreError: 向量检索执行失败时抛出
        """
        history = self._get_session_history(session_id)
        search_query = self._rewrite_query(user_question, history)
        try:
            query_embedding = self.embedding_client.embed_single(search_query)

        except Exception as e:
            logger.error(f"查询文本向量化失败：{str(e)}")
            raise EmbeddingError(f"向量化失败：{str(e)}") from e

        if enable_rerank is None:
            enable_rerank = settings.RERANK_ENABLE
        if rerank_top_n is None:
            rerank_top_n = settings.RERANK_TOP_N

        actual_top_k = top_k
        if actual_top_k is None:
            actual_top_k = settings.RECALL_TOP_K if enable_rerank else self.top_k

        try:
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=actual_top_k,
                similarity_threshold=similarity_threshold,
                collection_name=collection_name
            )
        except Exception as e:
            logger.error(f"向量检索执行失败，问题：{user_question}，错误：{str(e)}")
            raise VectorStoreError(f"检索失败：{str(e)}") from e

        filtered_results = search_results

        if self.reranker and enable_rerank and filtered_results:
            original_count = len(filtered_results)
            actual_rerank_n = min(rerank_top_n, original_count)

            start_time = time.time()
            filtered_results = self.reranker.rerank(
                query=search_query,
                documents=filtered_results,
                top_n=actual_rerank_n
            )
            cost_ms = (time.time() - start_time) * 1000

            logger.info(f"重排序完成：初筛{original_count}条 → 精排{len(filtered_results)}条，耗时{cost_ms:.0f}ms")

        # 拼接参考上下文
        context_blocks = []
        for idx, doc in enumerate(filtered_results):
            context_blocks.append(f"片段{idx + 1}:{doc['content']}")
        context_text = "\n---\n".join(context_blocks)

        history_block = self._format_history_text(history)
        final_system_prompt = self.system_prompt_template.format(
            history_block=history_block,
            context=context_text
        )

        return final_system_prompt, filtered_results

    def stream_query(
            self,
            user_question: str,
            top_k: int = None,
            similarity_threshold: float = 0.0,
            return_sources: bool = True,
            collection_name: str = None,
            enable_rerank: bool = None,
            rerank_top_n: int = None,
            session_id: Optional[str] = None):
        """流式 RAG 问答生成器，适配 SSE 前端接口

        逐块产出 content/sources/done 三类事件，内置零召回兜底与流式生成异常捕获，保证接口稳定不崩溃

        Args:
            user_question: 用户原始问题文本
            top_k: 召回片段数量，缺省按是否启用重排序取 RECALL_TOP_K 或默认值
            similarity_threshold: 相似度过滤阈值，未指定时兜底为 0.5
            return_sources: 是否返回来源片段
            collection_name: 目标知识库名称，缺省使用默认集合
            enable_rerank: 是否启用重排序，缺省取配置
            rerank_top_n: 重排序返回片段数，缺省取配置
            session_id: 会话 ID，用于多轮上下文记忆

        Yields:
            逐块生成的 SSE 事件字典，包含 content、sources、done 三类事件

        Raises:
            EmbeddingError: 查询文本向量化失败时抛出
            VectorStoreError: 向量检索执行失败时抛出
        """
        history = self._get_session_history(session_id)

        try:
            # 相似度阈值兜底：未指定时使用 0.5 默认值，过滤低质量无关召回
            if similarity_threshold <= 0:
                similarity_threshold = 0.5

            #  执行检索、重排、Prompt组装
            final_prompt, filtered_results = self.prepare_query_context(
                user_question=user_question,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                collection_name=collection_name,
                enable_rerank=enable_rerank,
                rerank_top_n=rerank_top_n,
                session_id=session_id
            )
        except (EmbeddingError, VectorStoreError) as e:
            logger.error(f"检索前置流程失败: {str(e)}")
            yield {"type": "content", "data": "抱歉，知识库检索服务异常，请稍后重试。"}
            yield {"type": "done", "data": {}}
            return

        # 零召回兜底：无匹配片段直接返回提示，不调用大模型
        if not filtered_results:
            empty_answer = "抱歉，未检索到与您问题相关的政策内容，请调整提问关键词后重试。"
            logger.info(f"流式问答零召回，问题: {user_question}")
            yield {"type": "content", "data": empty_answer}
            yield {"type": "sources", "data": []}
            yield {"type": "done", "data": {}}
            # 空结果也写入会话历史，保证多轮上下文连续性
            self._append_session_history(session_id, user_question, empty_answer)
            return

        # 大模型流式生成，加异常捕获避免中断
        full_answer = ""
        try:
            self.llm_client.clear_history()
            self.llm_client.messages[0]["content"] = final_prompt

            for chunk in self.llm_client.stream_chat(prompt=user_question):
                full_answer += chunk
                yield {"type": "content", "data": chunk}

        except Exception as e:
            logger.error(f"大模型流式生成失败，问题: {user_question}，错误: {str(e)}")
            error_tip = "\n\n服务异常，生成中断，请稍后重试。"
            full_answer += error_tip
            yield {"type": "content", "data": error_tip}
            yield {"type": "done", "data": {}}
            # 中断结果也写入历史，保证上下文不混乱
            self._append_session_history(session_id, user_question, full_answer)
            return

        # 流结束：返回引用来源和结束事件
        is_empty_answer = "暂无与该问题相关" in full_answer or "未检索到与您问题相关" in full_answer
        if return_sources and not is_empty_answer:
            yield {"type": "sources", "data": filtered_results}
        else:
            yield {"type": "sources", "data": []}
        yield {"type": "done", "data": {}}

        # 保存本轮完整对话到会话历史
        self._append_session_history(session_id, user_question, full_answer)
        logger.info(f"流式RAG问答完成，命中{len(filtered_results)}条片段")
