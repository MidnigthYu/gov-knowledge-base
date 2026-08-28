import requests
import streamlit as st
from config import BASE_URL, DEFAULT_COLLECTION, REQUEST_TIMEOUT

# ========== 通用工具函数 ==========
# 生成统一请求头，自动携带API Key鉴权
def get_headers():
    headers = {"Content-Type": "application/json"}
    if st.session_state.api_key and st.session_state.api_key.strip():
        headers["X-API-Key"] = st.session_state.api_key.strip()
    return headers

# 调用RAG问答接口
def ask_question(question, collection_name, enable_rerank=True):
    payload = {
        "question": question,
        "collection_name": collection_name,
        "enable_rerank": enable_rerank
    }
    res = requests.post(
        f"{BASE_URL}/api/qa/query",
        headers=get_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT
    )
    return res.json()

# ========== 统一渲染引用来源函数 ==========
def render_sources(sources: list):
    """统一渲染引用来源列表，完全保留原有UI样式，仅修复相似数字段名"""
    if not sources:
        return
    with st.expander("📑 查看引用来源"):
        for idx, source in enumerate(sources, 1):
            st.markdown(f"**来源 {idx}**")
            st.text(source.get("content", "无内容"))
            st.caption(f"相似度：{source.get('similarity', 0):.4f}")
            st.divider()

# ========== 页面基础配置 ==========
st.set_page_config(
    page_title="政务知识库RAG系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 隐藏Streamlit默认菜单和页脚
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ========== 初始化全局会话状态 ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 聊天历史记录
if "current_kb" not in st.session_state:
    st.session_state.current_kb = DEFAULT_COLLECTION  # 当前选中的知识库
if "kb_list" not in st.session_state:
    st.session_state.kb_list = []  # 知识库列表缓存
if "api_key" not in st.session_state:
    st.session_state.api_key = ""  # API鉴权密钥

# ========== 页面布局划分 ==========
# 左侧边栏：知识库管理区
with st.sidebar:
    st.title("📚 知识库管理")
    st.divider()

    # API密钥输入
    st.subheader("接口鉴权")
    st.session_state.api_key = st.text_input(
        "API密钥",
        type="password",
        placeholder="无鉴权可直接留空"
    )
    st.divider()

    # 知识库列表管理
    st.subheader("知识库列表")

    # 刷新列表按钮
    if st.button("🔄 刷新知识库列表", use_container_width=True):
        try:
            res = requests.get(
                f"{BASE_URL}/api/knowledge/list",
                headers=get_headers(),
                timeout=10
            )
            result = res.json()
            if result["code"] == 0:
                st.session_state.kb_list = result["data"]["collections"]
                st.success("列表刷新成功")
                # 自动选中默认知识库
                if st.session_state.current_kb not in st.session_state.kb_list and st.session_state.kb_list:
                    st.session_state.current_kb = st.session_state.kb_list[0]
                st.rerun()
            else:
                st.error(f"获取失败：{result['message']}")
        except Exception as e:
            st.error(f"连接后端失败：{str(e)}")

    # 知识库下拉选择器
    if st.session_state.kb_list:
        st.session_state.current_kb = st.selectbox(
            "当前使用知识库",
            options=st.session_state.kb_list,
            index=st.session_state.kb_list.index(st.session_state.current_kb)
            if st.session_state.current_kb in st.session_state.kb_list else 0
        )
    else:
        st.warning("暂无知识库，请点击上方按钮刷新")
    st.divider()

    # 文档上传入库
    st.subheader("文档上传入库")
    uploaded_file = st.file_uploader(
        "选择文档",
        type=["txt", "md", "pdf", "docx"],
        help="支持 PDF / Word / TXT / Markdown 格式"
    )

    if uploaded_file is not None:
        if st.button("📤 上传并入库", use_container_width=True, type="primary"):
            with st.spinner("正在解析文档并入库..."):
                try:
                    # 构造form-data格式请求，和后端接口对齐
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    form_data = {"collection_name": st.session_state.current_kb}
                    # 上传接口不用Content-Type，让requests自动生成boundary
                    headers = {}
                    if st.session_state.api_key.strip():
                        headers["X-API-Key"] = st.session_state.api_key.strip()

                    res = requests.post(
                        f"{BASE_URL}/api/knowledge/upload",
                        headers=headers,
                        files=files,
                        data=form_data,
                        timeout=REQUEST_TIMEOUT
                    )
                    result = res.json()
                    if result["code"] == 0:
                        st.success("文档入库成功！")
                        st.rerun()  # 入库成功自动刷新页面
                    else:
                        st.error(f"入库失败：{result['message']}")
                except Exception as e:
                    st.error(f"上传异常：{str(e)}")
    st.divider()

    # 知识库操作区
    st.subheader("快捷操作")

    if st.button("🗑️ 删除当前知识库", use_container_width=True):
        if "confirm_delete" not in st.session_state:
            st.session_state.confirm_delete = False

        if not st.session_state.confirm_delete:
            st.warning("⚠️ 删除后数据不可恢复，再次点击确认删除")
            st.session_state.confirm_delete = True
        else:
            with st.spinner("删除中..."):
                try:
                    res = requests.delete(
                        f"{BASE_URL}/api/knowledge/{st.session_state.current_kb}",
                        headers=get_headers(),
                        timeout=10
                    )
                    result = res.json()
                    if result["code"] == 0:
                        st.success("删除成功")
                        st.session_state.confirm_delete = False
                        st.session_state.kb_list = []
                        st.rerun()
                    else:
                        st.error(f"删除失败：{result['message']}")
                        st.session_state.confirm_delete = False
                except Exception as e:
                    st.error(f"删除异常：{str(e)}")
                    st.session_state.confirm_delete = False

    if st.button("🧹 清空当前对话", use_container_width=True):
        st.session_state.chat_history = []
        st.success("对话已清空")
        st.rerun()

# 右侧主区域：聊天对话区
st.title("🤖 政务知识库智能问答")
st.caption(f"当前知识库：{st.session_state.current_kb}")
st.divider()

chat_container = st.container()

with chat_container:
    # 空对话时显示欢迎语
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown("👋 您好，我是政务知识库智能助手。")
            st.markdown("您可以在左侧上传政务文档，然后向我提问，我会基于文档内容为您精准解答。")
    
    # 渲染历史聊天记录
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
            # 展示引用来源
            render_sources(msg.get("sources", []))


# 底部输入框
user_input = st.chat_input("请输入您的问题...")

if user_input:
    # 用户消息加入历史
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    # 刷新页面立即显示用户消息
    st.rerun()

# ========== 自动触发回答生成 ==========
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    last_question = st.session_state.chat_history[-1]["content"]
    
    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("正在检索知识库并生成回答..."):
                try:
                    result = ask_question(
                        last_question,
                        st.session_state.current_kb
                    )
                    
                    if result["code"] == 0:
                        answer = result["data"]["answer"]
                        sources = result["data"].get("sources", [])
                        
                        # 渲染回答
                        st.markdown(answer)
                        
                        # 渲染引用来源
                        render_sources(sources)
                      
                        # 存入历史记录
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        st.rerun()
                    else:
                        error_msg = f"抱歉，回答生成失败：{result['message']}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                        st.rerun()
                        
                except Exception as e:
                    error_msg = f"连接后端服务失败，请检查服务是否启动。错误信息：{str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    st.rerun()