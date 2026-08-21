import pytest, uuid
from fastapi.testclient import TestClient
from app.main import app

# ========== 测试配置 ==========
TEST_COLLECTION = "test-gov-regression"
TEST_DOCS_DIR = "/app/data/docs"
EXPECTED_TOTAL_FILES = 3
EXPECTED_MIN_CHUNKS = 5
EXPECTED_MAX_CHUNKS = 10
QA_HIT_MIN_SCORE = 0.3        
QA_ANSWER_MIN_LEN = 20
KEYWORD_MUST_HIT = "补贴"

# 业务错误码
CODE_PARAM_ERROR = 1001
CODE_VECTOR_SEARCH_ERROR = 4001
CODE_COLLECTION_NOT_EXIST = 2001

@pytest.fixture(scope="module")
def client():
    """模块级测试客户端，统一注入API鉴权头"""
    with TestClient(app) as c:
        c.headers.update({"X-API-Key": "test_key_123"})
        yield c


# ========== 环境清理 ==========
def test_00_clean_environment(client):
    """前置清理：删除测试集合，保证环境幂等"""
    response = client.delete(f"/api/knowledge/{TEST_COLLECTION}")
    # 删除成功或集合不存在均视为通过（幂等）
    assert response.status_code == 200
    assert response.json()["code"] in (0, 2001)


# ========== 知识库构建阈值测试 ==========
def test_01_build_success_rate(client):
    """构建成功率阈值：100%文件入库成功，0失败"""
    payload = {
        "docs_dir": TEST_DOCS_DIR,
        "collection_name": TEST_COLLECTION
    }
    response = client.post("/api/knowledge/build", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["code"] == 0
    assert data["data"]["total_files"] == EXPECTED_TOTAL_FILES
    assert data["data"]["success_files"] == EXPECTED_TOTAL_FILES
    assert data["data"]["failed_files"] == 0
    assert len(data["data"]["failed_details"]) == 0


def test_02_chunk_count_threshold(client):
    """分块数量阈值：在合理区间内，无异常分块"""
    payload = {
        "docs_dir": TEST_DOCS_DIR,
        "collection_name": TEST_COLLECTION
    }
    response = client.post("/api/knowledge/build", json=payload)
    data = response.json()
    total_chunks = data["data"]["total_chunks"]

    assert EXPECTED_MIN_CHUNKS <= total_chunks <= EXPECTED_MAX_CHUNKS
    # 单文档分块数不超过4块，防止分块粒度异常
    assert total_chunks / EXPECTED_TOTAL_FILES <= 4


# ========== 向量库状态阈值校验 ==========
def test_03_collection_exists(client):
    """向量库集合存在性校验：构建后集合可正常访问"""
    # 通过问答接口间接验证集合可用（不暴露底层向量库接口）
    payload = {"question": "测试", "collection_name": TEST_COLLECTION}
    response = client.post("/api/qa/query", json=payload)
    assert response.status_code == 200
    # 无集合不存在异常码
    assert response.json()["code"] != 4001


# ========== 问答召回命中阈值测试 ==========
def test_04_qa_recall_hit_rate(client):
    """召回命中率：已知政务问题必须命中对应片段"""
    payload = {
        "question": "技能提升补贴的申请条件是什么",
        "collection_name": TEST_COLLECTION
    }
    response = client.post("/api/qa/query", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["code"] == 0
    assert len(data["data"]["sources"]) >= 1
    assert KEYWORD_MUST_HIT in data["data"]["answer"]
    assert len(data["data"]["answer"]) >= QA_ANSWER_MIN_LEN


def test_05_qa_rerank_score_threshold(client):
    """Rerank得分阈值：Top1片段得分≥最低召回阈值"""
    payload = {
        "question": "小微企业所得税优惠政策",
        "collection_name": TEST_COLLECTION
    }
    response = client.post("/api/qa/query", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["code"] == 0
    chunks = data["data"]["sources"]
    if chunks:
        top_score = chunks[0].get("similarity", 1.0)
        assert top_score >= QA_HIT_MIN_SCORE


# ========== 低相关兜底阈值测试 ==========
def test_06_low_relevance_fallback(client):
    """低相关兜底：无关问题触发兜底提示，不编造答案"""
    payload = {
        "question": "今天的股票行情怎么样",
        "collection_name": TEST_COLLECTION
    }
    response = client.post("/api/qa/query", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["code"] == 0
    assert "暂无与该问题相关的政策信息" in data["data"]["answer"]
    assert len(data["data"]["sources"]) == 0

# ========== 边界异常测试 ==========
def test_07_empty_question(client):
    """边界：空问题返回参数校验错误码1001"""
    payload = {"question": "", "collection_name": TEST_COLLECTION}
    response = client.post("/api/qa/query", json=payload)
    data = response.json()
    assert data["code"] == CODE_PARAM_ERROR


def test_08_nonexistent_collection(client):
    """边界：不存在的集合返回检索失败错误码4001"""
    nonexist_collection = f"not-exist-{uuid.uuid4().hex[:12]}"
    payload = {"question": "测试问题", "collection_name": nonexist_collection}
    response = client.post("/api/qa/query", json=payload)
    data = response.json()
    assert data["code"] == CODE_VECTOR_SEARCH_ERROR


# ========== 后置清理 ==========
def test_99_cleanup_collection(client):
    """测试完成后清理测试集合，保持环境干净"""
    response = client.delete(f"/api/knowledge/{TEST_COLLECTION}")
    assert response.status_code == 200
    assert response.json()["code"] in (0, 2001)