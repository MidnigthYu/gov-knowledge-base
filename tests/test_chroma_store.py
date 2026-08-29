"""
Chroma 向量库单元测试
验证文档入库计数、检索返回格式、空检索与集合删除等核心行为
依赖：unittest、ChromaVectorStore、VectorStoreError
"""
import unittest
from app.vector_store import ChromaVectorStore
from app.common.exceptions import VectorStoreError

class TestChromaStore(unittest.TestCase):

    def setUp(self):
        self.store = ChromaVectorStore(collection_name="test_unit", persist=False)

    def tearDown(self):
        """每个测试结束后销毁集合，保证测试间数据完全隔离"""
        try:
            self.store.delete_collection()
        except Exception:
            pass

    def test_add_and_count(self):
        texts = ["测试文本1", "测试文本2"]
        embeddings = [[0.1]*1024, [0.2]*1024]
        self.store.add_documents(texts, embeddings)
        self.assertEqual(self.store.count(), 2)

    def test_search_return_format(self):
        texts = ["武汉惠企政策", "南京人才补贴"]
        embeddings = [[0.1]*1024, [0.9]*1024]
        self.store.add_documents(texts, embeddings)
        results = self.store.search([0.15]*1024, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("content", results[0])
        self.assertIn("similarity", results[0])

    def test_empty_search(self):
        results = self.store.search([0.1]*1024, top_k=5)
        self.assertEqual(len(results), 0)

    def test_delete_collection(self):
        self.store.add_documents(["test"], [[0.1]*1024])
        self.assertEqual(self.store.count(), 1)
        self.store.delete_collection()
        new_store = ChromaVectorStore(collection_name="test_unit", persist=False)
        self.assertEqual(new_store.count(), 0)

    def test_delete_by_ids_empty_list(self):
        """空ID列表调用静默跳过，数据量不变"""
        texts = ["测试文本1", "测试文本2"]
        embeddings = [[0.1]*1024, [0.2]*1024]
        self.store.add_documents(texts, embeddings, metadatas=[{"source": "test"}]*2)
        original_count = self.store.count()

        self.store.delete_by_ids([])
        self.assertEqual(self.store.count(), original_count)

    def test_delete_by_ids_existing_ids(self):
        """删除存在的文档ID，数量正确减少"""
        doc_ids = ["test_id_1", "test_id_2"]
        texts = ["文本1", "文本2"]
        embeddings = [[0.1]*1024, [0.2]*1024]
        self.store.add_documents(texts, embeddings, metadatas=[{"source": "test"}]*2, ids=doc_ids)
        self.assertEqual(self.store.count(), 2)

        self.store.delete_by_ids(["test_id_1"])
        self.assertEqual(self.store.count(), 1)

    def test_delete_by_ids_nonexistent_collection(self):
        """删除不存在的集合，抛出 VectorStoreError"""
        with self.assertRaises(VectorStoreError):
            self.store.delete_by_ids(["any_id"], collection_name="not_exist_collection")

    def test_delete_by_ids_nonexistent_ids(self):
        """删除不存在的ID，幂等不报错，数据量不变"""
        texts = ["测试文本"]
        embeddings = [[0.1]*1024]
        self.store.add_documents(texts, embeddings, metadatas=[{"source": "test"}])
        original_count = self.store.count()

        self.store.delete_by_ids(["nonexistent_id"])
        self.assertEqual(self.store.count(), original_count)


if __name__ == "__main__":
    unittest.main()