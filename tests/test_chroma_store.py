import unittest
from vector_store import ChromaVectorStore

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


if __name__ == "__main__":
    unittest.main()