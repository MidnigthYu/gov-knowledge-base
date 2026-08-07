import unittest
import os
import tempfile
from unittest.mock import MagicMock
from service.knowledge_manager import KnowledgeManager

class TestKnowledgeManager(unittest.TestCase):
    """知识库批量管理单元测试"""

    def setUp(self):
        self.mock_embedding = MagicMock()
        self.mock_vector_store = MagicMock()
        self.manager = KnowledgeManager(
            embedding_client=self.mock_embedding,
            vector_store=self.mock_vector_store,
            chunk_size=100,
            chunk_overlap=20
        )
    def test_empty_dir_return_zero(self):
        """空目录返回零值统计，不报错"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.manager.build_from_dir(temp_dir)
            self.assertEqual(result["total_files"], 0)
            self.assertEqual(result["total_chunks"], 0)
            self.assertEqual(result["success_files"], 0)

    def test_normal_files_build_success(self):
        """正常文件批量构建成功，分块数量符合预期"""
        with tempfile.TemporaryDirectory() as temp_dir:
            
            file1 = os.path.join(temp_dir, "policy1.txt")
            file2 = os.path.join(temp_dir, "policy2.txt")
            with open(file1, "w", encoding="utf-8") as f:
                f.write("小微企业所得税优惠政策内容测试" * 20)
            with open(file2, "w", encoding="utf-8") as f:
                f.write("个体工商户补贴政策说明" * 20)

            # Mock 嵌入与入库
            self.mock_embedding.batch_embed.return_value = [[0.1] * 2048] * 10
            self.mock_vector_store.add_documents.return_value = None

            result = self.manager.build_from_dir(temp_dir)
            self.assertEqual(result["total_files"], 2)
            self.assertEqual(result["success_files"], 2)
            self.assertEqual(result["failed_files"], 0)
            self.assertGreater(result["total_chunks"], 0)

            self.mock_vector_store.add_documents.assert_called_once()

    def test_corrupt_file_exception_isolated(self):
        """异常文件不中断任务，计入失败统计"""
        with tempfile.TemporaryDirectory() as temp_dir:
            normal_file = os.path.join(temp_dir, "normal.txt")
            with open(normal_file, "w", encoding="utf-8") as f:
                f.write("正常政策文本内容" * 10)
            
            bad_path = os.path.join(temp_dir, "bad.txt")
            os.mkdir(bad_path)

            self.mock_embedding.batch_embed.return_value = [[0.1] * 2048] * 5
            result = self.manager.build_from_dir(temp_dir)

            self.assertEqual(result["total_files"], 1)
            self.assertEqual(result["success_files"], 1)

if __name__ == "__main__":
    unittest.main()